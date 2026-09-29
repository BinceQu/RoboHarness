"""Fail-closed GPU certificate that a frozen wall map has no compatible pose.

This module does not localize the robot and cannot authorize map writes.  It
answers a narrower question: under an explicit static-wall support model, is
the *maximum possible* support anywhere in a frozen map too small for the
current structural scan to be a revisit?

The exhaustive search is a CUDA circular cross-correlation over every map
cell and a complete 360 degree yaw lattice.  Circular wrap, wall dilation and
an FFT rounding guard can only increase the reported support, so a negative
result remains conservative.  Known-free conflicts are deliberately ignored:
furniture may move and an old free observation is not hard evidence that a
current wall observation is novel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import math
import os
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from behavior_interface.geometry_localization import GridSpec, wrap_deg


GLOBAL_NEGATIVE_DEVICE_ENV = "BEHAVIOR_SLAM_GEOMETRY_DEVICE"


@dataclass(frozen=True)
class GlobalNegativeConfig:
    """Only sensor, grid and proof-confidence scales.

    ``minimum_static_wall_fraction`` is the sole environmental assumption in
    the negative result.  At its default value, more than 70 percent of a rich
    structural scan may be dynamic, missing or corrupted before novelty can
    be certified.  Callers should retain this value in logs with the result.
    """

    proof_resolution_m: float = 0.05
    yaw_step_deg: float = 1.0
    intrinsic_wall_tolerance_m: float = 0.12
    minimum_static_wall_fraction: float = 0.30
    minimum_source_wall_cells: int = 48
    minimum_target_wall_cells: int = 24
    minimum_source_extent_m: float = 1.00
    maximum_source_radius_m: float = 4.00
    direction_pair_min_m: float = 0.25
    direction_pair_max_m: float = 0.70
    direction_pair_min_repetitions: int = 3
    minimum_direction_pair_weight: float = 24.0
    minimum_direction_minor_ratio: float = 0.15
    yaw_batch_size: int = 8
    plausible_mode_separation_m: float = 0.50
    plausible_mode_separation_yaw_deg: float = 7.5
    diagnostic_modes_per_yaw: int = 3
    diagnostic_mode_limit: int = 8
    # cuFFT correlation of sparse integer masks is numerically very close to
    # an integer.  This guard is intentionally much larger than the usual
    # O(eps * log(grid_size) * source_count) error and is rounded upward again
    # before comparison, so floating-point error can only turn NEGATIVE into
    # UNKNOWN rather than manufacture a negative.
    fft_rounding_guard_fraction: float = 5.0e-3
    # These are hard fail-closed resource guards, never search truncation.
    # A configured budget smaller than the complete lattice returns UNKNOWN.
    maximum_yaw_bins: Optional[int] = None
    maximum_translation_cells: Optional[int] = None


@dataclass(frozen=True)
class FrozenWallTarget:
    """Immutable snapshot of historical wall points.

    Bytes, rather than a NumPy array, prevent a live occupancy buffer from
    changing beneath a recovery proof.  ``generation`` is diagnostic only;
    integrity is established by ``digest_sha256``.
    """

    wall_xy_f32: bytes
    point_count: int
    digest_sha256: str
    generation: str = ""
    finite: bool = True

    def wall_points(self) -> np.ndarray:
        return np.frombuffer(self.wall_xy_f32, dtype="<f4").reshape(-1, 2)


@dataclass(frozen=True)
class GlobalNegativeCertificate:
    """Structured result; it intentionally never grants a map write."""

    status: str
    reason: str
    novelty_certified: bool
    map_write_authorized: bool
    target_digest_sha256: str
    target_generation: str
    target_frozen: bool
    backend: str
    device: str
    cpu_fallback: bool
    search_complete: bool
    candidate_budget_truncated: bool
    source_wall_cells: int = 0
    target_wall_cells: int = 0
    source_extent_m: float = 0.0
    source_direction_minor_ratio: float = 0.0
    source_direction_pair_weight: float = 0.0
    source_2d_observable: bool = False
    upper_bound_supported_cells: int = 0
    upper_bound_support_fraction: float = 1.0
    required_static_support_cells: int = 0
    minimum_static_wall_fraction: float = 0.0
    proof_resolution_m: float = 0.0
    yaw_step_deg: float = 0.0
    yaw_bins_evaluated: int = 0
    translation_cells_evaluated: int = 0
    wall_dilation_cells: int = 0
    wall_dilation_m: float = 0.0
    plausible_mode_count_lower_bound: int = 0
    plausible_modes: Tuple[Tuple[float, float, float, float], ...] = ()
    hard_evidence: Tuple[str, ...] = ("source_wall", "frozen_target_wall")
    ignored_as_hard_evidence: Tuple[str, ...] = (
        "target_free_conflict",
        "dynamic_furniture",
        "odometry_distance",
    )
    conditional_assumptions: Tuple[str, ...] = (
        "source_points_share_one_rigid_coordinate_frame",
        "configured_fraction_of_source_points_are_static_structural_walls",
        "frozen_target_contains_all_previously_accepted_wall_evidence",
    )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _spatial_yaw_nms(
    rows: Iterable[Tuple[float, float, float, float]],
    *,
    count: int,
    separation_m: float,
    separation_yaw_deg: float,
) -> list[Tuple[float, float, float, float]]:
    selected: list[Tuple[float, float, float, float]] = []
    for row in sorted(rows, key=lambda item: item[3], reverse=True):
        if all(
            math.hypot(row[0] - old[0], row[1] - old[1])
            >= float(separation_m)
            or abs(wrap_deg(row[2] - old[2]))
            >= float(separation_yaw_deg)
            for old in selected
        ):
            selected.append(row)
        if len(selected) >= max(1, int(count)):
            break
    return selected


class GpuGlobalNegativeProver:
    """Exhaustively upper-bound frozen-wall support on CUDA."""

    def __init__(
        self,
        grid: GridSpec,
        *,
        config: GlobalNegativeConfig = GlobalNegativeConfig(),
        device: Optional[str] = None,
    ) -> None:
        self.grid = grid
        self.config = config
        self.device_name = str(
            device or os.environ.get(GLOBAL_NEGATIVE_DEVICE_ENV, "cuda:0")
        )
        self.last_backend_reason = "not_checked"

    def backend_info(self) -> Dict[str, Any]:
        return {
            "backend": "torch_cuda_exhaustive_wall_upper_bound",
            "device": self.device_name,
            "cpu_fallback": False,
            "last_reason": self.last_backend_reason,
        }

    def freeze_target(
        self,
        wall_points: np.ndarray,
        *,
        generation: str = "",
    ) -> FrozenWallTarget:
        values = np.asarray(wall_points, dtype="<f4").reshape(-1, 2)
        snapshot = np.ascontiguousarray(values).copy()
        payload = snapshot.tobytes(order="C")
        header = (
            f"{self.grid.resolution_m:.12g}|{self.grid.half_span_m:.12g}|"
            f"{len(snapshot)}|"
        ).encode("ascii")
        return FrozenWallTarget(
            wall_xy_f32=payload,
            point_count=int(len(snapshot)),
            digest_sha256=hashlib.sha256(header + payload).hexdigest(),
            generation=str(generation),
            finite=bool(np.isfinite(snapshot).all()),
        )

    def _target_integrity(self, target: FrozenWallTarget) -> bool:
        if len(target.wall_xy_f32) != 8 * int(target.point_count):
            return False
        header = (
            f"{self.grid.resolution_m:.12g}|{self.grid.half_span_m:.12g}|"
            f"{int(target.point_count)}|"
        ).encode("ascii")
        digest = hashlib.sha256(header + target.wall_xy_f32).hexdigest()
        if not hmac.compare_digest(digest, str(target.digest_sha256)):
            return False
        try:
            return bool(np.isfinite(target.wall_points()).all()) == bool(target.finite)
        except Exception:
            return False

    def _certificate(
        self,
        target: FrozenWallTarget,
        *,
        status: str = "UNKNOWN",
        reason: str,
        **updates: Any,
    ) -> GlobalNegativeCertificate:
        base: Dict[str, Any] = {
            "status": str(status),
            "reason": str(reason),
            "novelty_certified": str(status) == "NEGATIVE",
            "map_write_authorized": False,
            "target_digest_sha256": target.digest_sha256,
            "target_generation": target.generation,
            "target_frozen": True,
            "backend": "torch_cuda_exhaustive_wall_upper_bound",
            "device": self.device_name,
            "cpu_fallback": False,
            "search_complete": False,
            "candidate_budget_truncated": False,
            "minimum_static_wall_fraction": float(
                self.config.minimum_static_wall_fraction
            ),
            "proof_resolution_m": float(self.config.proof_resolution_m),
        }
        base.update(updates)
        return GlobalNegativeCertificate(**base)

    def _lattice(self) -> Tuple[Optional[Tuple[int, int, float]], str]:
        resolution = float(self.config.proof_resolution_m)
        if not math.isfinite(resolution) or resolution <= 0.0:
            return None, "invalid_proof_resolution"
        exact_size = 2.0 * float(self.grid.half_span_m) / resolution
        size = int(round(exact_size))
        if size < 8 or abs(float(size) - exact_size) > 1.0e-6:
            return None, "proof_grid_does_not_tile_map"
        requested_yaw_step = float(self.config.yaw_step_deg)
        if not math.isfinite(requested_yaw_step) or requested_yaw_step <= 0.0:
            return None, "invalid_yaw_step"
        yaw_bins = max(1, int(math.ceil(360.0 / requested_yaw_step)))
        yaw_step = 360.0 / float(yaw_bins)
        return (size, yaw_bins, yaw_step), "complete_lattice"

    def _cuda(self) -> Tuple[Optional[Any], Optional[Any], str]:
        try:
            import torch
        except Exception as exc:
            return None, None, f"torch_import_failed:{type(exc).__name__}"
        try:
            device = torch.device(self.device_name)
            if device.type != "cuda":
                return None, None, "non_cuda_device_rejected"
            if not torch.cuda.is_available():
                return None, None, "cuda_unavailable"
            torch.empty(1, device=device)
            return torch, device, "cuda_ready"
        except Exception as exc:
            return None, None, f"cuda_device_failed:{type(exc).__name__}"

    @staticmethod
    def _unique_input_cells(
        points: Any,
        *,
        cell_m: float,
        torch: Any,
    ) -> Any:
        bins = torch.round(points / float(cell_m)).to(dtype=torch.int64)
        unique_bins = torch.unique(bins, dim=0)
        return unique_bins.to(dtype=torch.float32) * float(cell_m)

    def _source_observability(
        self,
        source_offsets: Any,
        *,
        size: int,
        torch: Any,
        device: Any,
    ) -> Tuple[bool, float, float]:
        """Wall-normal information from repeated medium-baseline cell pairs."""

        resolution = float(self.config.proof_resolution_m)
        offsets = torch.round(source_offsets / resolution).to(torch.int64)
        mask = torch.zeros((size, size), device=device, dtype=torch.float32)
        mask[offsets[:, 1] % size, offsets[:, 0] % size] = 1.0
        spectrum = torch.fft.rfft2(mask)
        autocorrelation = torch.fft.irfft2(
            torch.conj(spectrum) * spectrum,
            s=(size, size),
        ).clamp_min(0.0)

        maximum_cells = int(math.floor(
            float(self.config.direction_pair_max_m) / resolution
        ))
        minimum_cells = int(math.ceil(
            float(self.config.direction_pair_min_m) / resolution
        ))
        if maximum_cells < minimum_cells or maximum_cells <= 0:
            return False, 0.0, 0.0
        axis = torch.arange(
            -maximum_cells, maximum_cells + 1,
            device=device,
            dtype=torch.int64,
        )
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        distance_cells = torch.sqrt(
            xx.to(torch.float32) ** 2 + yy.to(torch.float32) ** 2
        )
        half_plane = (yy > 0) | ((yy == 0) & (xx > 0))
        usable = (
            half_plane
            & (distance_cells >= float(minimum_cells))
            & (distance_cells <= float(maximum_cells))
        )
        dx = xx[usable]
        dy = yy[usable]
        lengths = distance_cells[usable].clamp_min(1.0)
        pair_counts = autocorrelation[dy % size, dx % size]
        weights = (
            pair_counts
            - float(max(0, int(self.config.direction_pair_min_repetitions) - 1))
        ).clamp_min(0.0) * lengths
        total_weight = float(weights.sum().item())
        if total_weight < float(self.config.minimum_direction_pair_weight):
            return False, 0.0, total_weight

        tangent_x = dx.to(torch.float32) / lengths
        tangent_y = dy.to(torch.float32) / lengths
        normal_x = -tangent_y
        normal_y = tangent_x
        information = torch.stack((
            torch.sum(weights * normal_x * normal_x),
            torch.sum(weights * normal_x * normal_y),
            torch.sum(weights * normal_y * normal_x),
            torch.sum(weights * normal_y * normal_y),
        )).reshape(2, 2)
        eigenvalues = torch.linalg.eigvalsh(information)
        ratio = float(
            (eigenvalues[0] / eigenvalues[1].clamp_min(1.0e-8)).item()
        )
        return (
            ratio >= float(self.config.minimum_direction_minor_ratio),
            ratio,
            total_weight,
        )

    def _dilation_cells(
        self,
        *,
        source_radius_m: float,
        yaw_step_deg: float,
    ) -> Tuple[int, float]:
        resolution = float(self.config.proof_resolution_m)
        yaw_half = math.radians(0.5 * float(yaw_step_deg))
        yaw_chord = 2.0 * float(source_radius_m) * math.sin(0.5 * yaw_half)
        # Per-axis bound: target raster, rotated-source raster and nearest
        # translation anchor each contribute at most half a proof cell.
        bound_m = (
            float(self.config.intrinsic_wall_tolerance_m)
            + 1.5 * resolution
            + yaw_chord
        )
        cells = max(0, int(math.ceil(bound_m / resolution)))
        return cells, float(cells) * resolution

    def prove(
        self,
        source_wall: np.ndarray,
        target: FrozenWallTarget,
        *,
        pivot: Tuple[float, float],
        source_free: Optional[np.ndarray] = None,
        target_free: Optional[np.ndarray] = None,
    ) -> GlobalNegativeCertificate:
        """Return NEGATIVE only after a complete conservative CUDA search.

        The free arrays are accepted so callers can use one geometry bundle,
        but are intentionally not read.  They cannot strengthen a negative.
        """

        del source_free, target_free
        if not self._target_integrity(target):
            return self._certificate(
                target,
                reason="frozen_target_integrity_failed",
                target_frozen=False,
            )
        lattice, lattice_reason = self._lattice()
        if lattice is None:
            return self._certificate(target, reason=lattice_reason)
        size, yaw_bins, yaw_step = lattice
        translations = int(size * size)
        if (
            self.config.maximum_yaw_bins is not None
            and yaw_bins > int(self.config.maximum_yaw_bins)
        ):
            return self._certificate(
                target,
                reason="yaw_budget_would_truncate_complete_search",
                candidate_budget_truncated=True,
            )
        if (
            self.config.maximum_translation_cells is not None
            and translations > int(self.config.maximum_translation_cells)
        ):
            return self._certificate(
                target,
                reason="translation_budget_would_truncate_complete_search",
                candidate_budget_truncated=True,
            )
        if not target.finite:
            return self._certificate(target, reason="nonfinite_frozen_target")
        try:
            source_values = np.asarray(source_wall, dtype=np.float32).reshape(-1, 2)
            pivot_values = np.asarray(pivot, dtype=np.float32).reshape(2)
        except Exception as exc:
            return self._certificate(
                target, reason=f"invalid_source_geometry:{type(exc).__name__}"
            )
        if not np.isfinite(source_values).all() or not np.isfinite(pivot_values).all():
            return self._certificate(target, reason="nonfinite_source_geometry")

        try:
            torch, device, backend_reason = self._cuda()
        except Exception as exc:
            self.last_backend_reason = f"cuda_backend_failed:{type(exc).__name__}"
            return self._certificate(target, reason=self.last_backend_reason)
        self.last_backend_reason = backend_reason
        if torch is None or device is None:
            return self._certificate(target, reason=backend_reason)

        try:
            certificate = self._prove_cuda(
                source_values,
                target,
                pivot_values,
                size=size,
                yaw_bins=yaw_bins,
                yaw_step=yaw_step,
                translations=translations,
                torch=torch,
                device=device,
            )
            torch.cuda.synchronize(device)
            return certificate
        except Exception as exc:
            self.last_backend_reason = f"cuda_search_failed:{type(exc).__name__}"
            return self._certificate(target, reason=self.last_backend_reason)

    def _prove_cuda(
        self,
        source_values: np.ndarray,
        target: FrozenWallTarget,
        pivot_values: np.ndarray,
        *,
        size: int,
        yaw_bins: int,
        yaw_step: float,
        translations: int,
        torch: Any,
        device: Any,
    ) -> GlobalNegativeCertificate:
        import torch.nn.functional as functional

        source_gpu = torch.from_numpy(source_values).to(device=device)
        pivot_gpu = torch.from_numpy(pivot_values).to(device=device)
        source_offsets = self._unique_input_cells(
            source_gpu - pivot_gpu,
            cell_m=float(self.grid.resolution_m),
            torch=torch,
        )
        source_count = int(source_offsets.shape[0])
        if source_count < int(self.config.minimum_source_wall_cells):
            return self._certificate(
                target,
                reason="insufficient_source_wall_support",
                source_wall_cells=source_count,
            )
        source_radius = float(torch.linalg.vector_norm(
            source_offsets, dim=1
        ).max().item())
        source_extent = float(torch.linalg.vector_norm(
            source_offsets.max(dim=0).values
            - source_offsets.min(dim=0).values
        ).item())
        if source_radius > float(self.config.maximum_source_radius_m):
            return self._certificate(
                target,
                reason="source_exceeds_sensor_support_model",
                source_wall_cells=source_count,
                source_extent_m=source_extent,
            )
        if source_extent < float(self.config.minimum_source_extent_m):
            return self._certificate(
                target,
                reason="insufficient_source_extent",
                source_wall_cells=source_count,
                source_extent_m=source_extent,
            )
        observable, direction_ratio, direction_weight = self._source_observability(
            source_offsets, size=size, torch=torch, device=device
        )
        common = {
            "source_wall_cells": source_count,
            "source_extent_m": source_extent,
            "source_direction_minor_ratio": direction_ratio,
            "source_direction_pair_weight": direction_weight,
            "source_2d_observable": observable,
        }
        if not observable:
            return self._certificate(
                target, reason="source_direction_unobservable", **common
            )

        target_values = np.asarray(target.wall_points(), dtype=np.float32)
        target_gpu = torch.from_numpy(target_values.copy()).to(device=device)
        half_span = float(self.grid.half_span_m)
        resolution = float(self.config.proof_resolution_m)
        target_columns = torch.floor(
            (target_gpu[:, 0] + half_span) / resolution
        ).to(torch.int64)
        target_rows = torch.floor(
            (target_gpu[:, 1] + half_span) / resolution
        ).to(torch.int64)
        target_inside = (
            (target_columns >= 0)
            & (target_rows >= 0)
            & (target_columns < size)
            & (target_rows < size)
        )
        if not bool(torch.all(target_inside).item()):
            return self._certificate(
                target,
                reason="frozen_target_outside_proof_extent",
                **common,
            )
        target_mask = torch.zeros(
            (size, size), device=device, dtype=torch.float32
        )
        target_mask[target_rows, target_columns] = 1.0
        target_count = int(target_mask.sum().item())
        if target_count < int(self.config.minimum_target_wall_cells):
            return self._certificate(
                target,
                reason="insufficient_frozen_target_wall_support",
                target_wall_cells=target_count,
                **common,
            )

        dilation_cells, dilation_m = self._dilation_cells(
            source_radius_m=source_radius, yaw_step_deg=yaw_step
        )
        if dilation_cells:
            target_upper = functional.max_pool2d(
                target_mask[None, None],
                kernel_size=2 * dilation_cells + 1,
                stride=1,
                padding=dilation_cells,
            )[0, 0]
        else:
            target_upper = target_mask
        target_spectrum = torch.fft.rfft2(target_upper)
        required_count = int(math.ceil(
            float(self.config.minimum_static_wall_fraction) * source_count
        ))
        numerical_guard = max(
            2.0,
            float(self.config.fft_rounding_guard_fraction) * source_count,
        )
        mode_rows: list[Tuple[float, float, float, float]] = []
        raw_maximum = 0.0
        batch_size = max(1, int(self.config.yaw_batch_size))

        with torch.inference_mode():
            for begin in range(0, yaw_bins, batch_size):
                indices = torch.arange(
                    begin, min(yaw_bins, begin + batch_size),
                    device=device,
                    dtype=torch.float32,
                )
                angles = torch.deg2rad(indices * float(yaw_step))
                cosine = torch.cos(angles)[:, None]
                sine = torch.sin(angles)[:, None]
                x = source_offsets[:, 0][None, :]
                y = source_offsets[:, 1][None, :]
                rotated_x = cosine * x - sine * y
                rotated_y = sine * x + cosine * y
                dx = torch.round(rotated_x / resolution).to(torch.int64)
                dy = torch.round(rotated_y / resolution).to(torch.int64)
                flat_indices = (dy % size) * size + (dx % size)
                kernels = torch.zeros(
                    (len(indices), size * size),
                    device=device,
                    dtype=torch.float32,
                )
                kernels.scatter_add_(
                    1, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32)
                )
                kernels = kernels.reshape(-1, size, size)
                source_spectrum = torch.fft.rfft2(kernels)
                correlations = torch.fft.irfft2(
                    torch.conj(source_spectrum) * target_spectrum[None, :, :],
                    s=(size, size),
                ).clamp_min(0.0)
                # Periodic wrap may add impossible edge-to-edge hits, never
                # remove an in-bounds hit.  Thus this scalar remains an upper
                # bound for every non-periodic translation.  Anchor decoding
                # below is diagnostic only and cannot create a NEGATIVE.
                raw_maximum = max(
                    raw_maximum, float(correlations.max().item())
                )

                available = correlations.reshape(len(indices), -1).clone()
                per_yaw = max(1, int(self.config.diagnostic_modes_per_yaw))
                separation_cells = max(1, int(math.ceil(
                    float(self.config.plausible_mode_separation_m) / resolution
                )))
                columns_grid = torch.arange(size, device=device)[None, :]
                rows_grid = torch.arange(size, device=device)[:, None]
                for _unused in range(per_yaw):
                    values, flat = available.max(dim=1)
                    for local_index in range(len(indices)):
                        conservative_value = float(values[local_index].item()) + numerical_guard
                        if conservative_value < float(required_count):
                            continue
                        flat_index = int(flat[local_index].item())
                        row = flat_index // size
                        column = flat_index % size
                        mode_rows.append((
                            -half_span + (float(column) + 0.5) * resolution,
                            -half_span + (float(row) + 0.5) * resolution,
                            wrap_deg(float(indices[local_index].item()) * yaw_step),
                            min(1.0, conservative_value / float(source_count)),
                        ))
                        nearby = (
                            (columns_grid - column) ** 2
                            + (rows_grid - row) ** 2
                        ) < separation_cells ** 2
                        available[local_index, nearby.reshape(-1)] = -torch.inf

        upper_count = min(
            source_count, int(math.ceil(raw_maximum + numerical_guard))
        )
        upper_fraction = float(upper_count) / float(source_count)
        plausible_modes = _spatial_yaw_nms(
            mode_rows,
            count=max(2, int(self.config.diagnostic_mode_limit)),
            separation_m=float(self.config.plausible_mode_separation_m),
            separation_yaw_deg=float(
                self.config.plausible_mode_separation_yaw_deg
            ),
        )
        result = {
            **common,
            "target_wall_cells": target_count,
            "upper_bound_supported_cells": upper_count,
            "upper_bound_support_fraction": upper_fraction,
            "required_static_support_cells": required_count,
            "yaw_step_deg": yaw_step,
            "yaw_bins_evaluated": yaw_bins,
            "translation_cells_evaluated": translations,
            "wall_dilation_cells": dilation_cells,
            "wall_dilation_m": dilation_m,
            "plausible_mode_count_lower_bound": len(plausible_modes),
            "plausible_modes": tuple(plausible_modes),
            "search_complete": True,
            "candidate_budget_truncated": False,
        }
        if upper_count < required_count:
            return self._certificate(
                target,
                status="NEGATIVE",
                reason="frozen_wall_support_exhaustively_below_model",
                **result,
            )
        if len(plausible_modes) >= 2:
            return self._certificate(
                target, reason="ambiguous_frozen_wall_modes", **result
            )
        return self._certificate(
            target, reason="possible_frozen_wall_match", **result
        )


__all__ = [
    "GLOBAL_NEGATIVE_DEVICE_ENV",
    "FrozenWallTarget",
    "GlobalNegativeCertificate",
    "GlobalNegativeConfig",
    "GpuGlobalNegativeProver",
]
