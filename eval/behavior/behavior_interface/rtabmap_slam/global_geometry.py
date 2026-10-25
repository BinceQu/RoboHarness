"""RTAB-Map-owned GPU coarse localization against a frozen occupancy map.

This module is deliberately a candidate generator, not a pose authority.  It
uses only frozen wall/free cells and a rolling depth scan already expressed in
the current odometry frame.  Every returned mode must still pass
``GpuGeometryValidator`` and the temporal pose bank before the RTAB-Map graph
may be changed.

The implementation follows the global-localization part of likelihood-field
AMCL, with two important safeguards for this interface:

* candidate translations are deterministically tiled over observed free space
  instead of hoping that a small local particle cloud covers the true mode;
* all separated high-scoring modes are returned, so an unrepresented repeated
  room cannot masquerade as low covariance.

All expensive orientation and scan scoring operations run on CUDA.  There is
no CPU scoring fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from behavior_interface.geometry_localization import GridSpec, unique_cells, wrap_deg


GLOBAL_GEOMETRY_DEVICE_ENV = "BEHAVIOR_SLAM_GEOMETRY_DEVICE"


@dataclass(frozen=True)
class GlobalGeometryConfig:
    """Sensor/grid-scale parameters; none are tied to a task or floor plan."""

    translation_step_m: float = 0.20
    orientation_bin_deg: float = 2.5
    orientation_peak_count: int = 12
    orientation_peak_separation_deg: float = 7.5
    maximum_source_wall_points: int = 1024
    maximum_source_free_points: int = 2048
    maximum_target_orientation_points: int = 2048
    orientation_neighbors: int = 16
    orientation_min_anisotropy: float = 0.35
    batch_size: int = 128
    top_per_yaw: int = 64
    peak_count: int = 8
    peak_separation_m: float = 0.50
    peak_separation_yaw_deg: float = 10.0
    robust_wall_keep_fraction: float = 0.70
    minimum_wall_points: int = 24


def _subsample(points: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(values) <= int(maximum):
        return values
    indices = np.linspace(0, len(values) - 1, int(maximum)).astype(np.int64)
    return values[indices]


def _wrapped_difference_deg(left: float, right: float) -> float:
    return abs(wrap_deg(float(left) - float(right)))


def nonmax_global_candidates(
    candidates: Iterable[Dict[str, float]],
    *,
    count: int,
    separation_m: float,
    separation_yaw_deg: float,
) -> list[Dict[str, float]]:
    """Keep spatially/orientationally distinct modes in score order.

    A lower-scoring separated mode is evidence of ambiguity, not clutter to be
    discarded.  This is the key distinction from estimating covariance only
    inside one already-collapsed particle cloud.
    """

    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: float(row["score"]),
        reverse=True,
    )
    selected: list[Dict[str, float]] = []
    for row in ordered:
        if all(
            math.hypot(
                float(row["x_m"]) - float(old["x_m"]),
                float(row["y_m"]) - float(old["y_m"]),
            ) >= float(separation_m)
            or _wrapped_difference_deg(
                float(row["yaw_correction_deg"]),
                float(old["yaw_correction_deg"]),
            ) >= float(separation_yaw_deg)
            for old in selected
        ):
            selected.append(row)
        if len(selected) >= max(1, int(count)):
            break
    return selected


def select_circular_orientation_peaks(
    scores: Sequence[float],
    *,
    bin_deg: float,
    count: int,
    separation_deg: float,
) -> list[Tuple[float, float]]:
    """Convert a 180-degree axis correlation into full-SE(2) yaw modes."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all():
        return []
    # A flat direction spectrum contains no yaw information.  Returning no
    # seed is safer than silently introducing a Manhattan-world prior.
    spread = float(np.max(values) - np.min(values))
    scale = max(1e-9, float(np.max(np.abs(values))))
    if spread <= 1e-5 * scale:
        return []
    median = float(np.median(values))
    meaningful = median + 0.05 * max(0.0, float(np.max(values)) - median)
    local = [
        index for index in range(len(values))
        if values[index] >= values[(index - 1) % len(values)]
        and values[index] >= values[(index + 1) % len(values)]
        and values[index] >= meaningful
    ]
    if not local:
        local = [int(np.argmax(values))]
    axes: list[Tuple[float, float]] = []
    for index in sorted(local, key=lambda item: values[item], reverse=True):
        yaw = float(index) * float(bin_deg)
        if all(
            min(
                abs(yaw - old[0]),
                180.0 - abs(yaw - old[0]),
            ) >= float(separation_deg)
            for old in axes
        ):
            axes.append((yaw, float(values[index])))
        if len(axes) >= max(1, int(math.ceil(int(count) / 2.0))):
            break

    full: list[Tuple[float, float]] = []
    for yaw, score in axes:
        full.append((wrap_deg(yaw), score))
        full.append((wrap_deg(yaw + 180.0), score))
    full.sort(key=lambda row: row[1], reverse=True)
    return full[: max(1, int(count))]


class GpuGlobalGeometrySeeder:
    """Generate competing whole-map SE(2) seeds on CUDA.

    ``seed()`` has no persistent pose state and cannot mutate the map.  A
    failed CUDA check or an ambiguous scene simply returns candidates/reasons
    for the caller's existing fail-closed validation path.
    """

    def __init__(
        self,
        grid: GridSpec,
        *,
        config: GlobalGeometryConfig = GlobalGeometryConfig(),
        device: Optional[str] = None,
    ) -> None:
        self.grid = grid
        self.config = config
        self.device_name = str(
            device or os.environ.get(GLOBAL_GEOMETRY_DEVICE_ENV, "cuda:0")
        )
        self.last_backend_reason = "not_checked"

    def backend_info(self) -> Dict[str, Any]:
        return {
            "backend": "torch_cuda_global_likelihood_field",
            "device": self.device_name,
            "cpu_fallback": False,
            "last_reason": self.last_backend_reason,
        }

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

    def _point_mask(self, points: np.ndarray, *, torch: Any, device: Any) -> Any:
        values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        mask = torch.zeros(
            (self.grid.size, self.grid.size),
            device=device,
            dtype=torch.bool,
        )
        if not len(values):
            return mask
        coordinates = torch.from_numpy(values).to(device=device)
        columns = torch.floor(
            (coordinates[:, 0] + float(self.grid.half_span_m))
            / float(self.grid.resolution_m)
        ).long()
        rows = torch.floor(
            (coordinates[:, 1] + float(self.grid.half_span_m))
            / float(self.grid.resolution_m)
        ).long()
        valid = (
            (columns >= 0)
            & (rows >= 0)
            & (columns < self.grid.size)
            & (rows < self.grid.size)
        )
        mask[rows[valid], columns[valid]] = True
        return mask

    def _wall_likelihood(self, wall: Any, *, torch: Any) -> Any:
        import torch.nn.functional as functional

        exact = wall.float()[None, None]
        field = exact
        # Discrete Gaussian-like likelihood field at 5 cm resolution.  Pooling
        # is CUDA-native and avoids a hidden CPU distance-transform hot path.
        for radius_cells in (1, 2, 3, 4):
            distance_m = radius_cells * float(self.grid.resolution_m)
            weight = math.exp(-0.5 * (distance_m / 0.12) ** 2)
            grown = functional.max_pool2d(
                exact,
                kernel_size=2 * radius_cells + 1,
                stride=1,
                padding=radius_cells,
            )
            field = torch.maximum(field, float(weight) * grown)
        return field[0, 0]

    def _orientation_histogram(
        self,
        points: np.ndarray,
        *,
        maximum: int,
        torch: Any,
        device: Any,
    ) -> Optional[Any]:
        values = _subsample(points, maximum)
        neighbor_count = min(
            len(values), max(4, int(self.config.orientation_neighbors))
        )
        if len(values) < max(int(self.config.minimum_wall_points), neighbor_count):
            return None
        points_gpu = torch.from_numpy(values).to(device=device)
        pairwise = torch.cdist(points_gpu, points_gpu)
        indices = torch.topk(
            pairwise, k=neighbor_count, dim=1, largest=False
        ).indices
        neighborhoods = points_gpu[indices]
        centered = neighborhoods - neighborhoods.mean(dim=1, keepdim=True)
        covariance = centered.transpose(1, 2) @ centered
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        anisotropy = 1.0 - (
            eigenvalues[:, 0] / eigenvalues[:, 1].clamp_min(1e-8)
        )
        usable = anisotropy >= float(self.config.orientation_min_anisotropy)
        if int(usable.sum().item()) < int(self.config.minimum_wall_points):
            return None
        tangents = eigenvectors[usable, :, 1]
        weights = anisotropy[usable].clamp(0.0, 1.0)
        bin_count = max(
            8, int(round(180.0 / float(self.config.orientation_bin_deg)))
        )
        positions = torch.remainder(
            torch.atan2(tangents[:, 1], tangents[:, 0]), math.pi
        ) * (float(bin_count) / math.pi)
        lower_float = torch.floor(positions)
        lower = lower_float.long() % bin_count
        fraction = positions - lower_float
        histogram = torch.zeros(bin_count, device=device, dtype=torch.float32)
        histogram.scatter_add_(0, lower, weights * (1.0 - fraction))
        histogram.scatter_add_(0, (lower + 1) % bin_count, weights * fraction)
        blurred = (
            histogram
            + 0.60 * torch.roll(histogram, 1)
            + 0.60 * torch.roll(histogram, -1)
            + 0.20 * torch.roll(histogram, 2)
            + 0.20 * torch.roll(histogram, -2)
        )
        return blurred / torch.linalg.vector_norm(blurred).clamp_min(1e-8)

    def _yaw_candidates(
        self,
        source_wall: np.ndarray,
        target_wall: np.ndarray,
        *,
        torch: Any,
        device: Any,
    ) -> list[Tuple[float, float]]:
        source = self._orientation_histogram(
            source_wall,
            maximum=int(self.config.maximum_source_wall_points),
            torch=torch,
            device=device,
        )
        target = self._orientation_histogram(
            target_wall,
            maximum=int(self.config.maximum_target_orientation_points),
            torch=torch,
            device=device,
        )
        if source is None or target is None:
            return []
        scores = torch.stack([
            torch.dot(target, torch.roll(source, shifts=index))
            for index in range(int(source.shape[0]))
        ])
        return select_circular_orientation_peaks(
            scores.detach().cpu().numpy(),
            bin_deg=180.0 / float(source.shape[0]),
            count=int(self.config.orientation_peak_count),
            separation_deg=float(self.config.orientation_peak_separation_deg),
        )

    def seed(
        self,
        source_wall: np.ndarray,
        source_free: np.ndarray,
        target_wall: np.ndarray,
        target_free: np.ndarray,
        *,
        pivot: Tuple[float, float],
    ) -> Dict[str, Any]:
        """Return global correction modes without accepting any of them."""

        torch, device, backend_reason = self._cuda()
        self.last_backend_reason = backend_reason
        base = {
            "accepted": False,
            "pose_correction_safe": False,
            "geometry_backend": self.backend_info(),
            "candidates": [],
            "search_complete": False,
            "candidate_budget_truncated": False,
        }
        if torch is None or device is None:
            return {**base, "reason": backend_reason}
        source_wall = unique_cells(source_wall, self.grid.resolution_m)
        source_free = unique_cells(source_free, self.grid.resolution_m)
        target_wall = unique_cells(target_wall, self.grid.resolution_m)
        target_free = unique_cells(target_free, self.grid.resolution_m)
        if (
            len(source_wall) < int(self.config.minimum_wall_points)
            or len(target_wall) < int(self.config.minimum_wall_points)
            or not len(target_free)
        ):
            return {**base, "reason": "insufficient_global_geometry"}

        yaw_candidates = self._yaw_candidates(
            source_wall, target_wall, torch=torch, device=device
        )
        if not yaw_candidates:
            return {**base, "reason": "global_yaw_unobservable"}
        yaw_budget_truncated = (
            len(yaw_candidates) >= int(self.config.orientation_peak_count)
        )

        translations = unique_cells(
            target_free, float(self.config.translation_step_m)
        )
        source_wall = _subsample(
            source_wall, int(self.config.maximum_source_wall_points)
        )
        source_free = _subsample(
            source_free, int(self.config.maximum_source_free_points)
        )
        wall_mask = self._point_mask(target_wall, torch=torch, device=device)
        free_mask = self._point_mask(target_free, torch=torch, device=device)
        wall_likelihood = self._wall_likelihood(wall_mask, torch=torch)
        translations_gpu = torch.from_numpy(translations).to(device=device)
        pivot_gpu = torch.tensor(pivot, device=device, dtype=torch.float32)
        wall_centered = (
            torch.from_numpy(source_wall).to(device=device) - pivot_gpu
        )
        free_centered = (
            torch.from_numpy(source_free).to(device=device) - pivot_gpu
            if len(source_free)
            else None
        )
        height = width = int(self.grid.size)
        x_min = y_min = -float(self.grid.half_span_m)

        def indices(world: Any) -> Tuple[Any, Any, Any]:
            columns = torch.floor(
                (world[..., 0] - x_min) / float(self.grid.resolution_m)
            ).long()
            rows = torch.floor(
                (world[..., 1] - y_min) / float(self.grid.resolution_m)
            ).long()
            inside = (
                (columns >= 0)
                & (rows >= 0)
                & (columns < width)
                & (rows < height)
            )
            return (
                rows.clamp(0, height - 1),
                columns.clamp(0, width - 1),
                inside,
            )

        results: list[Dict[str, float]] = []
        translation_peak_budget_truncated = False
        keep_wall = max(
            1,
            int(math.ceil(
                float(self.config.robust_wall_keep_fraction)
                * len(source_wall)
            )),
        )
        with torch.inference_mode():
            for yaw_deg, orientation_score in yaw_candidates:
                yaw = math.radians(float(yaw_deg))
                rotation = torch.tensor(
                    [[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]],
                    device=device,
                    dtype=torch.float32,
                )
                rotated_wall = wall_centered @ rotation.T
                rotated_free = (
                    None if free_centered is None else free_centered @ rotation.T
                )
                yaw_scores: list[Any] = []
                yaw_hit_ratios: list[Any] = []
                yaw_blocked_ratios: list[Any] = []
                batch = max(1, int(self.config.batch_size))
                for begin in range(0, len(translations), batch):
                    anchors = translations_gpu[begin:begin + batch]
                    wall_world = anchors[:, None, :] + rotated_wall[None, :, :]
                    wall_rows, wall_columns, wall_inside = indices(wall_world)
                    wall_values = (
                        wall_likelihood[wall_rows, wall_columns]
                        * wall_inside.float()
                    )
                    wall_score = torch.topk(
                        wall_values,
                        k=keep_wall,
                        dim=1,
                        sorted=False,
                    ).values.mean(dim=1)
                    hit_ratio = (
                        (wall_values >= 0.70) & wall_inside
                    ).float().mean(dim=1)

                    if rotated_free is None:
                        free_score = torch.zeros_like(wall_score)
                        blocked_score = torch.zeros_like(wall_score)
                    else:
                        free_world = anchors[:, None, :] + rotated_free[None, :, :]
                        free_rows, free_columns, free_inside = indices(free_world)
                        free_score = (
                            free_mask[free_rows, free_columns] & free_inside
                        ).float().mean(dim=1)
                        blocked_score = (
                            wall_mask[free_rows, free_columns] & free_inside
                        ).float().mean(dim=1)
                    score = (
                        0.80 * wall_score
                        + 0.20 * free_score
                        - 1.25 * blocked_score
                    )
                    yaw_scores.append(score)
                    yaw_hit_ratios.append(hit_ratio)
                    yaw_blocked_ratios.append(blocked_score)
                scores = torch.cat(yaw_scores)
                hit_ratios = torch.cat(yaw_hit_ratios)
                blocked_ratios = torch.cat(yaw_blocked_ratios)
                available = scores.clone()
                yaw_rows: list[Tuple[float, float, float, int]] = []
                peak_budget = max(1, int(self.config.top_per_yaw))
                # Suppress each spatial mode on CUDA before applying top-K.
                # Otherwise dozens of samples around one strong room can
                # evict a weaker, separated but valid repeated-room mode.
                for _unused in range(min(len(translations), peak_budget + 1)):
                    value, selected = torch.max(available, dim=0)
                    if not bool(torch.isfinite(value).item()):
                        break
                    index = int(selected.item())
                    yaw_rows.append((
                        float(value.item()),
                        float(hit_ratios[index].item()),
                        float(blocked_ratios[index].item()),
                        index,
                    ))
                    delta = translations_gpu - translations_gpu[index]
                    nearby = torch.sum(delta * delta, dim=1) < (
                        float(self.config.peak_separation_m) ** 2
                    )
                    available[nearby] = -torch.inf
                translation_peak_budget_truncated |= len(yaw_rows) > peak_budget
                for score, hit_ratio, blocked_ratio, index in yaw_rows[:peak_budget]:
                    x_m, y_m = translations[index]
                    results.append({
                        "score": float(score),
                        "wall_hit_ratio": float(hit_ratio),
                        "free_blocked_ratio": float(blocked_ratio),
                        "orientation_score": float(orientation_score),
                        "x_m": float(x_m),
                        "y_m": float(y_m),
                        "yaw_correction_deg": float(yaw_deg),
                        "dx_m": float(x_m) - float(pivot[0]),
                        "dy_m": float(y_m) - float(pivot[1]),
                    })
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        peak_budget = max(1, int(self.config.peak_count))
        peaks_with_sentinel = nonmax_global_candidates(
            results,
            count=peak_budget + 1,
            separation_m=float(self.config.peak_separation_m),
            separation_yaw_deg=float(self.config.peak_separation_yaw_deg),
        )
        budget_truncated = (
            len(peaks_with_sentinel) > peak_budget
            or yaw_budget_truncated
            or translation_peak_budget_truncated
        )
        peaks = peaks_with_sentinel[:peak_budget]
        return {
            **base,
            "reason": "global_candidates_generated" if peaks else "no_global_peaks",
            "candidate_translations": int(len(translations)),
            "candidate_yaws": int(len(yaw_candidates)),
            "candidates": peaks,
            "search_complete": True,
            "candidate_budget_truncated": bool(budget_truncated),
            "yaw_budget_truncated": bool(yaw_budget_truncated),
            "translation_peak_budget_truncated": bool(
                translation_peak_budget_truncated
            ),
        }


__all__ = [
    "GLOBAL_GEOMETRY_DEVICE_ENV",
    "GlobalGeometryConfig",
    "GpuGlobalGeometrySeeder",
    "nonmax_global_candidates",
    "select_circular_orientation_peaks",
]
