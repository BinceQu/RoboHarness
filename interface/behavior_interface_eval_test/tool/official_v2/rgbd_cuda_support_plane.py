"""CUDA back-projection and dominant support-plane fitting for V53."""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .depth_mesh_reconstruction import SupportPlane, _plane_basis
from .rgbd_projective_occupancy import ProjectiveOccupancyUnavailable


CUDA_SUPPORT_PLANE_VERSION = "rgbd_v53_cuda_support_plane_v1"
DEFAULT_MIN_FREE_CUDA_MIB = 512


def _device(
    requested_device: Optional[str],
    *,
    allow_cpu_reference: bool,
):
    try:
        import torch
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"Torch is required for V53 CUDA support-plane fitting: {exc}"
        ) from exc
    name = str(
        requested_device
        or os.environ.get("OFFICIAL_V2_LITE_OCCUPANCY_DEVICE", "cuda:0")
    ).strip()
    device = torch.device(name)
    if device.type == "cpu":
        if not allow_cpu_reference:
            raise ProjectiveOccupancyUnavailable(
                "CPU V53 support-plane fitting is disabled in runtime"
            )
    elif device.type != "cuda":
        raise ProjectiveOccupancyUnavailable(
            f"unsupported V53 support-plane device {device}"
        )
    try:
        torch.empty(1, dtype=torch.uint8, device=device)
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"V53 support-plane device {device} is unavailable: {exc}"
        ) from exc
    return torch, device


def _memory_preflight(torch, device, *, pixels: int) -> Dict[str, Any]:
    if device.type != "cuda":
        return {
            "cuda_free_mib_before": None,
            "estimated_required_mib": None,
        }
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    reserve_mib = max(
        0,
        int(
            os.environ.get(
                "OFFICIAL_V2_LITE_GPU_OCCUPANCY_MIN_FREE_MIB",
                str(DEFAULT_MIN_FREE_CUDA_MIB),
            )
        ),
    )
    # Depth, world XYZ, validity, samples, two design matrices, sort work,
    # and allocator headroom. Float64 is intentional to preserve V53 output.
    estimated_bytes = int(pixels * 192 + 96 * 1024**2)
    if int(free_bytes) - reserve_mib * 1024**2 < estimated_bytes:
        raise ProjectiveOccupancyUnavailable(
            f"insufficient CUDA memory for V53 support-plane fitting on "
            f"{device}: {free_bytes / 1024**2:.1f} MiB free, "
            f"{estimated_bytes / 1024**2:.1f} MiB estimated, "
            f"{reserve_mib} MiB reserved"
        )
    return {
        "cuda_free_mib_before": float(free_bytes / 1024**2),
        "estimated_required_mib": float(estimated_bytes / 1024**2),
    }


def _quat_to_matrix_tensor(torch, quaternion, *, device):
    x, y, z, w = torch.as_tensor(
        np.asarray(quaternion, dtype=np.float64).reshape(4),
        dtype=torch.float64,
        device=device,
    ).unbind()
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x))),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))),
        )
    )


def _least_squares(torch, samples):
    design = torch.stack(
        (
            samples[:, 0],
            samples[:, 1],
            torch.ones_like(samples[:, 0]),
        ),
        dim=1,
    )
    return torch.linalg.lstsq(
        design,
        samples[:, 2, None],
        rcond=None,
    ).solution[:, 0]


def backproject_and_fit_support_plane_cuda(
    depth: np.ndarray,
    *,
    camera_pos: Sequence[float],
    camera_quat_xyzw: Sequence[float],
    focal_length: float,
    horizontal_aperture: float,
    roi: Optional[Tuple[int, int, int, int]] = None,
    histogram_bin_m: float = 0.002,
    inlier_band_m: float = 0.003,
    requested_device: Optional[str] = None,
    allow_cpu_reference: bool = False,
) -> Tuple[np.ndarray, np.ndarray, SupportPlane, Dict[str, Any]]:
    """Back-project one depth image and fit V53's support plane on CUDA."""
    started = time.perf_counter()
    metric_depth = np.asarray(depth, dtype=np.float64)
    if metric_depth.ndim == 3:
        metric_depth = metric_depth[..., 0]
    if metric_depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {metric_depth.shape}")
    height, width = metric_depth.shape
    fx = float(focal_length) / float(horizontal_aperture) * float(width)
    if not np.isfinite(fx) or fx <= 0.0:
        raise ValueError(f"invalid camera intrinsics fx={fx}")
    if histogram_bin_m <= 0.0 or inlier_band_m <= 0.0:
        raise ValueError("support-plane bands must be positive")

    torch, device = _device(
        requested_device,
        allow_cpu_reference=bool(allow_cpu_reference),
    )
    memory = _memory_preflight(torch, device, pixels=height * width)
    try:
        with torch.inference_mode():
            dep = torch.as_tensor(
                metric_depth,
                dtype=torch.float64,
                device=device,
            )
            vv, uu = torch.meshgrid(
                torch.arange(height, dtype=torch.float64, device=device),
                torch.arange(width, dtype=torch.float64, device=device),
                indexing="ij",
            )
            camera_points = torch.stack(
                (
                    (uu - width / 2.0) / fx * dep,
                    -(vv - height / 2.0) / fx * dep,
                    -dep,
                ),
                dim=-1,
            )
            rotation = _quat_to_matrix_tensor(
                torch,
                camera_quat_xyzw,
                device=device,
            )
            position = torch.as_tensor(
                np.asarray(camera_pos, dtype=np.float64).reshape(3),
                dtype=torch.float64,
                device=device,
            )
            world = camera_points @ rotation.T + position
            valid = (
                torch.isfinite(dep)
                & (dep > 0.03)
                & (dep < 20.0)
                & torch.isfinite(world).all(dim=-1)
            )

            x0, y0, x1, y1 = (
                (0, 0, width, height)
                if roi is None
                else tuple(int(value) for value in roi)
            )
            samples = world[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
            if len(samples) < 100:
                raise ValueError(
                    "support-plane ROI has too few valid points: "
                    f"{len(samples)}"
                )
            z = samples[:, 2]
            quantiles = torch.quantile(
                z,
                torch.as_tensor(
                    [0.01, 0.99],
                    dtype=torch.float64,
                    device=device,
                ),
            )
            low = quantiles[0]
            high = quantiles[1]
            if float((high - low).item()) < float(histogram_bin_m):
                mode_z = torch.quantile(z, 0.5)
            else:
                lower_bin = torch.floor(low / histogram_bin_m).to(torch.int64)
                upper_bin = torch.ceil(high / histogram_bin_m).to(torch.int64)
                bin_count = int((upper_bin - lower_bin).item())
                indices = torch.floor(z / histogram_bin_m).to(torch.int64) - lower_bin
                at_upper_edge = indices == bin_count
                indices = torch.where(at_upper_edge, indices - 1, indices)
                included = (indices >= 0) & (indices < bin_count)
                histogram = torch.bincount(
                    indices[included],
                    minlength=bin_count,
                )
                mode_index = torch.argmax(histogram)
                mode_z = (
                    lower_bin.to(torch.float64)
                    + mode_index.to(torch.float64)
                    + 0.5
                ) * histogram_bin_m

            coarse_band = max(
                float(inlier_band_m) * 2.0,
                float(histogram_bin_m),
            )
            coarse = samples[torch.abs(z - mode_z) <= coarse_band]
            if len(coarse) < 100:
                raise ValueError(
                    f"support-plane mode has too few points: {len(coarse)}"
                )
            coefficient = _least_squares(torch, coarse)
            predicted = (
                coarse[:, 0] * coefficient[0]
                + coarse[:, 1] * coefficient[1]
                + coefficient[2]
            )
            residual = coarse[:, 2] - predicted
            residual_center = torch.quantile(residual, 0.5)
            keep = torch.abs(residual - residual_center) <= inlier_band_m
            inliers = coarse[keep]
            if len(inliers) < 100:
                inliers = coarse
            coefficient = _least_squares(torch, inliers)
            a, b, c = coefficient.unbind()
            scale = torch.sqrt(a * a + b * b + 1.0)
            normal = torch.stack((-a, -b, torch.ones_like(a))) / scale
            offset = -c / scale
            if bool(normal[2] < 0.0):
                normal = -normal
                offset = -offset
            signed = inliers @ normal + offset
            absolute_mm = torch.abs(signed) * 1000.0
            residual_metrics = torch.quantile(
                absolute_mm,
                torch.as_tensor(
                    [0.5, 0.95],
                    dtype=torch.float64,
                    device=device,
                ),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            world_np = world.cpu().numpy()
            valid_np = valid.cpu().numpy()
            normal_np = normal.cpu().numpy()
            offset_value = float(offset.item())
            residual_values = residual_metrics.cpu().numpy()
            inlier_count = int(len(inliers))
    except ValueError:
        raise
    except Exception as exc:
        raise ProjectiveOccupancyUnavailable(
            f"V53 CUDA support-plane fitting failed on {device}: {exc}"
        ) from exc

    basis_x, basis_y = _plane_basis(normal_np)
    plane = SupportPlane(
        normal=np.asarray(normal_np, dtype=np.float64),
        offset=offset_value,
        origin=-offset_value * np.asarray(normal_np, dtype=np.float64),
        basis_x=basis_x,
        basis_y=basis_y,
        inlier_count=inlier_count,
        residual_median_mm=float(residual_values[0]),
        residual_p95_mm=float(residual_values[1]),
    )
    metadata = {
        "support_plane_backend": CUDA_SUPPORT_PLANE_VERSION,
        "support_plane_device": str(device),
        "support_plane_elapsed_s": float(time.perf_counter() - started),
        **memory,
    }
    return world_np, valid_np, plane, metadata


__all__ = [
    "CUDA_SUPPORT_PLANE_VERSION",
    "backproject_and_fit_support_plane_cuda",
]
