"""CUDA one-way Lowe-ratio matcher for RTAB-Map's ``PyMatcher`` API.

The matcher consumes only descriptors already extracted from compliant head
RGB observations. It is used for long-baseline place verification; local
tracking with a qvel pose guess keeps RTAB-Map's projected search window.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_torch: Any = None
_device: Any = None
_ratio_threshold = 0.80


def init(
    descriptor_dim: int,
    match_threshold: float,
    iterations: int,
    cuda: int,
    model_path: str,
) -> None:
    """Initialize a CUDA-only, model-free SIFT matcher."""

    del iterations, model_path
    if not int(cuda):
        raise RuntimeError("Kornia SIFT matching is CUDA-only")
    if int(descriptor_dim) != 128:
        raise ValueError("Kornia SIFT descriptors must have 128 columns")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kornia SIFT matcher requested, but CUDA is unavailable")
    threshold = float(match_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("mutual SIFT ratio threshold must be within (0, 1)")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0")
    torch.empty(1, device=device)

    global _torch, _device, _ratio_threshold
    _torch = torch
    _device = device
    _ratio_threshold = threshold


def _descriptors(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 128:
        raise ValueError(f"{name} must have shape Nx128")
    if array.dtype != np.float32:
        raise ValueError(f"{name} must use float32 descriptors")
    if array.shape[0] > 4096:
        raise ValueError(f"{name} exceeds the bounded matcher capacity")
    return np.ascontiguousarray(array)


def _unique_target_pairs(
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Keep the best source for each target descriptor, deterministically."""

    source_rows = np.asarray(source_rows, dtype=np.int64)
    target_rows = np.asarray(target_rows, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)
    if not (len(source_rows) == len(target_rows) == len(distances)):
        raise ValueError("match arrays must have equal length")
    if not len(source_rows):
        return np.empty((0, 2), dtype=np.int64)
    # Primary key is distance, then source id.  Iterating this order and taking
    # the first occurrence of each target is stable across CUDA/CPU versions.
    order = np.lexsort((source_rows, distances))
    used_targets: set[int] = set()
    pairs = []
    for index in order:
        target = int(target_rows[index])
        if target in used_targets:
            continue
        used_targets.add(target)
        pairs.append((int(source_rows[index]), target))
    pairs.sort()
    return np.ascontiguousarray(pairs, dtype=np.int64)


def match(
    kpts_from: np.ndarray,
    kpts_to: np.ndarray,
    scores_from: np.ndarray,
    scores_to: np.ndarray,
    descriptors_from: np.ndarray,
    descriptors_to: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Return unique one-way Lowe-ratio matches as ``[from, to]``.

    Long-baseline and reverse-view revisits often violate a second ratio test
    in the opposite direction.  RTAB-Map's following 3-D RANSAC and ICP stages
    provide metric verification, so this adapter follows the standard one-way
    NNDR proposal rule and only prevents duplicate votes for one target.
    """

    del kpts_from, kpts_to, scores_from, scores_to, image_width, image_height
    if _torch is None or _device is None:
        raise RuntimeError("Kornia SIFT matcher was used before init")

    source = _descriptors(descriptors_from, "descriptors_from")
    target = _descriptors(descriptors_to, "descriptors_to")
    if source.shape[0] < 2 or target.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)

    with _torch.inference_mode():
        source_gpu = _torch.from_numpy(source).to(_device)
        target_gpu = _torch.from_numpy(target).to(_device)
        distances = _torch.cdist(source_gpu, target_gpu, p=2.0)

        source_values, source_indices = _torch.topk(
            distances, k=2, dim=1, largest=False, sorted=True
        )
        source_best = source_indices[:, 0]
        source_rows = _torch.arange(source.shape[0], device=_device)
        source_ratio_ok = source_values[:, 0] <= (
            _ratio_threshold * source_values[:, 1]
        )
        accepted_rows = source_rows[source_ratio_ok]
        accepted_targets = source_best[source_ratio_ok]
        accepted_distances = source_values[source_ratio_ok, 0]

    return _unique_target_pairs(
        accepted_rows.to("cpu").numpy(),
        accepted_targets.to("cpu").numpy(),
        accepted_distances.to("cpu").numpy(),
    )
