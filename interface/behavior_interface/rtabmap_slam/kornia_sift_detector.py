"""CUDA-only Kornia SIFT adapter for RTAB-Map's ``PyDetector`` API.

RTAB-Map passes only the current head-camera grayscale image to ``detect``.
The adapter deliberately has no CPU fallback: selecting this backend is an
explicit promise that visual feature extraction runs on CUDA.
"""

from __future__ import annotations

import random
import traceback
from typing import Any

import numpy as np


_torch: Any = None
_kornia: Any = None
_device: Any = None
_extractor: Any = None


def shutdown() -> None:
    """Release CUDA tensors before the embedded interpreter tears down."""

    global _torch, _kornia, _device, _extractor
    torch_module = _torch
    try:
        if torch_module is not None and torch_module.cuda.is_initialized():
            torch_module.cuda.synchronize()
    finally:
        # Py_Finalize clears extension modules in an implementation-dependent
        # order. Drop CUDA-owning Python objects while torch is still intact.
        _extractor = None
        _device = None
        _kornia = None
        if torch_module is not None and torch_module.cuda.is_initialized():
            torch_module.cuda.empty_cache()
        _torch = None


def init(cuda: int) -> None:
    """Initialize one persistent CUDA SIFT extractor for the worker process."""

    if not int(cuda):
        raise RuntimeError("Kornia SIFT is CUDA-only; PyDetector/Cuda must be true")

    import kornia
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kornia SIFT requested, but CUDA is not available")

    # RTAB-Map owns the CPU-side graph and grid work. Keep PyTorch from opening
    # an additional host thread pool for small tensor bookkeeping operations.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only before the first inter-op task. A
        # previous PyDetector instance in the same embedded interpreter may
        # already have established the requested single-thread pool.
        pass
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Do not enable torch.use_deterministic_algorithms() process-wide here.
    # Kornia's SIFT orientation path reaches CUDA affine_grid; PyTorch 2.5 can
    # make that allocation fail inside an embedded interpreter when the global
    # deterministic dispatcher is enabled.  This inference-only pipeline has
    # no backward atomic reductions. Fixed seeds, deterministic cuDNN, and the
    # worker's CUBLAS_WORKSPACE_CONFIG cover the CUDA operations used here.

    device = torch.device("cuda:0")
    # Force context creation now. This makes an invalid CUDA_VISIBLE_DEVICES,
    # driver error, or out-of-memory condition fail the worker startup probe.
    torch.empty(1, device=device)
    extractor = kornia.feature.SIFTFeature(
        num_features=1400,
        upright=False,
        rootsift=False,
        device=device,
    ).eval()

    global _torch, _kornia, _device, _extractor
    _torch = torch
    _kornia = kornia
    _device = device
    _extractor = extractor


def detect(image_buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return RTAB-Map-compatible ``(Nx3 keypoints, Nx128 descriptors)``."""

    if _extractor is None or _torch is None or _device is None:
        raise RuntimeError("Kornia SIFT detector was used before init(cuda=1)")

    gray = np.asarray(image_buffer)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("PyDetector image must be a 2-D uint8 grayscale array")
    image = _torch.from_numpy(np.ascontiguousarray(gray)).to(
        device=_device,
        dtype=_torch.float32,
    )[None, None]
    image.mul_(1.0 / 255.0)

    try:
        with _torch.inference_mode():
            lafs, responses, descriptors = _extractor(image)
            centers = _kornia.feature.get_laf_center(lafs)
            scores = responses.reshape(responses.shape[0], responses.shape[1], 1)
            points = _torch.cat((centers, scores), dim=-1)[0]
            descriptors = descriptors[0]
    except Exception:
        # PyDetector reports Python failures through RTAB-Map's logger, which
        # may be disabled before worker startup completes. Keep the original
        # traceback visible on the worker's configured stderr stream.
        traceback.print_exc()
        raise

    # The device-to-host copies synchronize this frame's CUDA work. RTAB-Map's
    # C API retains the returned arrays only for the duration of this call.
    points_cpu = points.detach().to(device="cpu", dtype=_torch.float32).numpy()
    descriptors_cpu = (
        descriptors.detach().to(device="cpu", dtype=_torch.float32).numpy()
    )
    if points_cpu.shape[0] != descriptors_cpu.shape[0]:
        raise RuntimeError("Kornia returned mismatched keypoint and descriptor counts")
    return (
        np.ascontiguousarray(points_cpu, dtype=np.float32),
        np.ascontiguousarray(descriptors_cpu, dtype=np.float32),
    )
