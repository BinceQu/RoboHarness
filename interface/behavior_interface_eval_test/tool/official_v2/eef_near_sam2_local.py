"""eef_near 持物 RGB 分割：SAM 2.1 点+框提示。

官方 capture 不改，也不用仿真 seg。权重缺失或关闭时返回 None，
外层继续只用 3D 生长。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional
from pathlib import Path

import numpy as np

_LOG = logging.getLogger(__name__)

_DATA_ROOT = Path(__file__).resolve().parents[4] / "data"
_DEFAULT_REPO = str(_DATA_ROOT / "sam2")
_DEFAULT_DEPS = str(_DATA_ROOT / "sam2-python-deps")
_DEFAULT_CKPT = str(_DATA_ROOT / "sam2.1_hiera_small.pt")
_DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

_LOCK = threading.Lock()
_PREDICTOR = None
_LOAD_FAILED = False
_IMAGE_KEY: Optional[int] = None


def sam2_enabled() -> bool:
    """环境变量 `EEF_NEAR_SAM2=0` 可关。"""
    flag = os.environ.get("EEF_NEAR_SAM2", "1").strip().lower()
    if flag in {"0", "false", "off", "no"}:
        return False
    ckpt = os.environ.get("EEF_NEAR_SAM2_CKPT", _DEFAULT_CKPT)
    return os.path.isfile(ckpt)


def _pick_device() -> str:
    forced = os.environ.get("EEF_NEAR_SAM2_DEVICE", "").strip()
    if forced:
        return forced
    try:
        import torch
    except Exception:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    # A masked interface sees its owned physical GPU as local cuda:0.  Never
    # select another process-wide card by free-memory heuristics: that lets a
    # direct/unmasked invocation steal a sibling interface's GPU.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    physical = os.environ.get("BEHAVIOR_INTERFACE_PHYSICAL_GPU", "").strip()
    if visible and "," not in visible and visible.isdigit():
        return "cuda:0"
    if (not visible) and physical.isdigit():
        owned = int(physical)
        if 0 <= owned < int(torch.cuda.device_count()):
            return f"cuda:{owned}"
    if visible:
        entries = [item.strip() for item in visible.split(",")]
        if physical.isdigit() and physical in entries:
            return f"cuda:{entries.index(physical)}"
    # No ownership metadata: choose the deterministic local default instead
    # of scanning all cards and silently borrowing another interface's GPU.
    return "cuda:0"


def _ensure_predictor():
    global _PREDICTOR, _LOAD_FAILED
    if _PREDICTOR is not None or _LOAD_FAILED:
        return _PREDICTOR
    with _LOCK:
        if _PREDICTOR is not None or _LOAD_FAILED:
            return _PREDICTOR
        if not sam2_enabled():
            _LOAD_FAILED = True
            return None
        repo = os.environ.get("EEF_NEAR_SAM2_REPO", _DEFAULT_REPO)
        deps = os.environ.get("EEF_NEAR_SAM2_DEPS", _DEFAULT_DEPS)
        ckpt = os.environ.get("EEF_NEAR_SAM2_CKPT", _DEFAULT_CKPT)
        cfg = os.environ.get("EEF_NEAR_SAM2_CONFIG", _DEFAULT_CFG)
        if not os.path.isdir(os.path.join(repo, "sam2")) or not os.path.isfile(ckpt):
            _LOG.warning("SAM2 路径不可用 repo=%s ckpt=%s", repo, ckpt)
            _LOAD_FAILED = True
            return None
        import sys

        if deps and os.path.isdir(deps) and deps not in sys.path:
            sys.path.insert(0, deps)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            device = _pick_device()
            if device == "cpu":
                _LOG.warning("SAM2 无 GPU，跳过以免拖慢 capture")
                _LOAD_FAILED = True
                return None
            model = build_sam2(cfg, ckpt, device=device, mode="eval")
            _PREDICTOR = SAM2ImagePredictor(model)
            _LOG.info("SAM2 已加载 device=%s ckpt=%s", device, ckpt)
        except Exception:
            _LOG.exception("SAM2 加载失败，回退 3D 持物生长")
            _LOAD_FAILED = True
            return None
        return _PREDICTOR


def _farthest_points(rows: np.ndarray, cols: np.ndarray, count: int) -> np.ndarray:
    """从种子像素里均匀抽点，坐标是 (u, v)。"""
    points = np.stack(
        [
            np.asarray(cols, dtype=np.float32).reshape(-1),
            np.asarray(rows, dtype=np.float32).reshape(-1),
        ],
        axis=1,
    )
    if points.shape[0] == 0:
        return points
    take = max(1, min(int(count), int(points.shape[0])))
    if points.shape[0] <= take:
        return points
    center = points.mean(axis=0)
    chosen = [int(np.argmin(np.sum((points - center) ** 2, axis=1)))]
    nearest = np.full((points.shape[0],), np.inf, dtype=np.float64)
    for _ in range(take - 1):
        last = points[chosen[-1]]
        nearest = np.minimum(nearest, np.sum((points - last) ** 2, axis=1))
        chosen.append(int(np.argmax(nearest)))
    return points[np.asarray(chosen, dtype=np.int64)]


def predict_held_mask_bgr(
    scene_bgr: np.ndarray,
    seed_rows: np.ndarray,
    seed_cols: np.ndarray,
    *,
    point_count: int = 8,
    box_pad_px: int = 40,
) -> Optional[np.ndarray]:
    """用夹爪种子点+紧框预测持物 mask。失败返回 None。"""
    if scene_bgr is None:
        return None
    image = np.asarray(scene_bgr)
    if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return None
    seed_rows = np.asarray(seed_rows).reshape(-1)
    seed_cols = np.asarray(seed_cols).reshape(-1)
    if seed_rows.size < 1:
        return None
    predictor = _ensure_predictor()
    if predictor is None:
        return None
    rgb = np.ascontiguousarray(image[..., :3][:, :, ::-1])
    height, width = rgb.shape[:2]
    keep = (
        (seed_rows >= 0)
        & (seed_rows < height)
        & (seed_cols >= 0)
        & (seed_cols < width)
    )
    seed_rows = seed_rows[keep]
    seed_cols = seed_cols[keep]
    if seed_rows.size < 1:
        return None
    points = _farthest_points(seed_rows, seed_cols, point_count)
    box = np.array(
        [
            float(seed_cols.min()) - float(box_pad_px),
            float(seed_rows.min()) - float(box_pad_px),
            float(seed_cols.max()) + float(box_pad_px),
            float(seed_rows.max()) + float(box_pad_px),
        ],
        dtype=np.float32,
    )
    try:
        import torch

        device = predictor.device
        use_cuda = str(device).startswith("cuda")
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_cuda
            else torch.autocast("cpu", enabled=False)
        )
        global _IMAGE_KEY
        key = id(scene_bgr)
        with torch.inference_mode(), context:
            if _IMAGE_KEY != key:
                predictor.set_image(rgb)
                _IMAGE_KEY = key
            masks, _ious, _low = predictor.predict(
                point_coords=points,
                point_labels=np.ones((points.shape[0],), dtype=np.int32),
                box=box,
                multimask_output=False,
            )
        mask = np.asarray(masks[0], dtype=bool)
        if mask.shape[:2] != (height, width):
            return None
        return mask
    except Exception:
        _LOG.exception("SAM2 推理失败")
        return None
