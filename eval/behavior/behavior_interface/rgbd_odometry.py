"""自研 EgoMap 的相邻帧 RGB-D 里程估计。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Any, Iterable, Optional

import numpy as np


RATIO_LIMIT = 0.70
MIN_RATIO_MATCHES = 12
MIN_RGBD_PAIRS = 8
MIN_SE2_INLIERS = 12
MIN_SE2_INLIER_RATIO = 0.55
MAX_SE2_RMSE_M = 0.06
MAX_QVEL_TRANSLATION_ERROR_M = 0.18
MAX_QVEL_YAW_ERROR_DEG = 5.0
MAX_DEPTH_M = 8.0
MAX_DEPTH_PATCH_SPAN_M = 0.12
MAX_HEIGHT_DELTA_M = 0.18
RANSAC_THRESHOLD_M = 0.12
RANSAC_ITERATIONS = 400
# CPU/CUDA 必须抽到同一组假设，固定算法种子只用于数值复现。
RANSAC_RANDOM_SEED = 0x13527DB

# 长期定位的质量门槛。它们只决定是否接受一次视觉位姿，不会修改地图。
KEYFRAME_RATIO_LIMIT = 0.75
KEYFRAME_MIN_RATIO_MATCHES = 12
KEYFRAME_MIN_RGBD_PAIRS = 8
KEYFRAME_MIN_INLIERS = 12
KEYFRAME_MIN_INLIER_RATIO = 0.50
KEYFRAME_MAX_RMSE_M = 0.08
KEYFRAME_MIN_DEPTH_M = 0.45
KEYFRAME_MAX_RANGE_M = 8.0
KEYFRAME_MAX_KEYPOINTS = 900
# 长期地点库不能只留下路线尾部，否则回到起点时只剩一个偶然首视角。
# 96 帧仍是严格有界的几十 MB；超出后按全程/近期各半分层保留。
KEYFRAME_MAX_COUNT = 256
KEYFRAME_MIN_GAP = 8
PLACE_KEYFRAME_MAX_COUNT = 512
PLACE_KEYFRAME_MIN_GAP = 2
PLACE_KEYFRAME_MAX_GAP = 4
PLACE_KEYFRAME_MIN_TRANSLATION_M = 0.20
PLACE_KEYFRAME_MIN_YAW_DEG = 10.0
KEYFRAME_MIN_SPAN_M = 0.80
KEYFRAME_MIN_CROSS_SPAN_M = 0.20
KEYFRAME_MIN_HEIGHT_SPAN_M = 0.10
# 长时回环不能把相邻帧当成独立证据。这个间隔只用于关键帧之间的
# 几何验证，不影响相邻 RGB-D 里程。
KEYFRAME_MIN_FRAME_GAP = 60
KEYFRAME_INDEPENDENT_GAP = 32
# 两张关键帧只有时间分开还不够：机器人原地转身时，同一扇门可以在几十帧
# 后再次出现，却没有提供新的视差。长期定位要求历史相机中心也确实移动过。
KEYFRAME_INDEPENDENT_TRANSLATION_M = 0.35
# Exact pairwise-independent evidence selection is combinatorial in the worst
# case.  Exceeding this deterministic work budget rejects localization rather
# than using an under-counted heuristic to rank competing place modes.
KEYFRAME_INDEPENDENT_SEARCH_MAX_STATES = 250_000
KEYFRAME_MIN_ABOVE_FLOOR_INLIERS = 8
KEYFRAME_MIN_PIXEL_COVERAGE_X = 0.15
KEYFRAME_MIN_PIXEL_COVERAGE_Y = 0.12
# A cluster only identifies a place mode.  The metric pose must come from one
# registration that can observe all planar degrees of freedom on its own.
KEYFRAME_MAX_TRANSLATION_STD_M = 0.025
KEYFRAME_MAX_YAW_STD_DEG = 0.5
KEYFRAME_CLUSTER_XY_M = 0.60
KEYFRAME_CLUSTER_YAW_DEG = 18.0
KEYFRAME_MIN_INDEPENDENT = 2
KEYFRAME_SINGLE_MIN_INLIERS = 24
KEYFRAME_SINGLE_MIN_RATIO = 0.65
KEYFRAME_SINGLE_MIN_ABOVE_FLOOR = 0
KEYFRAME_MAX_POSE_JUMP_M = 3.0
KEYFRAME_MAX_POSE_YAW_JUMP_DEG = 40.0
# 第二个相互冲突的度量簇达到主簇六成权重时，地点具有歧义。该比例只
# 比较传感器支持度，不依赖房型；宁可本帧不校正，也不能在重复房间中强选。
KEYFRAME_AMBIGUOUS_SECOND_WEIGHT_RATIO = 0.60

# 地点召回后的度量验证门。地点库只负责缩小搜索范围；只有多视点 RGB-D
# 几何同时通过这些门，才允许把候选交给位姿图或冻结后的定位器。门槛按
# 传感器观测覆盖和数值稳定性定义，不含任务、房间、轨迹或固定帧号。
PLACE_GEOMETRY_MIN_MATCHES = 12
PLACE_GEOMETRY_MIN_INLIERS = 12
PLACE_GEOMETRY_MIN_INLIER_RATIO = 0.45
PLACE_GEOMETRY_MAX_RMSE_M = 0.10
PLACE_GEOMETRY_MIN_INDEPENDENT = 2
PLACE_GEOMETRY_INDEPENDENT_GAP = 32
PLACE_GEOMETRY_INDEPENDENT_TRANSLATION_M = 0.35

# 长时地点召回使用在线历史关键帧自建的视觉词典。外观分数只负责提出
# 多个历史地点候选，不能直接修改位姿；实际授权仍由深度几何和跨时段一致性
# 完成。所有容量和间隔都是传感器/计算预算，不含任务名、场景名或固定帧号。
SEQUENCE_MIN_KEYFRAMES = 12
SEQUENCE_VISUAL_WORDS = 128
SEQUENCE_SAMPLE_PER_KEYFRAME = 96
SEQUENCE_KMEANS_ITERATIONS = 6
SEQUENCE_WINDOW_OBSERVATIONS = 6
SEQUENCE_QUERY_GAP = 4
SEQUENCE_MODEL_REFRESH_KEYFRAMES = 8
SEQUENCE_MAX_CANDIDATES = 6
# 两个候选至少相隔两个完整查询窗口，才算来自不同历史时段。这里必须按
# 真实帧号而不是关键帧数组下标判断，因为长期库满后会同时包含稀疏历史和
# 稠密近期条目，数组间距不再代表时间间距。
SEQUENCE_CANDIDATE_SEPARATION_FRAMES = (
    2 * SEQUENCE_WINDOW_OBSERVATIONS * SEQUENCE_QUERY_GAP
)
SEQUENCE_MIN_ROBUST_Z = 2.0
SEQUENCE_SPEED_RATIOS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
# 地面是大面积、跨地点重复且近似共面的纹理，不能作为地点词。该高度只
# 过滤外观召回；深度几何仍保留完整自由空间和竖直结构。
SEQUENCE_MIN_FEATURE_HEIGHT_M = 0.20

# 头部相机能看到机器人自己的前臂、夹爪和躯干。它们在机体系里几乎不动，
# 却会在不同位置的 RGB 帧之间给出“完美”的假匹配。地图投影已经使用同一
# 包络排除这些点；视觉里程和长期定位也必须遵守同一观测边界。
SELF_CLEAR_FORWARD_M = 0.95
SELF_CLEAR_HALF_WIDTH_M = 0.55
SELF_CLEAR_MIN_Z_M = 0.08

# 视觉前端默认优先使用当前进程可见的 CUDA。严格官方 interface 会继承单卡
# CUDA_VISIBLE_DEVICES，因此这里永远只看到那一张卡，不会跨卡抢别的仿真。
FEATURE_BACKEND_ENV = "BEHAVIOR_SLAM_FEATURE_BACKEND"
ODOMETRY_FEATURE_BACKEND_ENV = "BEHAVIOR_SLAM_ODOMETRY_FEATURE_BACKEND"
LOCALIZATION_FEATURE_BACKEND_ENV = "BEHAVIOR_SLAM_LOCALIZATION_FEATURE_BACKEND"
MATCH_BACKEND_ENV = "BEHAVIOR_SLAM_MATCH_BACKEND"
RANSAC_BACKEND_ENV = "BEHAVIOR_SLAM_RANSAC_BACKEND"
RANSAC_BACKEND_AUTO = "auto"
RANSAC_BACKEND_CUDA = "cuda"
RANSAC_BACKEND_NUMPY = "numpy"
RANSAC_BACKEND_DEFAULT = RANSAC_BACKEND_AUTO
FEATURE_BACKEND_AUTO = "auto"
FEATURE_BACKEND_CUDA = "cuda"
FEATURE_BACKEND_OPENCV = "opencv"
OPENCV_THREADS_ENV = "BEHAVIOR_SLAM_OPENCV_THREADS"
OPENCV_THREADS_DEFAULT = 1

_RANSAC_LAST_BACKEND = "uninitialized"
_RANSAC_LAST_DEVICE = "uninitialized"
_RANSAC_LAST_FALLBACK_REASON = ""


def _self_point_mask(points: np.ndarray) -> np.ndarray:
    """返回机体包络内的非地面点。输入必须是机体系 XYZ。"""
    points = np.asarray(points)
    return (
        (points[:, 0] <= SELF_CLEAR_FORWARD_M)
        & (np.abs(points[:, 1]) <= SELF_CLEAR_HALF_WIDTH_M)
        & (points[:, 2] > SELF_CLEAR_MIN_Z_M)
    )


@dataclass(frozen=True)
class RgbdMotion:
    forward_m: float
    left_m: float
    yaw_deg: float
    ratio_matches: int
    rgbd_pairs: int
    inliers: int
    inlier_ratio: float
    rmse_m: float
    qvel_translation_error_m: float
    qvel_yaw_error_deg: float
    # 以下量只描述本次 RGB-D 刚体解自身的几何可观测性，不使用地图、任务
    # 标签或真值。给默认值是为了兼容测试替身和旧的诊断脚本。
    geometry_major_span_m: float = 0.0
    geometry_minor_span_m: float = 0.0
    pixel_coverage_x: float = 0.0
    pixel_coverage_y: float = 0.0
    translation_std_m: float = math.inf
    yaw_std_deg: float = math.inf
    normal_matrix_condition: float = math.inf


@dataclass
class _Frame:
    pixels: np.ndarray
    descriptors: Optional[np.ndarray]
    depth: np.ndarray
    camera: dict[str, Any]


def _se2_observability(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    pixels: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    fx: float,
) -> dict[str, float]:
    """从内点几何估计 SE(2) 解的覆盖和局部协方差。

    协方差来自点到点残差的一阶雅可比。噪声下限由一个像素在当前中值
    距离对应的米数给出，避免仿真中的近零残差产生虚假的无限置信度。
    所有输出都是观测自身的性质，不依赖场景名、轨迹长度或标注点。
    """
    source = np.asarray(source_xy, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    matched_pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    count = min(len(source), len(target), len(matched_pixels))
    if count < 2:
        return {
            "geometry_major_span_m": 0.0,
            "geometry_minor_span_m": 0.0,
            "pixel_coverage_x": 0.0,
            "pixel_coverage_y": 0.0,
            "translation_std_m": math.inf,
            "yaw_std_deg": math.inf,
            "normal_matrix_condition": math.inf,
        }
    source = source[:count]
    target = target[:count]
    matched_pixels = matched_pixels[:count]

    centered = source - np.mean(source, axis=0, keepdims=True)
    _eigenvalues, axes = np.linalg.eigh(centered.T @ centered / count)
    projected = centered @ axes
    spans = np.quantile(projected, 0.95, axis=0) - np.quantile(
        projected, 0.05, axis=0
    )
    minor_span = float(max(0.0, np.min(spans)))
    major_span = float(max(0.0, np.max(spans)))

    transformed = (np.asarray(rotation, dtype=np.float64) @ source.T).T
    predicted = transformed + np.asarray(translation, dtype=np.float64)
    residual = predicted - target
    # d(Rp+t)/d(tx,ty,theta)，theta 以弧度计。
    jacobian = np.zeros((2 * count, 3), dtype=np.float64)
    jacobian[0::2, 0] = 1.0
    jacobian[1::2, 1] = 1.0
    jacobian[0::2, 2] = -transformed[:, 1]
    jacobian[1::2, 2] = transformed[:, 0]
    normal = jacobian.T @ jacobian

    ranges = np.linalg.norm(source, axis=1)
    median_range = float(np.median(ranges)) if len(ranges) else 0.0
    focal = max(1.0, float(fx))
    # 特征位置约有一个像素量级的不确定性；5mm 是线性深度和数值误差下限。
    sensor_sigma_m = max(0.005, median_range / focal)
    dof = max(1, 2 * count - 3)
    empirical_variance = float(np.sum(residual * residual) / dof)
    variance = max(sensor_sigma_m * sensor_sigma_m, empirical_variance)
    try:
        covariance = variance * np.linalg.inv(normal)
        condition = float(np.linalg.cond(normal))
    except np.linalg.LinAlgError:
        covariance = np.full((3, 3), math.inf, dtype=np.float64)
        condition = math.inf
    translation_std = math.sqrt(max(
        0.0,
        float(max(covariance[0, 0], covariance[1, 1])),
    ))
    yaw_std_deg = math.degrees(math.sqrt(max(0.0, float(covariance[2, 2]))))
    pixel_span = np.ptp(matched_pixels, axis=0)
    return {
        "geometry_major_span_m": major_span,
        "geometry_minor_span_m": minor_span,
        "pixel_coverage_x": float(pixel_span[0] / max(1, int(image_width))),
        "pixel_coverage_y": float(pixel_span[1] / max(1, int(image_height))),
        "translation_std_m": float(translation_std),
        "yaw_std_deg": float(yaw_std_deg),
        "normal_matrix_condition": condition,
    }


@dataclass(frozen=True)
class ImageFeatures:
    """一帧可在相邻里程与长期定位之间复用的局部特征。"""

    pixels: np.ndarray
    descriptors: np.ndarray
    backend: str


@dataclass(frozen=True)
class _DescriptorMatch:
    """与 ``cv2.DMatch`` 相同的最小字段集，供 CUDA 匹配结果复用。"""

    queryIdx: int
    trainIdx: int
    distance: float


class _OpenCvSiftExtractor:
    backend = "opencv_sift_cpu"
    device = "cpu"

    def __init__(self, nfeatures: int, *, fallback_reason: str = "") -> None:
        import cv2

        self.fallback_reason = str(fallback_reason)
        self.frames = 0
        self._detector = cv2.SIFT_create(
            nfeatures=int(nfeatures),
            contrastThreshold=0.01,
        )

    def extract(self, gray: np.ndarray) -> Optional[ImageFeatures]:
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        self.frames += 1
        if descriptors is None or not keypoints:
            return None
        pixels = np.asarray([point.pt for point in keypoints], dtype=np.float64)
        return ImageFeatures(
            pixels=pixels,
            descriptors=np.ascontiguousarray(descriptors, dtype=np.float32),
            backend=self.backend,
        )


class _KorniaSiftExtractor:
    """无学习权重的 CUDA DoG + SIFT；不下载模型，也不读取地图外信息。"""

    backend = "kornia_sift_cuda"

    def __init__(self, nfeatures: int, *, device: str = "cuda:0") -> None:
        import kornia
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch 看不到 CUDA")
        self._kornia = kornia
        self._torch = torch
        self._device = torch.device(device)
        self.device = str(self._device)
        self.fallback_reason = ""
        self.frames = 0
        self._model = kornia.feature.SIFTFeature(
            num_features=int(nfeatures),
            upright=False,
            rootsift=False,
            device=self._device,
        ).eval()

    def extract(self, gray: np.ndarray) -> Optional[ImageFeatures]:
        image = self._torch.from_numpy(np.ascontiguousarray(gray)).to(
            device=self._device,
            dtype=self._torch.float32,
        )[None, None]
        image = image / 255.0
        with self._torch.inference_mode():
            lafs, _responses, descriptors = self._model(image)
            centers = self._kornia.feature.get_laf_center(lafs)[0]
        self.frames += 1
        if int(descriptors.shape[1]) <= 0:
            return None
        # 后续几何门的数据量很小；只把关键点和描述子搬回 CPU，RGB/depth
        # 仍在调用方的合规观测内，不在 GPU 上保留整段录制。
        pixels = centers.detach().cpu().numpy().astype(np.float64, copy=False)
        desc = descriptors[0].detach().cpu().numpy().astype(np.float32, copy=False)
        return ImageFeatures(
            pixels=np.ascontiguousarray(pixels),
            descriptors=np.ascontiguousarray(desc),
            backend=self.backend,
        )


def _opencv_thread_limit() -> int:
    try:
        return max(1, int(os.environ.get(
            OPENCV_THREADS_ENV, str(OPENCV_THREADS_DEFAULT)
        )))
    except (TypeError, ValueError):
        return OPENCV_THREADS_DEFAULT


def _make_sift_extractor(
    nfeatures: int,
    *,
    backend_env: str = FEATURE_BACKEND_ENV,
    default_backend: str = FEATURE_BACKEND_AUTO,
):
    """建立视觉前端；CUDA 不可用时明确回退到受限线程的 OpenCV。"""
    import cv2

    # OpenCV 默认会取满整机核心。即使 CUDA 前端生效，基础矩阵和 BFMatcher
    # 仍走 OpenCV，因此这里统一钳住线程数，避免一次建图把仿真 FPS 拖垮。
    cv2.setNumThreads(_opencv_thread_limit())
    # 通用变量可显式覆盖两条链；否则相邻里程默认保精度走 OpenCV，长期
    # 定位默认优先 CUDA。两者不能绑死：上面的基准已经证明快速 GPU SIFT
    # 直接替换帧间里程会显著增加双墙。
    requested = os.environ.get(
        FEATURE_BACKEND_ENV,
        os.environ.get(backend_env, default_backend),
    ).strip().lower()
    if requested not in {
        FEATURE_BACKEND_AUTO,
        FEATURE_BACKEND_CUDA,
        FEATURE_BACKEND_OPENCV,
    }:
        requested = FEATURE_BACKEND_AUTO
    fallback_reason = ""
    if requested in {FEATURE_BACKEND_AUTO, FEATURE_BACKEND_CUDA}:
        try:
            return _KorniaSiftExtractor(nfeatures)
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
    return _OpenCvSiftExtractor(
        nfeatures,
        fallback_reason=fallback_reason,
    )


def _cuda_knn_ratio_matches(
    query_descriptors: np.ndarray,
    train_descriptors: np.ndarray,
    *,
    ratio_limit: float,
    device_name: str = "cuda:0",
) -> list[_DescriptorMatch]:
    """在 CUDA 上做与 OpenCV BFMatcher 等价的二近邻比例测试。"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 看不到 CUDA")
    query_array = np.ascontiguousarray(query_descriptors, dtype=np.float32)
    train_array = np.ascontiguousarray(train_descriptors, dtype=np.float32)
    if len(query_array) == 0 or len(train_array) < 2:
        return []
    # 只限制 PyTorch 的 CPU 辅助线程；距离矩阵本身在 CUDA 上计算。
    torch.set_num_threads(1)
    device = torch.device(device_name)
    train = torch.from_numpy(train_array).to(device=device, non_blocking=True)
    output: list[_DescriptorMatch] = []
    with torch.inference_mode():
        for first in range(0, len(query_array), 1024):
            query = torch.from_numpy(query_array[first:first + 1024]).to(
                device=device,
                non_blocking=True,
            )
            distances = torch.cdist(query, train, p=2.0)
            values, indices = torch.topk(
                distances,
                k=2,
                dim=1,
                largest=False,
                sorted=True,
            )
            accepted = values[:, 0] < float(ratio_limit) * values[:, 1]
            selected = torch.nonzero(accepted, as_tuple=False).flatten()
            if int(selected.numel()) == 0:
                continue
            query_indices = (selected + first).detach().cpu().numpy()
            train_indices = indices[selected, 0].detach().cpu().numpy()
            nearest = values[selected, 0].detach().cpu().numpy()
            output.extend(
                _DescriptorMatch(
                    queryIdx=int(query_index),
                    trainIdx=int(train_index),
                    distance=float(distance),
                )
                for query_index, train_index, distance in zip(
                    query_indices,
                    train_indices,
                    nearest,
                )
            )
    return output


def _wrap_rad(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _gray_u8(rgb: np.ndarray) -> np.ndarray:
    import cv2

    image = np.asarray(rgb)
    if image.ndim == 2:
        return np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("RGB 必须是 HxWx3")
    source = np.ascontiguousarray(image[:, :, :3])
    if source.dtype != np.uint8:
        finite = source[np.isfinite(source)]
        if finite.size == 0:
            raise ValueError("RGB 没有有限像素")
        if np.issubdtype(source.dtype, np.floating) and float(finite.max()) <= 1.5:
            source = source * 255.0
        source = np.nan_to_num(source, nan=0.0, posinf=255.0, neginf=0.0)
        source = np.clip(source, 0, 255).astype(np.uint8)
    return cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)


def _camera_intrinsics(
    camera: dict[str, Any],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    focal_length = float(camera.get("focal_length", 17.0))
    aperture = float(camera.get("horizontal_aperture", 40.0))
    fallback_fx = focal_length * width / aperture if aperture > 1e-9 else 306.0
    return (
        float(camera.get("fx", fallback_fx)),
        float(camera.get("fy", fallback_fx)),
        float(camera.get("cx", width * 0.5)),
        float(camera.get("cy", height * 0.5)),
    )


def _unproject(
    pixels: np.ndarray,
    depth: np.ndarray,
    camera: dict[str, Any],
    *,
    minimum_depth_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    from behavior_interface.skills.reach_point_pitch_recovery import (
        quat_to_mat_xyzw,
    )

    arr = np.asarray(depth, dtype=np.float64).squeeze()
    if arr.ndim != 2:
        raise ValueError("depth 必须是二维数组")
    height, width = arr.shape
    cols = np.rint(pixels[:, 0]).astype(np.int64)
    rows = np.rint(pixels[:, 1]).astype(np.int64)
    inside = (cols >= 1) & (cols < width - 1) & (rows >= 1) & (rows < height - 1)
    sampled = np.full(len(pixels), np.nan, dtype=np.float64)
    indices = np.flatnonzero(inside)
    if indices.size:
        patch = np.stack(
            [
                arr[rows[indices] + dy, cols[indices] + dx]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ],
            axis=1,
        )
        patch_valid = np.isfinite(patch) & (patch > 1e-4)
        count = np.count_nonzero(patch_valid, axis=1)
        low = np.min(np.where(patch_valid, patch, np.inf), axis=1)
        high = np.max(np.where(patch_valid, patch, -np.inf), axis=1)
        stable = (count >= 5) & ((high - low) <= MAX_DEPTH_PATCH_SPAN_M)
        if np.any(stable):
            # 每行最多 9 个值；一次 nanmedian 比逐关键点进入 Python 快得多。
            values = np.where(patch_valid[stable], patch[stable], np.nan)
            sampled[indices[stable]] = np.nanmedian(values, axis=1)
    valid = (
        np.isfinite(sampled)
        & (sampled >= float(minimum_depth_m))
        & (sampled <= MAX_DEPTH_M)
    )
    fx, fy, cx, cy = _camera_intrinsics(camera, width, height)
    camera_points = np.column_stack([
        (pixels[:, 0] - cx) / fx * sampled,
        -(pixels[:, 1] - cy) / fy * sampled,
        -sampled,
    ])
    relative = dict(camera.get("robot_relative_pose") or {})
    position = np.asarray(relative.get("pos"), dtype=np.float64).reshape(3)
    rotation = quat_to_mat_xyzw(relative.get("quat"))
    robot_points = position + (rotation @ camera_points.T).T
    valid &= np.all(np.isfinite(robot_points), axis=1)
    return robot_points, valid


def _solve_se2(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    return rotation, target_center - rotation @ source_center


def _ransac_se2_numpy(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(source) < MIN_RGBD_PAIRS:
        raise ValueError("有效 RGB-D 对应不足")
    rng = np.random.default_rng(RANSAC_RANDOM_SEED)
    # 采样序列保持不变，只把每个假设逐一进入 Python 的评分改成一次数组
    # 运算。这里通常只有几十到几百个对应，400xN 的布尔矩阵远小于一帧
    # RGB；它能显著减少在线线程占用，又不改变 RANSAC 的证据或门槛。
    samples = np.asarray([
        rng.choice(len(source), size=2, replace=False)
        for _ in range(RANSAC_ITERATIONS)
    ], dtype=np.int64)
    source_vectors = source[samples[:, 1]] - source[samples[:, 0]]
    target_vectors = target[samples[:, 1]] - target[samples[:, 0]]
    valid_hypothesis = (
        np.linalg.norm(source_vectors, axis=1) >= 0.15
    ) & (
        np.linalg.norm(target_vectors, axis=1) >= 0.15
    )
    yaws = np.arctan2(target_vectors[:, 1], target_vectors[:, 0]) - np.arctan2(
        source_vectors[:, 1], source_vectors[:, 0]
    )
    cos_yaw = np.cos(yaws)
    sin_yaw = np.sin(yaws)
    source_anchor = source[samples[:, 0]]
    target_anchor = target[samples[:, 0]]
    translated_x = (
        target_anchor[:, 0]
        - cos_yaw * source_anchor[:, 0]
        + sin_yaw * source_anchor[:, 1]
    )
    translated_y = (
        target_anchor[:, 1]
        - sin_yaw * source_anchor[:, 0]
        - cos_yaw * source_anchor[:, 1]
    )
    predicted_x = (
        cos_yaw[:, None] * source[None, :, 0]
        - sin_yaw[:, None] * source[None, :, 1]
        + translated_x[:, None]
    )
    predicted_y = (
        sin_yaw[:, None] * source[None, :, 0]
        + cos_yaw[:, None] * source[None, :, 1]
        + translated_y[:, None]
    )
    residual_sq = (
        (predicted_x - target[None, :, 0]) ** 2
        + (predicted_y - target[None, :, 1]) ** 2
    )
    hypotheses = residual_sq <= RANSAC_THRESHOLD_M ** 2
    hypotheses[~valid_hypothesis] = False
    counts = np.count_nonzero(hypotheses, axis=1)
    best_index = int(np.argmax(counts))
    best = hypotheses[best_index]
    if np.count_nonzero(best) < MIN_RGBD_PAIRS:
        raise ValueError("SE(2) RANSAC 没有稳定解")
    rotation, translation = _solve_se2(source[best], target[best])
    residual = np.linalg.norm(
        (rotation @ source.T).T + translation - target,
        axis=1,
    )
    best = residual <= RANSAC_THRESHOLD_M
    rotation, translation = _solve_se2(source[best], target[best])
    return rotation, translation, best


def _ransac_se2_cuda(
    source: np.ndarray,
    target: np.ndarray,
    *,
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 CUDA 上批量评分同一组确定性 SE(2) 假设。"""
    import torch

    if len(source) < MIN_RGBD_PAIRS:
        raise ValueError("有效 RGB-D 对应不足")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 看不到 CUDA")
    torch.set_num_threads(1)
    rng = np.random.default_rng(RANSAC_RANDOM_SEED)
    samples = np.asarray([
        rng.choice(len(source), size=2, replace=False)
        for _ in range(RANSAC_ITERATIONS)
    ], dtype=np.int64)
    cuda_device = torch.device(str(device))
    if cuda_device.type != "cuda":
        raise RuntimeError("SE(2) RANSAC requires a CUDA device")
    source_tensor = torch.from_numpy(
        np.ascontiguousarray(source, dtype=np.float64)
    ).to(device=cuda_device, dtype=torch.float64, non_blocking=True)
    target_tensor = torch.from_numpy(
        np.ascontiguousarray(target, dtype=np.float64)
    ).to(device=cuda_device, dtype=torch.float64, non_blocking=True)
    sample_tensor = torch.from_numpy(samples).to(
        device=cuda_device, dtype=torch.long, non_blocking=True
    )
    with torch.inference_mode():
        source_vectors = (
            source_tensor[sample_tensor[:, 1]]
            - source_tensor[sample_tensor[:, 0]]
        )
        target_vectors = (
            target_tensor[sample_tensor[:, 1]]
            - target_tensor[sample_tensor[:, 0]]
        )
        valid_hypothesis = (
            torch.linalg.vector_norm(source_vectors, dim=1) >= 0.15
        ) & (
            torch.linalg.vector_norm(target_vectors, dim=1) >= 0.15
        )
        yaws = (
            torch.atan2(target_vectors[:, 1], target_vectors[:, 0])
            - torch.atan2(source_vectors[:, 1], source_vectors[:, 0])
        )
        cos_yaw, sin_yaw = torch.cos(yaws), torch.sin(yaws)
        source_anchor = source_tensor[sample_tensor[:, 0]]
        target_anchor = target_tensor[sample_tensor[:, 0]]
        translated_x = (
            target_anchor[:, 0]
            - cos_yaw * source_anchor[:, 0]
            + sin_yaw * source_anchor[:, 1]
        )
        translated_y = (
            target_anchor[:, 1]
            - sin_yaw * source_anchor[:, 0]
            - cos_yaw * source_anchor[:, 1]
        )
        predicted_x = (
            cos_yaw[:, None] * source_tensor[None, :, 0]
            - sin_yaw[:, None] * source_tensor[None, :, 1]
            + translated_x[:, None]
        )
        predicted_y = (
            sin_yaw[:, None] * source_tensor[None, :, 0]
            + cos_yaw[:, None] * source_tensor[None, :, 1]
            + translated_y[:, None]
        )
        residual_sq = (
            (predicted_x - target_tensor[None, :, 0]).square()
            + (predicted_y - target_tensor[None, :, 1]).square()
        )
        hypotheses = residual_sq <= RANSAC_THRESHOLD_M ** 2
        hypotheses[~valid_hypothesis] = False
        counts = torch.count_nonzero(hypotheses, dim=1)
        best_index = int(torch.argmax(counts).item())
        best = hypotheses[best_index].detach().cpu().numpy().astype(bool, copy=False)
    torch.cuda.synchronize(cuda_device)
    if np.count_nonzero(best) < MIN_RGBD_PAIRS:
        raise ValueError("SE(2) RANSAC 没有稳定解")
    # 精修仍复用原 NumPy SVD，保证最终位姿的数值口径完全相同。
    rotation, translation = _solve_se2(source[best], target[best])
    residual = np.linalg.norm(
        (rotation @ source.T).T + translation - target,
        axis=1,
    )
    best = residual <= RANSAC_THRESHOLD_M
    rotation, translation = _solve_se2(source[best], target[best])
    return rotation, translation, best


def _ransac_se2(
    source: np.ndarray,
    target: np.ndarray,
    *,
    require_cuda: bool = False,
    cuda_device: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按运行环境选择 RANSAC 评分设备。

    普通里程和定位保留原来的 NumPy 降级语义。长期地点验证把
    ``require_cuda`` 设为真时，CUDA 不可用或执行失败都是 UNKNOWN；绝不
    静默切到 CPU 后用另一条数值路径授权回环。
    """
    global _RANSAC_LAST_BACKEND
    global _RANSAC_LAST_DEVICE
    global _RANSAC_LAST_FALLBACK_REASON
    selected_cuda_device = str(cuda_device or "cuda:0")

    if require_cuda:
        try:
            result = _ransac_se2_cuda(
                source, target, device=selected_cuda_device
            )
            _RANSAC_LAST_BACKEND = "torch_cuda"
            _RANSAC_LAST_DEVICE = selected_cuda_device
            _RANSAC_LAST_FALLBACK_REASON = ""
            return result
        except Exception as exc:
            _RANSAC_LAST_BACKEND = "torch_cuda_failed"
            _RANSAC_LAST_DEVICE = selected_cuda_device
            _RANSAC_LAST_FALLBACK_REASON = f"{type(exc).__name__}: {exc}"
            raise

    requested = str(
        os.environ.get(RANSAC_BACKEND_ENV, RANSAC_BACKEND_DEFAULT) or ""
    ).strip().lower()
    if requested not in {
        RANSAC_BACKEND_AUTO,
        RANSAC_BACKEND_CUDA,
        RANSAC_BACKEND_NUMPY,
    }:
        requested = RANSAC_BACKEND_AUTO
    if requested != RANSAC_BACKEND_NUMPY:
        try:
            result = _ransac_se2_cuda(
                source, target, device=selected_cuda_device
            )
            _RANSAC_LAST_BACKEND = "torch_cuda"
            _RANSAC_LAST_DEVICE = selected_cuda_device
            _RANSAC_LAST_FALLBACK_REASON = ""
            return result
        except Exception as exc:
            _RANSAC_LAST_FALLBACK_REASON = f"{type(exc).__name__}: {exc}"
    else:
        _RANSAC_LAST_FALLBACK_REASON = ""
    _RANSAC_LAST_BACKEND = "numpy_cpu"
    _RANSAC_LAST_DEVICE = "cpu"
    return _ransac_se2_numpy(source, target)


class RgbdOdometry:
    """相邻 RGB-D 帧里程；证据不足时返回 None，让调用方保留 qvel。"""

    def __init__(self, *, nfeatures: int = 3000, feature_extractor=None) -> None:
        self._feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else _make_sift_extractor(
                int(nfeatures),
                backend_env=ODOMETRY_FEATURE_BACKEND_ENV,
                default_backend=FEATURE_BACKEND_OPENCV,
            )
        )
        self._previous: Optional[_Frame] = None
        self.last_features: Optional[ImageFeatures] = None
        self.last_motion: Optional[RgbdMotion] = None
        self.accepted = 0
        self.rejected = 0
        self.last_reason = ""
        self.matching_backend = "uninitialized"
        self.matching_device = "uninitialized"
        self.matching_fallback_reason = ""
        self.ransac_backend = "uninitialized"
        self.ransac_device = "uninitialized"
        self.ransac_fallback_reason = ""

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def feature_backend(self) -> str:
        return str(getattr(self._feature_extractor, "backend", "unknown"))

    @property
    def feature_device(self) -> str:
        return str(getattr(self._feature_extractor, "device", "unknown"))

    @property
    def feature_fallback_reason(self) -> str:
        return str(getattr(self._feature_extractor, "fallback_reason", ""))

    def reset(self) -> None:
        self._previous = None
        self.last_features = None
        self.last_motion = None
        self.accepted = 0
        self.rejected = 0
        self.last_reason = ""
        self.matching_backend = "uninitialized"
        self.matching_device = "uninitialized"
        self.matching_fallback_reason = ""
        self.ransac_backend = "uninitialized"
        self.ransac_device = "uninitialized"
        self.ransac_fallback_reason = ""

    def _ratio_matches(
        self,
        previous_descriptors: np.ndarray,
        current_descriptors: np.ndarray,
    ) -> list[Any]:
        """优先在 CUDA 上匹配；失败时显式记录并退回单线程 OpenCV。"""
        import cv2

        requested = str(
            os.environ.get(MATCH_BACKEND_ENV, FEATURE_BACKEND_AUTO) or ""
        ).strip().lower()
        if requested not in {
            FEATURE_BACKEND_AUTO,
            FEATURE_BACKEND_CUDA,
            FEATURE_BACKEND_OPENCV,
        }:
            requested = FEATURE_BACKEND_AUTO
        fallback_reason = ""
        if requested in {FEATURE_BACKEND_AUTO, FEATURE_BACKEND_CUDA}:
            try:
                matches = _cuda_knn_ratio_matches(
                    previous_descriptors,
                    current_descriptors,
                    ratio_limit=RATIO_LIMIT,
                )
                self.matching_backend = "torch_cdist_cuda"
                self.matching_device = "cuda:0"
                self.matching_fallback_reason = ""
                return matches
            except Exception as exc:
                fallback_reason = f"{type(exc).__name__}: {exc}"
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
            previous_descriptors,
            current_descriptors,
            k=2,
        )
        self.matching_backend = "opencv_bf_cpu"
        self.matching_device = "cpu"
        self.matching_fallback_reason = fallback_reason
        return [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < RATIO_LIMIT * second.distance
        ]

    def _frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera: dict[str, Any],
    ) -> _Frame:
        gray = _gray_u8(rgb)
        features = self._feature_extractor.extract(gray)
        self.last_features = features
        return _Frame(
            pixels=(
                np.zeros((0, 2), dtype=np.float64)
                if features is None
                else features.pixels
            ),
            descriptors=None if features is None else features.descriptors,
            depth=np.asarray(depth),
            camera=dict(camera),
        )

    def update(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera: dict[str, Any],
        *,
        qvel_delta: tuple[float, float, float],
    ) -> Optional[RgbdMotion]:
        import cv2

        self.last_motion = None
        current = self._frame(rgb, depth, camera)
        previous = self._previous
        self._previous = current
        if previous is None:
            self.last_reason = "first_frame"
            return None
        if previous.descriptors is None or current.descriptors is None:
            self.rejected += 1
            self.last_reason = "no_descriptors"
            return None
        matches = self._ratio_matches(
            previous.descriptors,
            current.descriptors,
        )
        if len(matches) < MIN_RATIO_MATCHES:
            self.rejected += 1
            self.last_reason = "few_ratio_matches"
            return None
        previous_pixels = np.asarray(
            [previous.pixels[match.queryIdx] for match in matches],
            dtype=np.float64,
        )
        current_pixels = np.asarray(
            [current.pixels[match.trainIdx] for match in matches],
            dtype=np.float64,
        )
        _fundamental, mask = cv2.findFundamentalMat(
            previous_pixels,
            current_pixels,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=1.5,
            confidence=0.999,
        )
        if mask is None:
            self.rejected += 1
            self.last_reason = "no_fundamental"
            return None
        epipolar = mask.reshape(-1).astype(bool)
        previous_robot, previous_valid = _unproject(
            previous_pixels, previous.depth, previous.camera
        )
        current_robot, current_valid = _unproject(
            current_pixels, current.depth, current.camera
        )
        valid = epipolar & previous_valid & current_valid
        # 机体自身的前臂、夹爪和躯干会在相邻帧中保持相似外观；它们不是环境
        # 约束，必须在视觉里程求解前排除，否则会产生近零的假运动。
        valid &= ~_self_point_mask(previous_robot)
        valid &= ~_self_point_mask(current_robot)
        valid &= (
            np.abs(previous_robot[:, 2] - current_robot[:, 2])
            <= MAX_HEIGHT_DELTA_M
        )
        source = current_robot[valid, :2]
        target = previous_robot[valid, :2]
        geometry_pixels = current_pixels[valid]
        if len(source) < MIN_RGBD_PAIRS:
            self.rejected += 1
            self.last_reason = "few_rgbd_pairs"
            return None
        try:
            rotation, translation, inliers = _ransac_se2(source, target)
        except ValueError:
            self.ransac_backend = _RANSAC_LAST_BACKEND
            self.ransac_device = _RANSAC_LAST_DEVICE
            self.ransac_fallback_reason = _RANSAC_LAST_FALLBACK_REASON
            self.rejected += 1
            self.last_reason = "ransac_failed"
            return None
        self.ransac_backend = _RANSAC_LAST_BACKEND
        self.ransac_device = _RANSAC_LAST_DEVICE
        self.ransac_fallback_reason = _RANSAC_LAST_FALLBACK_REASON
        residual = np.linalg.norm(
            (rotation @ source[inliers].T).T
            + translation
            - target[inliers],
            axis=1,
        )
        yaw_rad = math.atan2(rotation[1, 0], rotation[0, 0])
        qx, qy, qyaw_deg = [float(value) for value in qvel_delta]
        translation_error = math.hypot(
            float(translation[0]) - qx,
            float(translation[1]) - qy,
        )
        yaw_error_deg = abs(math.degrees(_wrap_rad(
            yaw_rad - math.radians(qyaw_deg)
        )))
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = float(np.mean(inliers))
        rmse_m = float(np.sqrt(np.mean(residual ** 2)))
        depth_shape = np.asarray(current.depth).squeeze().shape
        image_height, image_width = (
            (int(depth_shape[0]), int(depth_shape[1]))
            if len(depth_shape) == 2
            else (720, 720)
        )
        fx, _fy, _cx, _cy = _camera_intrinsics(
            current.camera,
            image_width,
            image_height,
        )
        observability = _se2_observability(
            source[inliers],
            target[inliers],
            rotation,
            translation,
            geometry_pixels[inliers],
            image_width=image_width,
            image_height=image_height,
            fx=fx,
        )
        reliable = (
            inlier_count >= MIN_SE2_INLIERS
            and inlier_ratio >= MIN_SE2_INLIER_RATIO
            and rmse_m <= MAX_SE2_RMSE_M
            and translation_error <= MAX_QVEL_TRANSLATION_ERROR_M
            and yaw_error_deg <= MAX_QVEL_YAW_ERROR_DEG
        )
        if not reliable:
            self.rejected += 1
            self.last_reason = "quality_gate"
            return None
        self.accepted += 1
        self.last_reason = "accepted"
        motion = RgbdMotion(
            forward_m=float(translation[0]),
            left_m=float(translation[1]),
            yaw_deg=math.degrees(yaw_rad),
            ratio_matches=len(matches),
            rgbd_pairs=len(source),
            inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            rmse_m=rmse_m,
            qvel_translation_error_m=translation_error,
            qvel_yaw_error_deg=yaw_error_deg,
            **observability,
        )
        self.last_motion = motion
        return motion


@dataclass(frozen=True)
class VisualLocalization:
    """一次长期关键帧定位的结果。

    ``keyframe_pose`` 是被匹配关键帧在地图中的位姿，``pose`` 是当前帧在同
    一地图中的位姿。所有数值都来自 RGB-D 对应的 SE(2) 解；类本身不持有或
    修改占据栅格。
    """

    x_m: float
    y_m: float
    yaw_deg: float
    keyframe_id: str
    ratio_matches: int
    rgbd_pairs: int
    inliers: int
    inlier_ratio: float
    rmse_m: float
    candidate_count: int
    independent_candidates: int
    keyframe_frame_index: int = -1
    oldest_candidate_frame_index: int = -1
    candidate_frame_gap: int = 0
    above_floor_inliers: int = 0
    independent_translation_span_m: float = 0.0
    # 当前机体系在代表关键帧机体系里的相对位姿。位姿图使用这个观测，
    # 而不是从两个可能已经漂移的绝对位姿相减。
    keyframe_to_current_x_m: float = 0.0
    keyframe_to_current_y_m: float = 0.0
    keyframe_to_current_yaw_deg: float = 0.0
    # 仅用于审计证据来源；默认值保持旧调用方的直接 RGB-D 语义。
    evidence_type: str = "direct_rgbd_geometry"
    place_geometry_candidates: int = 0
    place_geometry_independent: int = 0
    geometry_major_span_m: float = 0.0
    geometry_minor_span_m: float = 0.0
    pixel_coverage_x: float = 0.0
    pixel_coverage_y: float = 0.0
    translation_std_m: float = math.inf
    yaw_std_deg: float = math.inf
    normal_matrix_condition: float = math.inf
    winning_cluster_candidates: int = 0
    winning_cluster_observable_candidates: int = 0
    witness_selection_reason: str = ""

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x_m, self.y_m, self.yaw_deg

    @property
    def keyframe_to_current_pose(self) -> tuple[float, float, float]:
        return (
            self.keyframe_to_current_x_m,
            self.keyframe_to_current_y_m,
            self.keyframe_to_current_yaw_deg,
        )


@dataclass(frozen=True)
class VisualPlaceCandidate:
    """连续外观序列提出的历史地点候选。

    ``pose`` 只是候选序列终点对应关键帧的地图位姿，不是测得的当前位姿。
    调用方必须把它送入独立深度几何验证，并保留所有竞争候选。
    """

    x_m: float
    y_m: float
    yaw_deg: float
    keyframe_id: str
    keyframe_frame_index: int
    candidate_frame_gap: int
    appearance_score: float
    robust_z: float
    sequence_observations: int
    direction: int
    speed_ratio: float
    rank: int
    candidate_count: int
    evidence_type: str = "appearance_sequence"

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x_m, self.y_m, self.yaw_deg


@dataclass
class _VisualKeyframe:
    image_id: str
    frame_index: int
    pose: tuple[float, float, float]
    pixels: np.ndarray
    descriptors: np.ndarray
    robot_points: np.ndarray
    appearance_descriptors: Optional[np.ndarray] = None
    appearance_backend: str = ""


@dataclass
class _PlaceKeyframe:
    """稠密地点检索条目，同时保留对齐的 RGB-D 点供二次验证。"""

    image_id: str
    frame_index: int
    pose: tuple[float, float, float]
    appearance_descriptors: np.ndarray
    appearance_backend: str
    # 默认空数组兼容旧诊断/单测手工构造的地点条目。
    pixels: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    robot_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )


@dataclass
class _AppearanceObservation:
    image_id: str
    frame_index: int
    descriptors: np.ndarray
    backend: str
    pixels: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    robot_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )


class FeatureKeyframeLocalizer:
    """基于 RGB-D 特征的长期定位器。

    关键帧只保存特征描述子和对应的机体系点，匹配结果只用于更新调用方的
    当前位姿。这里刻意没有 ``OccupancyGrid``、轨迹或路标接口，避免把视觉
    结果反向写成地图形状。
    """

    def __init__(
        self,
        *,
        nfeatures: int = KEYFRAME_MAX_KEYPOINTS,
        keyframe_gap: int = KEYFRAME_MIN_GAP,
        max_keyframes: int = KEYFRAME_MAX_COUNT,
        max_place_keyframes: int = PLACE_KEYFRAME_MAX_COUNT,
        ratio_limit: float = KEYFRAME_RATIO_LIMIT,
        feature_extractor=None,
    ) -> None:
        self.nfeatures = max(64, int(nfeatures))
        self._feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else _make_sift_extractor(
                self.nfeatures,
                backend_env=LOCALIZATION_FEATURE_BACKEND_ENV,
                default_backend=FEATURE_BACKEND_AUTO,
            )
        )
        self.keyframe_gap = max(1, int(keyframe_gap))
        self.max_keyframes = max(2, int(max_keyframes))
        self.max_place_keyframes = max(2, int(max_place_keyframes))
        self.ratio_limit = float(ratio_limit)
        self._keyframes: list[_VisualKeyframe] = []
        self._place_keyframes: list[_PlaceKeyframe] = []
        self.attempts = 0
        self.accepted = 0
        self.rejected = 0
        self.keyframes_added = 0
        self.last_reason = ""
        self.last_result: Optional[VisualLocalization] = None
        self.matching_backend = "opencv_bf_cpu"
        self.matching_device = "cpu"
        self.matching_fallback_reason = ""
        self._extract_cache_key = ""
        self._extract_cache_value: Optional[
            tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = None
        self._appearance_queries: list[_AppearanceObservation] = []
        self._sequence_database_generation = 0
        self._sequence_model_generation = -1
        self._sequence_force_refresh = False
        self._sequence_model: Optional[dict[str, Any]] = None
        self.last_place_candidates: tuple[VisualPlaceCandidate, ...] = ()
        self.sequence_attempts = 0
        self.sequence_candidates = 0
        self.sequence_last_reason = "not_ready"
        self.sequence_backend = "torch_cuda_bow_sequence"
        self.sequence_device = "uninitialized"
        self.place_geometry_attempts = 0
        self.place_geometry_accepted = 0
        self.place_geometry_rejected = 0
        self.place_geometry_last_reason = "not_attempted"
        self.place_geometry_reason_counts: dict[str, int] = {}
        self.last_place_geometry_rows: tuple[dict[str, Any], ...] = ()

    @property
    def keyframe_count(self) -> int:
        return len(self._keyframes)

    @property
    def keyframe_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([
            *(item.image_id for item in self._keyframes),
            *(item.image_id for item in self._place_keyframes),
        ]))

    @property
    def place_keyframe_count(self) -> int:
        return len(self._place_keyframes)

    @property
    def place_keyframe_ids(self) -> tuple[str, ...]:
        """Return the deterministic IDs currently backed by place RGB-D data."""
        return tuple(dict.fromkeys(
            item.image_id for item in self._place_keyframes
        ))

    @property
    def place_database_generation(self) -> int:
        return int(self._sequence_database_generation)

    @property
    def feature_backend(self) -> str:
        return str(getattr(self._feature_extractor, "backend", "unknown"))

    @property
    def feature_device(self) -> str:
        return str(getattr(self._feature_extractor, "device", "unknown"))

    @property
    def feature_fallback_reason(self) -> str:
        return str(getattr(self._feature_extractor, "fallback_reason", ""))

    def reset(self) -> None:
        self._keyframes.clear()
        self._place_keyframes.clear()
        self.attempts = 0
        self.accepted = 0
        self.rejected = 0
        self.keyframes_added = 0
        self.last_reason = ""
        self.last_result = None
        self.matching_backend = "opencv_bf_cpu"
        self.matching_device = "cpu"
        self.matching_fallback_reason = ""
        self._extract_cache_key = ""
        self._extract_cache_value = None
        self._appearance_queries.clear()
        self._sequence_database_generation = 0
        self._sequence_model_generation = -1
        self._sequence_force_refresh = False
        self._sequence_model = None
        self.last_place_candidates = ()
        self.sequence_attempts = 0
        self.sequence_candidates = 0
        self.sequence_last_reason = "not_ready"
        self.sequence_device = "uninitialized"
        self.place_geometry_attempts = 0
        self.place_geometry_accepted = 0
        self.place_geometry_rejected = 0
        self.place_geometry_last_reason = "not_attempted"
        self.place_geometry_reason_counts = {}
        self.last_place_geometry_rows = ()

    def update_keyframe_poses(
        self,
        poses: dict[str, tuple[float, float, float]],
    ) -> int:
        """位姿图移动 submap 后，同步关键帧在地图中的绝对位姿。"""
        updated = 0
        for keyframe in self._keyframes:
            pose = poses.get(keyframe.image_id)
            if pose is None:
                continue
            keyframe.pose = tuple(float(value) for value in pose)
            updated += 1
        for keyframe in self._place_keyframes:
            pose = poses.get(keyframe.image_id)
            if pose is None:
                continue
            keyframe.pose = tuple(float(value) for value in pose)
            updated += 1
        return updated

    def _prune_keyframes(self) -> None:
        """同时保留全程覆盖和近期视角，避免长路线遗忘早期地点。"""
        if len(self._keyframes) <= self.max_keyframes:
            return
        recent_count = max(2, self.max_keyframes // 2)
        old_slots = self.max_keyframes - recent_count
        recent = self._keyframes[-recent_count:]
        older = self._keyframes[:-recent_count]
        if len(older) <= old_slots:
            self._keyframes = [*older, *recent]
            return
        indices = np.linspace(
            0,
            len(older) - 1,
            num=old_slots,
        ).round().astype(np.int64)
        self._keyframes = [
            *[older[int(index)] for index in indices],
            *recent,
        ]

    def _prune_place_keyframes(self) -> None:
        """地点索引保留均匀历史与近期连续段，适配任意长度路线。"""
        if len(self._place_keyframes) <= self.max_place_keyframes:
            return
        recent_count = max(2, self.max_place_keyframes // 2)
        old_slots = self.max_place_keyframes - recent_count
        recent = self._place_keyframes[-recent_count:]
        older = self._place_keyframes[:-recent_count]
        indices = np.linspace(
            0,
            len(older) - 1,
            num=old_slots,
        ).round().astype(np.int64)
        self._place_keyframes = [
            *[older[int(index)] for index in indices],
            *recent,
        ]

    @staticmethod
    def _appearance_subset(
        image_features: Optional[ImageFeatures],
        depth: np.ndarray,
        camera: dict[str, Any],
        *,
        maximum: int,
    ) -> Optional[np.ndarray]:
        """从已提取特征中留下环境描述子，避免序列召回观察机器人自身。"""
        if (
            image_features is None
            or image_features.descriptors is None
            or len(image_features.pixels) == 0
        ):
            return None
        pixels = np.asarray(image_features.pixels, dtype=np.float64)
        descriptors = np.asarray(image_features.descriptors, dtype=np.float32)
        count = min(len(pixels), len(descriptors), max(1, int(maximum)))
        pixels = pixels[:count]
        descriptors = descriptors[:count]
        try:
            robot_points, valid = _unproject(
                pixels,
                depth,
                camera,
                minimum_depth_m=KEYFRAME_MIN_DEPTH_M,
            )
        except (KeyError, TypeError, ValueError):
            return None
        valid &= np.isfinite(robot_points).all(axis=1)
        valid &= ~_self_point_mask(robot_points)
        valid &= robot_points[:, 2] > SEQUENCE_MIN_FEATURE_HEIGHT_M
        valid &= (
            np.hypot(robot_points[:, 0], robot_points[:, 1])
            <= KEYFRAME_MAX_RANGE_M
        )
        if int(np.count_nonzero(valid)) < KEYFRAME_MIN_RGBD_PAIRS:
            return None
        return np.ascontiguousarray(descriptors[valid], dtype=np.float32)

    def _extract(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera: dict[str, Any],
        image_features: Optional[ImageFeatures] = None,
        cache_key: str = "",
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """提取特征并给每个描述子绑定一个可反解的机体系点。"""
        key = str(cache_key or "")
        if key and key == self._extract_cache_key:
            return self._extract_cache_value
        features = image_features
        source_backend = str(getattr(features, "backend", ""))
        if (
            features is not None
            and source_backend
            and source_backend != self.feature_backend
        ):
            # OpenCV SIFT 与 Kornia SIFT 的描述子不能交叉匹配。短程里程可
            # 复用同后端特征；不同后端必须由长期定位器独立提取。
            features = None
        if features is None:
            features = self._feature_extractor.extract(_gray_u8(rgb))
        if features is None or features.descriptors is None or len(features.pixels) == 0:
            if key:
                self._extract_cache_key = key
                self._extract_cache_value = None
            return None
        pixels = np.asarray(features.pixels, dtype=np.float64)
        descriptors = np.asarray(features.descriptors, dtype=np.float32)
        if len(pixels) > self.nfeatures:
            pixels = pixels[:self.nfeatures]
            descriptors = descriptors[:self.nfeatures]
        robot_points, valid = _unproject(
            pixels,
            depth,
            camera,
            minimum_depth_m=KEYFRAME_MIN_DEPTH_M,
        )
        valid &= np.isfinite(robot_points).all(axis=1)
        # 长期关键帧也必须使用与相邻帧里程相同的环境观测边界，不能把机器人
        # 自身当成可回访的固定地标。
        valid &= ~_self_point_mask(robot_points)
        valid &= np.hypot(robot_points[:, 0], robot_points[:, 1]) <= KEYFRAME_MAX_RANGE_M
        if int(np.count_nonzero(valid)) < KEYFRAME_MIN_RGBD_PAIRS:
            if key:
                self._extract_cache_key = key
                self._extract_cache_value = None
            return None
        extracted = (
            pixels[valid],
            np.ascontiguousarray(descriptors[valid], dtype=np.float32),
            np.asarray(robot_points[valid], dtype=np.float64),
        )
        if key:
            self._extract_cache_key = key
            self._extract_cache_value = extracted
        return extracted

    def add_keyframe(
        self,
        image_id: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera: dict[str, Any],
        pose: tuple[float, float, float],
        *,
        frame_index: Optional[int] = None,
        force: bool = False,
        image_features: Optional[ImageFeatures] = None,
        appearance_features: Optional[ImageFeatures] = None,
    ) -> bool:
        """在建图阶段加入一个关键帧；重复或证据不足时不加入。"""
        frame_no = int(frame_index if frame_index is not None else self.keyframes_added)
        key = str(image_id or f"frame-{frame_no}")
        if (
            not force
            and self._keyframes
            and frame_no - self._keyframes[-1].frame_index < self.keyframe_gap
        ):
            return False
        extracted = self._extract(
            rgb,
            depth,
            camera,
            image_features=image_features,
            cache_key=key,
        )
        if extracted is None:
            self.last_reason = "keyframe_no_rgbd_features"
            return False
        pixels, descriptors, robot_points = extracted
        item = _VisualKeyframe(
            image_id=key,
            frame_index=frame_no,
            pose=tuple(float(value) for value in pose),
            pixels=pixels,
            descriptors=descriptors,
            robot_points=robot_points,
        )
        if self._keyframes and self._keyframes[-1].image_id == key:
            self._keyframes[-1] = item
        else:
            self._keyframes.append(item)
            self.keyframes_added += 1
        self._prune_keyframes()
        self.last_reason = "keyframe_added"
        return True

    def add_place_keyframe(
        self,
        image_id: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera: dict[str, Any],
        pose: tuple[float, float, float],
        *,
        frame_index: Optional[int] = None,
        force: bool = False,
        image_features: Optional[ImageFeatures] = None,
    ) -> bool:
        """提交稠密外观条目；它只召回地点，不改变度量关键帧集合。"""
        frame_no = int(frame_index if frame_index is not None else self.keyframes_added)
        key = str(image_id or f"frame-{frame_no}")
        if not force and self._place_keyframes:
            previous = self._place_keyframes[-1]
            gap = frame_no - previous.frame_index
            if gap < PLACE_KEYFRAME_MIN_GAP:
                return False
            if gap < PLACE_KEYFRAME_MAX_GAP:
                translation = math.hypot(
                    float(pose[0]) - float(previous.pose[0]),
                    float(pose[1]) - float(previous.pose[1]),
                )
                yaw = abs(self._yaw_delta(
                    float(pose[2]), float(previous.pose[2])
                ))
                if (
                    translation < PLACE_KEYFRAME_MIN_TRANSLATION_M
                    and yaw < PLACE_KEYFRAME_MIN_YAW_DEG
                ):
                    return False
        extracted = self._extract(
            rgb,
            depth,
            camera,
            image_features=image_features,
            cache_key=key,
        )
        if extracted is None:
            self.sequence_last_reason = "place_keyframe_no_rgbd_features"
            return False
        pixels, descriptors, robot_points = extracted
        # 地点库和查询都使用长期 CUDA 前端，并去掉地板纹理。地板通常覆盖
        # 整个住宅，会给不同房间制造相同外观；它仍保留在独立深度几何门中。
        above_ground = robot_points[:, 2] > SEQUENCE_MIN_FEATURE_HEIGHT_M
        appearance = np.ascontiguousarray(
            descriptors[above_ground],
            dtype=np.float32,
        )
        if len(appearance) < KEYFRAME_MIN_RGBD_PAIRS:
            self.sequence_last_reason = "place_keyframe_above_ground_features_missing"
            return False
        item = _PlaceKeyframe(
            image_id=key,
            frame_index=frame_no,
            pose=tuple(float(value) for value in pose),
            appearance_descriptors=appearance,
            appearance_backend=self.feature_backend,
            pixels=np.ascontiguousarray(pixels[above_ground], dtype=np.float64),
            robot_points=np.ascontiguousarray(
                robot_points[above_ground], dtype=np.float64
            ),
        )
        if self._place_keyframes and self._place_keyframes[-1].image_id == key:
            self._place_keyframes[-1] = item
        else:
            self._place_keyframes.append(item)
        self._sequence_database_generation += 1
        self._prune_place_keyframes()
        self.sequence_last_reason = "place_keyframe_added"
        return True

    def seal_place_database(self) -> None:
        """通知地点库建图已冻结；下一次查询用全部现存关键帧重建索引。"""
        self._sequence_force_refresh = True

    def reset_query_sequence(self) -> None:
        """Drop transient query history while preserving the place database."""

        self._appearance_queries.clear()
        self.last_place_candidates = ()
        self.last_place_geometry_rows = ()
        self.last_result = None
        self.sequence_last_reason = "query_sequence_reset"

    @staticmethod
    def _sequence_training_sample(
        keyframes: Iterable[_PlaceKeyframe],
    ) -> np.ndarray:
        parts = []
        for keyframe in keyframes:
            values = np.asarray(
                keyframe.appearance_descriptors, dtype=np.float32
            )
            if len(values) > SEQUENCE_SAMPLE_PER_KEYFRAME:
                indices = np.linspace(
                    0,
                    len(values) - 1,
                    num=SEQUENCE_SAMPLE_PER_KEYFRAME,
                ).round().astype(np.int64)
                values = values[indices]
            if len(values):
                parts.append(values)
        if not parts:
            return np.empty((0, 128), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)

    @staticmethod
    def _sequence_histogram(
        descriptors: np.ndarray,
        centers: Any,
        *,
        torch: Any,
        device: Any,
    ) -> Any:
        import torch.nn.functional as functional

        histogram = torch.zeros(
            int(centers.shape[0]), device=device, dtype=torch.float32
        )
        values = np.asarray(descriptors, dtype=np.float32)
        if not len(values):
            return histogram
        with torch.inference_mode():
            for first in range(0, len(values), 2048):
                batch = torch.from_numpy(
                    np.ascontiguousarray(values[first:first + 2048])
                ).to(device=device, non_blocking=True)
                batch = functional.normalize(batch, dim=1)
                assignment = torch.argmax(batch @ centers.T, dim=1)
                histogram += torch.bincount(
                    assignment, minlength=int(centers.shape[0])
                ).float()
        return histogram

    def _build_sequence_model(self) -> bool:
        """在 CUDA 上从在线历史关键帧建立固定大小的词袋索引。"""
        try:
            import torch
            import torch.nn.functional as functional
        except Exception as exc:
            self.sequence_last_reason = (
                f"sequence_torch_import_failed:{type(exc).__name__}"
            )
            return False
        if not torch.cuda.is_available():
            self.sequence_last_reason = "sequence_cuda_unavailable"
            return False
        device_name = (
            self.feature_device
            if self.feature_device.startswith("cuda")
            else "cuda:0"
        )
        try:
            device = torch.device(device_name)
            torch.empty(1, device=device)
        except Exception as exc:
            self.sequence_last_reason = (
                f"sequence_cuda_device_failed:{type(exc).__name__}"
            )
            return False
        torch.set_num_threads(1)

        usable = [
            keyframe
            for keyframe in self._place_keyframes
            if keyframe.appearance_descriptors is not None
            and len(keyframe.appearance_descriptors) >= KEYFRAME_MIN_RGBD_PAIRS
        ]
        if len(usable) < SEQUENCE_MIN_KEYFRAMES:
            self.sequence_last_reason = "sequence_keyframes_not_ready"
            return False
        # 同一索引不能混用不同特征定义。正常在线运行只有一个共享里程前端；
        # 若中途切换后端，保留数量最多的同源历史并明确记录。
        backend_counts: dict[str, int] = {}
        for keyframe in usable:
            backend = str(keyframe.appearance_backend or "unknown")
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
        backend = max(backend_counts, key=backend_counts.get)
        usable = [
            keyframe
            for keyframe in usable
            if str(keyframe.appearance_backend or "unknown") == backend
        ]
        if len(usable) < SEQUENCE_MIN_KEYFRAMES:
            self.sequence_last_reason = "sequence_backend_history_too_short"
            return False
        samples = self._sequence_training_sample(usable)
        if len(samples) < 8:
            self.sequence_last_reason = "sequence_descriptors_not_ready"
            return False

        values = functional.normalize(
            torch.from_numpy(samples).to(device=device, non_blocking=True), dim=1
        )
        word_count = min(
            SEQUENCE_VISUAL_WORDS,
            max(8, int(values.shape[0])),
        )
        initial = torch.linspace(
            0,
            int(values.shape[0]) - 1,
            steps=word_count,
            device=device,
        ).round().long()
        centers = values[initial].clone()
        with torch.inference_mode():
            for _ in range(SEQUENCE_KMEANS_ITERATIONS):
                sums = torch.zeros_like(centers)
                counts = torch.zeros(
                    word_count, device=device, dtype=torch.float32
                )
                for first in range(0, int(values.shape[0]), 4096):
                    batch = values[first:first + 4096]
                    assignment = torch.argmax(batch @ centers.T, dim=1)
                    sums.index_add_(0, assignment, batch)
                    counts.index_add_(
                        0,
                        assignment,
                        torch.ones(len(batch), device=device),
                    )
                occupied = counts > 0
                centers[occupied] = sums[occupied] / counts[occupied, None]
                centers = functional.normalize(centers, dim=1)

        histograms = torch.stack([
            self._sequence_histogram(
                keyframe.appearance_descriptors,
                centers,
                torch=torch,
                device=device,
            )
            for keyframe in usable
        ])
        document_frequency = torch.count_nonzero(histograms > 0, dim=0).float()
        inverse_frequency = torch.log(
            (float(len(usable)) + 1.0) / (document_frequency + 1.0)
        ) + 1.0
        vectors = functional.normalize(
            torch.log1p(histograms) * inverse_frequency,
            dim=1,
        )
        self._sequence_model = {
            "keyframes": tuple(usable),
            "centers": centers,
            "idf": inverse_frequency,
            "vectors": vectors,
            "device": device,
            "backend": backend,
        }
        self._sequence_model_generation = self._sequence_database_generation
        self._sequence_force_refresh = False
        self.sequence_device = str(device)
        self.sequence_last_reason = "sequence_model_ready"
        return True

    def _ensure_sequence_model(self) -> bool:
        model = self._sequence_model
        if model is None:
            return self._build_sequence_model()
        current_ids = {item.image_id for item in self._place_keyframes}
        model_ids = {item.image_id for item in model["keyframes"]}
        removed = not model_ids.issubset(current_ids)
        stale = (
            self._sequence_database_generation
            - self._sequence_model_generation
            >= SEQUENCE_MODEL_REFRESH_KEYFRAMES
        )
        if self._sequence_force_refresh or removed or stale:
            return self._build_sequence_model()
        return True

    def _score_place_sequence(
        self,
        *,
        min_frame_gap: int = 0,
    ) -> tuple[VisualPlaceCandidate, ...]:
        """对完整查询窗口评分并保留显著、彼此分离的竞争地点。"""
        model = self._sequence_model
        if model is None:
            return ()
        backend = str(model["backend"])
        queries = [
            query for query in self._appearance_queries
            if query.backend == backend
        ][-SEQUENCE_WINDOW_OBSERVATIONS:]
        if len(queries) < SEQUENCE_WINDOW_OBSERVATIONS:
            self.sequence_last_reason = "sequence_query_window_not_ready"
            return ()
        import torch
        import torch.nn.functional as functional

        device = model["device"]
        query_histograms = torch.stack([
            self._sequence_histogram(
                query.descriptors,
                model["centers"],
                torch=torch,
                device=device,
            )
            for query in queries
        ])
        query_vectors = functional.normalize(
            torch.log1p(query_histograms) * model["idf"], dim=1
        )
        similarity = model["vectors"] @ query_vectors.T
        map_count = int(similarity.shape[0])
        window = len(queries)
        map_frames = torch.tensor(
            [item.frame_index for item in model["keyframes"]],
            device=device,
            dtype=torch.float32,
        )
        current_frame = int(queries[-1].frame_index)
        keyframes = model["keyframes"]
        query_frames = torch.tensor(
            [item.frame_index for item in queries],
            device=device,
            dtype=torch.float32,
        )
        query_offsets = query_frames[-1] - query_frames
        endpoints = torch.arange(map_count, device=device, dtype=torch.long)
        # 地点库达到容量后，旧历史会被均匀抽样而近期仍保持稠密。按真实帧号
        # 找邻帧，才能在这种非均匀时间轴上正确比较速度；按数组下标会把抽样
        # 密度变化误判成机器人加速或倒车。
        if map_count > 1:
            gaps = torch.diff(map_frames).clamp_min(1.0)
            left_gap = torch.cat([gaps[:1], gaps])
            right_gap = torch.cat([gaps, gaps[-1:]])
            local_gap = torch.minimum(left_gap, right_gap)
        else:
            local_gap = torch.full_like(map_frames, PLACE_KEYFRAME_MAX_GAP)
        score_parts = []
        metadata: list[tuple[int, float]] = []
        with torch.inference_mode():
            for direction in (-1, 1):
                for speed in SEQUENCE_SPEED_RATIOS:
                    desired_frames = (
                        map_frames[None, :]
                        - float(direction)
                        * query_offsets[:, None]
                        * float(speed)
                    )
                    temporal_error = torch.abs(
                        map_frames[:, None, None]
                        - desired_frames[None, :, :]
                    )
                    nearest_error, indices = torch.min(temporal_error, dim=0)
                    tolerance = torch.maximum(
                        0.60 * local_gap[indices],
                        torch.full_like(nearest_error, PLACE_KEYFRAME_MAX_GAP),
                    )
                    in_time = torch.all(nearest_error <= tolerance, dim=0)
                    # 粗抽样历史允许最近邻，但一个历史帧不能重复冒充整段序列。
                    distinct = 1 + torch.sum(
                        indices[1:] != indices[:-1], dim=0
                    )
                    valid = in_time & (distinct >= min(window, 4))
                    query_indices = torch.arange(
                        window, device=device, dtype=torch.long
                    )[:, None]
                    values = similarity[indices, query_indices]
                    scores = torch.mean(values, dim=0)
                    scores = torch.where(
                        valid, scores, torch.full_like(scores, -1.0)
                    )
                    score_parts.append(scores)
                    metadata.append((int(direction), float(speed)))
        all_scores = torch.stack(score_parts)
        # 回环检索的时间排除必须发生在 top-k 之前。先取 top-k、再由调用者
        # 丢掉近期帧，会让稠密近期视图占满候选，真正的长期闭环永远进不了
        # 后续深度几何门。
        eligible_endpoints = (
            current_frame - map_frames >= float(max(0, int(min_frame_gap)))
        )
        all_scores = torch.where(
            eligible_endpoints[None, :],
            all_scores,
            torch.full_like(all_scores, -1.0),
        )
        valid_scores = all_scores[all_scores >= 0.0]
        if int(valid_scores.numel()) < 2:
            self.sequence_last_reason = "sequence_no_complete_hypothesis"
            return ()
        median = torch.median(valid_scores)
        mad = torch.median(torch.abs(valid_scores - median))
        std = torch.std(valid_scores, unbiased=False)
        spread = torch.maximum(1.4826 * mad, 0.25 * std)
        spread = torch.clamp_min(spread, 1e-6)
        robust_z = (all_scores - median) / spread
        flat_order = torch.argsort(all_scores.flatten(), descending=True)
        scores_cpu = all_scores.detach().cpu().numpy()
        z_cpu = robust_z.detach().cpu().numpy()
        selected: list[tuple[int, int, float, float]] = []
        seen_endpoint_frames: list[int] = []
        for flat in flat_order.detach().cpu().tolist():
            hypothesis_index = int(flat) // map_count
            endpoint = int(flat) % map_count
            score = float(scores_cpu[hypothesis_index, endpoint])
            z_value = float(z_cpu[hypothesis_index, endpoint])
            if score < 0.0 or z_value < SEQUENCE_MIN_ROBUST_Z:
                break
            endpoint_frame = int(keyframes[endpoint].frame_index)
            if any(
                abs(endpoint_frame - old)
                < SEQUENCE_CANDIDATE_SEPARATION_FRAMES
                for old in seen_endpoint_frames
            ):
                continue
            selected.append((hypothesis_index, endpoint, score, z_value))
            seen_endpoint_frames.append(endpoint_frame)
            if len(selected) >= SEQUENCE_MAX_CANDIDATES:
                break
        if not selected:
            self.sequence_last_reason = "sequence_no_prominent_place"
            return ()

        candidates = []
        for rank, (hypothesis_index, endpoint, score, z_value) in enumerate(
            selected, start=1
        ):
            direction, speed = metadata[hypothesis_index]
            keyframe = keyframes[endpoint]
            candidates.append(VisualPlaceCandidate(
                x_m=float(keyframe.pose[0]),
                y_m=float(keyframe.pose[1]),
                yaw_deg=float(keyframe.pose[2]),
                keyframe_id=str(keyframe.image_id),
                keyframe_frame_index=int(keyframe.frame_index),
                candidate_frame_gap=(
                    current_frame - int(keyframe.frame_index)
                ),
                appearance_score=score,
                robust_z=z_value,
                sequence_observations=window,
                direction=direction,
                speed_ratio=speed,
                rank=rank,
                candidate_count=len(selected),
            ))
        self.sequence_last_reason = "sequence_candidates_ready"
        return tuple(candidates)

    def observe_appearance(
        self,
        image_id: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera: dict[str, Any],
        *,
        frame_index: int,
        min_frame_gap: int = 0,
    ) -> tuple[VisualPlaceCandidate, ...]:
        """稀疏提取长期 CUDA 特征，并运行 CUDA 序列地点召回。"""
        frame_no = int(frame_index)
        if self._appearance_queries:
            last = self._appearance_queries[-1]
            if frame_no == last.frame_index:
                return tuple(
                    item for item in self.last_place_candidates
                    if item.candidate_frame_gap >= max(0, int(min_frame_gap))
                )
            if frame_no - last.frame_index < SEQUENCE_QUERY_GAP:
                return tuple(
                    item for item in self.last_place_candidates
                    if item.candidate_frame_gap >= max(0, int(min_frame_gap))
                )
        if not self.feature_device.startswith("cuda"):
            self.sequence_last_reason = "sequence_long_feature_cuda_required"
            self.last_place_candidates = ()
            return ()
        extracted = self._extract(
            rgb,
            depth,
            camera,
            image_features=None,
            cache_key=str(image_id),
        )
        if extracted is None:
            self.sequence_last_reason = "sequence_current_features_missing"
            self.last_place_candidates = ()
            return ()
        pixels, descriptors, robot_points = extracted
        above_ground = robot_points[:, 2] > SEQUENCE_MIN_FEATURE_HEIGHT_M
        pixels = np.ascontiguousarray(pixels[above_ground], dtype=np.float64)
        robot_points = np.ascontiguousarray(
            robot_points[above_ground], dtype=np.float64
        )
        descriptors = np.ascontiguousarray(
            descriptors[above_ground], dtype=np.float32
        )
        if len(descriptors) < KEYFRAME_MIN_RGBD_PAIRS:
            self.sequence_last_reason = "sequence_above_ground_features_missing"
            self.last_place_candidates = ()
            return ()
        self._appearance_queries.append(_AppearanceObservation(
            image_id=str(image_id),
            frame_index=frame_no,
            descriptors=descriptors,
            backend=self.feature_backend,
            pixels=pixels,
            robot_points=robot_points,
        ))
        keep = SEQUENCE_WINDOW_OBSERVATIONS + 2
        if len(self._appearance_queries) > keep:
            del self._appearance_queries[:-keep]
        self.sequence_attempts += 1
        if not self._ensure_sequence_model():
            self.last_place_candidates = ()
            return ()
        self.last_place_candidates = self._score_place_sequence(
            min_frame_gap=min_frame_gap
        )
        self.sequence_candidates += len(self.last_place_candidates)
        return self.last_place_candidates

    @staticmethod
    def _compose_pose(
        keyframe_pose: tuple[float, float, float],
        translation: np.ndarray,
        yaw_rad: float,
    ) -> tuple[float, float, float]:
        kx, ky, kyaw_deg = [float(value) for value in keyframe_pose]
        angle = math.radians(kyaw_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        tx, ty = float(translation[0]), float(translation[1])
        return (
            kx + cos_a * tx - sin_a * ty,
            ky + sin_a * tx + cos_a * ty,
            math.degrees(_wrap_rad(angle + float(yaw_rad))),
        )

    def _cuda_ratio_matches(
        self,
        keyframes: list[_VisualKeyframe],
        current_descriptors: np.ndarray,
    ) -> list[list[_DescriptorMatch]]:
        """在 CUDA 上批量完成历史关键帧到当前帧的二近邻比例测试。"""
        import torch

        device_name = self.feature_device
        if not device_name.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("长期特征后端没有可用 CUDA")
        if len(current_descriptors) < 2:
            return [[] for _ in keyframes]
        lengths = np.asarray(
            [len(item.descriptors) for item in keyframes], dtype=np.int64
        )
        if not len(lengths) or int(np.sum(lengths)) <= 0:
            return [[] for _ in keyframes]
        offsets = np.concatenate([
            np.zeros(1, dtype=np.int64),
            np.cumsum(lengths),
        ])
        historical = np.ascontiguousarray(
            np.concatenate([item.descriptors for item in keyframes], axis=0),
            dtype=np.float32,
        )
        device = torch.device(device_name)
        current = torch.from_numpy(
            np.ascontiguousarray(current_descriptors, dtype=np.float32)
        ).to(device=device, non_blocking=True)
        query = torch.from_numpy(historical).to(device=device, non_blocking=True)
        matches: list[list[_DescriptorMatch]] = [[] for _ in keyframes]
        # 限制瞬时距离矩阵，避免长期运行时给仿真显存制造尖峰。
        chunk_size = 2048
        with torch.inference_mode():
            for first in range(0, int(query.shape[0]), chunk_size):
                last = min(int(query.shape[0]), first + chunk_size)
                distances = torch.cdist(query[first:last], current, p=2.0)
                nearest_distance, nearest_index = torch.topk(
                    distances, k=2, dim=1, largest=False, sorted=True
                )
                accepted = (
                    nearest_distance[:, 0]
                    < self.ratio_limit * nearest_distance[:, 1]
                )
                selected = torch.nonzero(accepted, as_tuple=False).flatten()
                if int(selected.numel()) == 0:
                    continue
                global_query = selected + first
                global_cpu = global_query.detach().cpu().numpy().astype(
                    np.int64, copy=False
                )
                train_cpu = nearest_index[selected, 0].detach().cpu().numpy()
                distance_cpu = nearest_distance[selected, 0].detach().cpu().numpy()
                owners = np.searchsorted(
                    offsets[1:], global_cpu, side="right"
                )
                for global_index, owner, train_index, distance in zip(
                    global_cpu, owners, train_cpu, distance_cpu
                ):
                    matches[int(owner)].append(_DescriptorMatch(
                        queryIdx=int(global_index - offsets[int(owner)]),
                        trainIdx=int(train_index),
                        distance=float(distance),
                    ))
        self.matching_backend = "torch_cdist_cuda"
        self.matching_device = str(device)
        self.matching_fallback_reason = ""
        return matches

    @staticmethod
    def _ratio_pairs_numpy(
        query: np.ndarray,
        train: np.ndarray,
        ratio_limit: float,
    ) -> list[_DescriptorMatch]:
        """小批量 CPU 备用比例测试；结果顺序与 OpenCV BFMatcher 一致。"""
        query = np.asarray(query, dtype=np.float32)
        train = np.asarray(train, dtype=np.float32)
        if len(query) == 0 or len(train) < 2:
            return []
        distances = np.linalg.norm(
            query[:, None, :] - train[None, :, :], axis=2
        )
        order = np.argsort(distances, axis=1, kind="stable")[:, :2]
        rows = np.arange(len(query))
        best = distances[rows, order[:, 0]]
        second = distances[rows, order[:, 1]]
        accepted = best < float(ratio_limit) * second
        return [
            _DescriptorMatch(
                queryIdx=int(index),
                trainIdx=int(order[index, 0]),
                distance=float(best[index]),
            )
            for index in np.flatnonzero(accepted)
        ]

    def _mutual_ratio_matches_cpu(
        self,
        keyframes: list[_PlaceKeyframe],
        current_descriptors: np.ndarray,
    ) -> list[list[_DescriptorMatch]]:
        """地点关键帧的双向比例测试 CPU 备用实现。"""
        import cv2

        result: list[list[_DescriptorMatch]] = [[] for _ in keyframes]
        current = np.ascontiguousarray(current_descriptors, dtype=np.float32)
        if len(current) < 2:
            return result
        for owner, keyframe in enumerate(keyframes):
            historical = np.ascontiguousarray(
                keyframe.appearance_descriptors, dtype=np.float32
            )
            if len(historical) < 2:
                continue
            try:
                forward_pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
                    historical, current, k=2
                )
                reverse_pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
                    current, historical, k=2
                )
                forward = [
                    first
                    for pair in forward_pairs
                    if len(pair) == 2
                    for first, second in [pair]
                    if first.distance < self.ratio_limit * second.distance
                ]
                reverse = {
                    int(first.queryIdx): int(first.trainIdx)
                    for pair in reverse_pairs
                    if len(pair) == 2
                    for first, second in [pair]
                    if first.distance < self.ratio_limit * second.distance
                }
                result[owner] = [
                    _DescriptorMatch(
                        queryIdx=int(item.queryIdx),
                        trainIdx=int(item.trainIdx),
                        distance=float(item.distance),
                    )
                    for item in forward
                    if reverse.get(int(item.trainIdx)) == int(item.queryIdx)
                ]
            except cv2.error:
                forward = self._ratio_pairs_numpy(
                    historical, current, self.ratio_limit
                )
                reverse = self._ratio_pairs_numpy(
                    current, historical, self.ratio_limit
                )
                reverse_by_current = {
                    item.queryIdx: item.trainIdx for item in reverse
                }
                result[owner] = [
                    item for item in forward
                    if reverse_by_current.get(item.trainIdx) == item.queryIdx
                ]
        self.matching_backend = "mutual_ratio_cpu"
        self.matching_device = "cpu"
        return result

    def _cuda_mutual_ratio_matches(
        self,
        keyframes: list[_PlaceKeyframe],
        current_descriptors: np.ndarray,
    ) -> list[list[_DescriptorMatch]]:
        """在 CUDA 上对地点候选执行双向比例测试。

        单向最近邻容易把重复纹理的历史描述子都投到同一个当前点。双向
        一致性把这种 many-to-one 假匹配挡在 RGB-D 几何之前，同时把距离
        矩阵按块计算，避免长期运行造成显存峰值。
        """
        import torch

        result: list[list[_DescriptorMatch]] = [[] for _ in keyframes]
        device_name = self.feature_device
        if not device_name.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("长期特征后端没有可用 CUDA")
        current_np = np.ascontiguousarray(current_descriptors, dtype=np.float32)
        if len(current_np) < 2 or not keyframes:
            return result
        device = torch.device(device_name)
        current = torch.from_numpy(current_np).to(
            device=device, non_blocking=True
        )
        with torch.inference_mode():
            for owner, keyframe in enumerate(keyframes):
                historical_np = np.ascontiguousarray(
                    keyframe.appearance_descriptors, dtype=np.float32
                )
                if len(historical_np) < 2:
                    continue
                historical = torch.from_numpy(historical_np).to(
                    device=device, non_blocking=True
                )
                # 地点候选已经由序列层缩到很小；每个候选独立计算反向
                # 最近邻，才能让不同历史视点各自贡献一份几何证据。
                distances = torch.cdist(historical, current, p=2.0)
                forward_distance, forward_index = torch.topk(
                    distances, k=2, dim=1, largest=False, sorted=True
                )
                reverse_distance, reverse_index = torch.topk(
                    distances.transpose(0, 1),
                    k=2,
                    dim=1,
                    largest=False,
                    sorted=True,
                )
                forward_ok = (
                    forward_distance[:, 0]
                    < self.ratio_limit * forward_distance[:, 1]
                )
                reverse_ok = (
                    reverse_distance[:, 0]
                    < self.ratio_limit * reverse_distance[:, 1]
                )
                historical_indices = torch.arange(
                    int(historical.shape[0]), device=device
                )
                current_indices = forward_index[:, 0]
                mutual = (
                    forward_ok
                    & reverse_ok[current_indices]
                    & (reverse_index[current_indices, 0] == historical_indices)
                )
                selected = torch.nonzero(mutual, as_tuple=False).flatten()
                q_cpu = selected.detach().cpu().numpy().astype(np.int64)
                t_cpu = (
                    current_indices[selected]
                    .detach().cpu().numpy().astype(np.int64)
                )
                d_cpu = (
                    forward_distance[selected, 0]
                    .detach().cpu().numpy().astype(np.float32)
                )
                result[owner] = [
                    _DescriptorMatch(
                        queryIdx=int(query_index),
                        trainIdx=int(train_index),
                        distance=float(distance),
                    )
                    for query_index, train_index, distance in zip(
                        q_cpu, t_cpu, d_cpu
                    )
                ]
        self.matching_backend = "torch_cdist_cuda_mutual"
        self.matching_device = str(device)
        self.matching_fallback_reason = ""
        return result

    def _match_keyframe(
        self,
        keyframe: _VisualKeyframe,
        current_pixels: np.ndarray,
        current_descriptors: np.ndarray,
        current_points: np.ndarray,
        *,
        precomputed_matches: Optional[list[_DescriptorMatch]] = None,
        camera: Optional[dict[str, Any]] = None,
        image_shape: Optional[tuple[int, int]] = None,
        require_cuda_ransac: bool = False,
    ) -> Optional[dict[str, Any]]:
        import cv2

        if keyframe.descriptors is None or len(keyframe.descriptors) < KEYFRAME_MIN_RGBD_PAIRS:
            return None
        if precomputed_matches is None:
            if require_cuda_ransac:
                return None
            pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
                keyframe.descriptors,
                current_descriptors,
                k=2,
            )
            matches = [
                first
                for pair in pairs
                if len(pair) == 2
                for first, second in [pair]
                if first.distance < self.ratio_limit * second.distance
            ]
        else:
            matches = list(precomputed_matches)
        if len(matches) < KEYFRAME_MIN_RATIO_MATCHES:
            return None
        # 同一个当前描述子只保留最相似的一次，避免重复纹理把 RANSAC 的票
        # 人为放大。
        unique: dict[int, Any] = {}
        for match in matches:
            old = unique.get(int(match.trainIdx))
            if old is None or match.distance < old.distance:
                unique[int(match.trainIdx)] = match
        matches = list(unique.values())
        if len(matches) < KEYFRAME_MIN_RATIO_MATCHES:
            return None
        key_pixels = np.asarray(
            [keyframe.pixels[match.queryIdx] for match in matches],
            dtype=np.float64,
        )
        now_pixels = np.asarray(
            [current_pixels[match.trainIdx] for match in matches],
            dtype=np.float64,
        )
        # 基础矩阵只是去掉明显的跨物体错误；平面/纯旋转退化时保留原始
        # RGB-D 对应，真正的几何门槛由后面的 SE(2) RANSAC 负责。
        geometric = np.ones(len(matches), dtype=bool)
        if len(matches) >= 8:
            try:
                _fundamental, mask = cv2.findFundamentalMat(
                    key_pixels,
                    now_pixels,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=2.0,
                    confidence=0.995,
                )
                if mask is not None:
                    candidate = mask.reshape(-1).astype(bool)
                    if int(np.count_nonzero(candidate)) >= max(8, int(0.45 * len(matches))):
                        geometric = candidate
            except cv2.error:
                pass
        indices = np.flatnonzero(geometric)
        source = current_points[[matches[index].trainIdx for index in indices], :2]
        target = keyframe.robot_points[[matches[index].queryIdx for index in indices], :2]
        matched_now_pixels = now_pixels[indices]
        source_z = current_points[[matches[index].trainIdx for index in indices], 2]
        target_z = keyframe.robot_points[[matches[index].queryIdx for index in indices], 2]
        valid = np.abs(source_z - target_z) <= max(0.25, MAX_HEIGHT_DELTA_M)
        source, target = source[valid], target[valid]
        matched_now_pixels = matched_now_pixels[valid]
        source_z, target_z = source_z[valid], target_z[valid]
        if len(source) < KEYFRAME_MIN_RGBD_PAIRS:
            return None
        try:
            rotation, translation, inliers = _ransac_se2(
                source,
                target,
                require_cuda=bool(require_cuda_ransac),
                cuda_device=(
                    self.feature_device if require_cuda_ransac else None
                ),
            )
        except (ValueError, FloatingPointError):
            return None
        except Exception:
            if require_cuda_ransac:
                return None
            raise
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < KEYFRAME_MIN_INLIERS:
            return None
        residual = np.linalg.norm(
            (rotation @ source[inliers].T).T + translation - target[inliers],
            axis=1,
        )
        rmse = float(np.sqrt(np.mean(residual ** 2))) if len(residual) else math.inf
        ratio = float(inlier_count / max(1, len(source)))
        inlier_points = source[inliers]
        span = np.ptp(inlier_points, axis=0) if len(inlier_points) else np.zeros(2)
        height_span = float(np.ptp(source_z[inliers])) if inlier_count else 0.0
        # 重复门板/地板经常能给出很多“完美”匹配，但都挤在一个小平面
        # 上。没有足够的横向和高度覆盖时，不把它当成长时地点证据。
        above_floor = int(np.count_nonzero(source_z[inliers] > 0.20))
        pixel_span = (
            np.ptp(matched_now_pixels[inliers], axis=0)
            if inlier_count
            else np.zeros(2, dtype=np.float64)
        )
        # 先要求有足够长的实体结构和离地支持，再允许“窄墙”通过。窄墙
        # 只能靠图像覆盖证明它不是一片地板；不能仅凭高匹配数放行。
        if image_shape is None:
            image_height, image_width = 720, 720
        else:
            image_height, image_width = (
                int(image_shape[0]), int(image_shape[1])
            )
        if current_pixels.size:
            image_width = max(
                image_width, int(np.ceil(np.max(current_pixels[:, 0]) + 1.0))
            )
            image_height = max(
                image_height, int(np.ceil(np.max(current_pixels[:, 1]) + 1.0))
            )
        image_covered = (
            float(pixel_span[0]) >= KEYFRAME_MIN_PIXEL_COVERAGE_X * image_width
            and float(pixel_span[1]) >= KEYFRAME_MIN_PIXEL_COVERAGE_Y * image_height
        )
        covered = (
            max(float(span[0]), float(span[1])) >= KEYFRAME_MIN_SPAN_M
            and height_span >= KEYFRAME_MIN_HEIGHT_SPAN_M
            and (
                min(float(span[0]), float(span[1])) >= KEYFRAME_MIN_CROSS_SPAN_M
                or (above_floor >= KEYFRAME_MIN_ABOVE_FLOOR_INLIERS and image_covered)
            )
        )
        if (
            ratio < KEYFRAME_MIN_INLIER_RATIO
            or rmse > KEYFRAME_MAX_RMSE_M
            or not covered
        ):
            return None
        fx = _camera_intrinsics(
            camera or {}, int(image_width), int(image_height)
        )[0]
        observability = _se2_observability(
            source[inliers],
            target[inliers],
            rotation,
            translation,
            matched_now_pixels[inliers],
            image_width=int(image_width),
            image_height=int(image_height),
            fx=float(fx),
        )
        return {
            "pose": self._compose_pose(
                keyframe.pose,
                translation,
                math.atan2(rotation[1, 0], rotation[0, 0]),
            ),
            "relative_pose": (
                float(translation[0]),
                float(translation[1]),
                math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])),
            ),
            "keyframe": keyframe,
            "ratio_matches": len(matches),
            "rgbd_pairs": len(source),
            "inliers": inlier_count,
            "inlier_ratio": ratio,
            "rmse_m": rmse,
            "above_floor_inliers": above_floor,
            "pixel_span": (float(pixel_span[0]), float(pixel_span[1])),
            **observability,
            "weight": float(inlier_count * max(0.1, ratio) / max(0.01, rmse + 0.01)),
        }

    def _match_place_keyframe(
        self,
        keyframe: _PlaceKeyframe,
        current_pixels: np.ndarray,
        current_descriptors: np.ndarray,
        current_points: np.ndarray,
        *,
        precomputed_matches: Optional[list[_DescriptorMatch]] = None,
        report: Optional[dict[str, Any]] = None,
        camera: Optional[dict[str, Any]] = None,
        image_shape: Optional[tuple[int, int]] = None,
        require_cuda_ransac: bool = False,
    ) -> Optional[dict[str, Any]]:
        """把外观召回的地点重新用 RGB-D 点做度量匹配。

        这里不使用地点条目的绝对位姿来“修正”当前帧；绝对位姿只作为
        ``_compose_pose`` 的历史锚点。任何不满足空间覆盖、残差或比例门的
        候选都丢弃，因此重复纹理不会直接变成回环。
        """
        metrics = report if report is not None else {}
        metrics.update({
            "keyframe_id": str(keyframe.image_id),
            "keyframe_frame_index": int(keyframe.frame_index),
            "accepted": False,
            "reason": "not_evaluated",
        })
        if (
            len(keyframe.appearance_descriptors) < PLACE_GEOMETRY_MIN_MATCHES
            or len(keyframe.robot_points) < PLACE_GEOMETRY_MIN_MATCHES
            or len(keyframe.pixels) != len(keyframe.robot_points)
            or len(keyframe.appearance_descriptors) != len(keyframe.robot_points)
        ):
            metrics["reason"] = "keyframe_rgbd_features_missing"
            return None
        if precomputed_matches is None:
            matches = self._mutual_ratio_matches_cpu(
                [keyframe], current_descriptors
            )[0]
        else:
            matches = list(precomputed_matches)
        metrics["mutual_ratio_matches"] = int(len(matches))
        if len(matches) < PLACE_GEOMETRY_MIN_MATCHES:
            metrics["reason"] = "few_mutual_ratio_matches"
            return None
        # 双向匹配已经保证当前索引唯一，历史索引仍再去重一次，以免
        # 非标准/旧缓存条目把一个历史点重复投票。
        unique: dict[int, _DescriptorMatch] = {}
        for match in matches:
            old = unique.get(int(match.queryIdx))
            if old is None or match.distance < old.distance:
                unique[int(match.queryIdx)] = match
        matches = list(unique.values())
        metrics["unique_matches"] = int(len(matches))
        if len(matches) < PLACE_GEOMETRY_MIN_MATCHES:
            metrics["reason"] = "few_unique_matches"
            return None
        try:
            source_indices = [int(match.trainIdx) for match in matches]
            target_indices = [int(match.queryIdx) for match in matches]
            source = np.asarray(current_points[source_indices, :2], dtype=np.float64)
            target = np.asarray(
                keyframe.robot_points[target_indices, :2], dtype=np.float64
            )
            source_z = np.asarray(current_points[source_indices, 2], dtype=np.float64)
            target_z = np.asarray(
                keyframe.robot_points[target_indices, 2], dtype=np.float64
            )
            matched_pixels = np.asarray(
                current_pixels[source_indices], dtype=np.float64
            )
        except (IndexError, TypeError, ValueError):
            metrics["reason"] = "invalid_match_indices"
            return None
        finite = (
            np.isfinite(source).all(axis=1)
            & np.isfinite(target).all(axis=1)
            & np.isfinite(source_z)
            & np.isfinite(target_z)
            & np.isfinite(matched_pixels).all(axis=1)
        )
        height_ok = np.abs(source_z - target_z) <= max(0.30, MAX_HEIGHT_DELTA_M)
        valid = finite & height_ok
        source, target = source[valid], target[valid]
        source_z, target_z = source_z[valid], target_z[valid]
        matched_pixels = matched_pixels[valid]
        metrics["rgbd_pairs"] = int(len(source))
        if len(source) < PLACE_GEOMETRY_MIN_MATCHES:
            metrics["reason"] = "few_rgbd_pairs"
            return None
        try:
            rotation, translation, inliers = _ransac_se2(
                source,
                target,
                require_cuda=bool(require_cuda_ransac),
                cuda_device=(
                    self.feature_device if require_cuda_ransac else None
                ),
            )
        except (ValueError, FloatingPointError):
            metrics["reason"] = "se2_ransac_failed"
            return None
        except Exception:
            # A strict place query treats CUDA/backend failure as UNKNOWN.
            # The fallback reason remains available in the localizer stats.
            metrics["reason"] = "cuda_se2_ransac_failed"
            return None
        inlier_count = int(np.count_nonzero(inliers))
        metrics["inliers"] = inlier_count
        if inlier_count < PLACE_GEOMETRY_MIN_INLIERS:
            metrics["reason"] = "few_se2_inliers"
            return None
        residual = np.linalg.norm(
            (rotation @ source[inliers].T).T + translation - target[inliers],
            axis=1,
        )
        rmse = float(np.sqrt(np.mean(residual ** 2))) if len(residual) else math.inf
        ratio = float(inlier_count / max(1, len(source)))
        inlier_points = source[inliers]
        span = (
            np.ptp(inlier_points, axis=0)
            if len(inlier_points)
            else np.zeros(2, dtype=np.float64)
        )
        pixel_span = (
            np.ptp(matched_pixels[inliers], axis=0)
            if inlier_count
            else np.zeros(2, dtype=np.float64)
        )
        if image_shape is None:
            image_height, image_width = 720, 720
        else:
            image_height, image_width = (
                int(image_shape[0]), int(image_shape[1])
            )
        if len(current_pixels):
            image_width = max(
                image_width, int(np.ceil(np.max(current_pixels[:, 0]) + 1.0))
            )
            image_height = max(
                image_height, int(np.ceil(np.max(current_pixels[:, 1]) + 1.0))
            )
        image_covered = (
            float(pixel_span[0]) >= KEYFRAME_MIN_PIXEL_COVERAGE_X * image_width
            and float(pixel_span[1]) >= KEYFRAME_MIN_PIXEL_COVERAGE_Y * image_height
        )
        covered = (
            max(float(span[0]), float(span[1])) >= 0.60
            and (
                min(float(span[0]), float(span[1])) >= 0.12
                or (inlier_count >= 8 and image_covered)
            )
        )
        metrics.update({
            "inlier_ratio": ratio,
            "rmse_m": rmse,
            "span_m": [float(span[0]), float(span[1])],
            "pixel_span": [float(pixel_span[0]), float(pixel_span[1])],
            "covered": bool(covered),
        })
        if (
            ratio < PLACE_GEOMETRY_MIN_INLIER_RATIO
            or not math.isfinite(rmse)
            or rmse > PLACE_GEOMETRY_MAX_RMSE_M
            or not covered
        ):
            if ratio < PLACE_GEOMETRY_MIN_INLIER_RATIO:
                metrics["reason"] = "low_inlier_ratio"
            elif not math.isfinite(rmse) or rmse > PLACE_GEOMETRY_MAX_RMSE_M:
                metrics["reason"] = "high_se2_rmse"
            else:
                metrics["reason"] = "insufficient_spatial_coverage"
            return None
        yaw_rad = math.atan2(rotation[1, 0], rotation[0, 0])
        fx = _camera_intrinsics(
            camera or {}, int(image_width), int(image_height)
        )[0]
        observability = _se2_observability(
            source[inliers],
            target[inliers],
            rotation,
            translation,
            matched_pixels[inliers],
            image_width=int(image_width),
            image_height=int(image_height),
            fx=float(fx),
        )
        metrics.update({
            "accepted": True,
            "reason": "accepted",
            "relative_pose": [
                float(translation[0]),
                float(translation[1]),
                math.degrees(yaw_rad),
            ],
            **observability,
        })
        return {
            "pose": self._compose_pose(keyframe.pose, translation, yaw_rad),
            "relative_pose": (
                float(translation[0]),
                float(translation[1]),
                math.degrees(yaw_rad),
            ),
            "keyframe": keyframe,
            "ratio_matches": len(matches),
            "rgbd_pairs": len(source),
            "inliers": inlier_count,
            "inlier_ratio": ratio,
            "rmse_m": rmse,
            "above_floor_inliers": int(
                np.count_nonzero(source_z[inliers] > SEQUENCE_MIN_FEATURE_HEIGHT_M)
            ),
            "pixel_span": (float(pixel_span[0]), float(pixel_span[1])),
            "evidence_type": "place_rgbd_geometry",
            **observability,
            "weight": float(
                inlier_count * max(0.1, ratio) / max(0.01, rmse + 0.01)
            ),
        }

    @staticmethod
    def _yaw_delta(a: float, b: float) -> float:
        return math.degrees(_wrap_rad(math.radians(float(a) - float(b))))

    def _clusters(self, candidates: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        clusters: list[list[dict[str, Any]]] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (
                -float(item["weight"]),
                int(item["keyframe"].frame_index),
                str(item["keyframe"].image_id),
            ),
        ):
            pose = candidate["pose"]
            assigned = None
            for cluster in clusters:
                if all(
                    math.hypot(
                        float(pose[0]) - float(item["pose"][0]),
                        float(pose[1]) - float(item["pose"][1]),
                    ) <= KEYFRAME_CLUSTER_XY_M
                    and abs(self._yaw_delta(pose[2], item["pose"][2]))
                    <= KEYFRAME_CLUSTER_YAW_DEG
                    for item in cluster
                ):
                    assigned = cluster
                    break
            if assigned is None:
                clusters.append([candidate])
            else:
                assigned.append(candidate)
        return clusters

    @staticmethod
    def _independent_keyframe_candidates(
        cluster: Iterable[dict[str, Any]],
        *,
        min_frame_gap: int = KEYFRAME_INDEPENDENT_GAP,
        min_translation_m: float = KEYFRAME_INDEPENDENT_TRANSLATION_M,
    ) -> Optional[list[dict[str, Any]]]:
        """Return an exact maximum set of pairwise independent observations."""
        ordered = sorted(
            cluster,
            key=lambda item: (
                -float(item["weight"]),
                int(item["keyframe"].frame_index),
                str(item["keyframe"].image_id),
                str(item.get("evidence_type", "direct_rgbd_geometry")),
            ),
        )
        if not ordered:
            return []
        adjacency = [0] * len(ordered)
        for first_index, first in enumerate(ordered):
            first_keyframe = first["keyframe"]
            first_frame = int(first_keyframe.frame_index)
            first_x = float(first_keyframe.pose[0])
            first_y = float(first_keyframe.pose[1])
            for second_index in range(first_index + 1, len(ordered)):
                second_keyframe = ordered[second_index]["keyframe"]
                independent = (
                    abs(first_frame - int(second_keyframe.frame_index))
                    >= int(min_frame_gap)
                    and math.hypot(
                        first_x - float(second_keyframe.pose[0]),
                        first_y - float(second_keyframe.pose[1]),
                    ) >= float(min_translation_m)
                )
                if independent:
                    adjacency[first_index] |= 1 << second_index
                    adjacency[second_index] |= 1 << first_index

        best: tuple[int, ...] = ()
        visited_states = 0
        exhausted = False

        def search(selected: tuple[int, ...], available: int) -> None:
            nonlocal best, exhausted, visited_states
            if exhausted:
                return
            visited_states += 1
            if visited_states > KEYFRAME_INDEPENDENT_SEARCH_MAX_STATES:
                exhausted = True
                return
            if len(selected) > len(best):
                best = selected
            if len(selected) + available.bit_count() <= len(best):
                return
            while available and not exhausted:
                if len(selected) + available.bit_count() <= len(best):
                    return
                bit = available & -available
                available ^= bit
                index = bit.bit_length() - 1
                search(selected + (index,), available & adjacency[index])

        search((), (1 << len(ordered)) - 1)
        if exhausted:
            return None
        return [ordered[index] for index in best]

    @staticmethod
    def _keyframe_translation_span(candidates: Iterable[dict[str, Any]]) -> float:
        poses = [item["keyframe"].pose for item in candidates]
        return max(
            (
                math.hypot(
                    float(first[0]) - float(second[0]),
                    float(first[1]) - float(second[1]),
                )
                for index, first in enumerate(poses)
                for second in poses[index + 1:]
            ),
            default=0.0,
        )

    @staticmethod
    def _is_observable_cluster_witness(candidate: dict[str, Any]) -> bool:
        """Return whether one registration is a complete planar pose witness."""
        try:
            pose = tuple(float(value) for value in candidate["pose"])
            relative_pose = tuple(
                float(value) for value in candidate["relative_pose"]
            )
            observability = (
                float(candidate["geometry_minor_span_m"]),
                float(candidate["pixel_coverage_x"]),
                float(candidate["pixel_coverage_y"]),
                float(candidate["translation_std_m"]),
                float(candidate["yaw_std_deg"]),
                float(candidate["normal_matrix_condition"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if (
            len(pose) != 3
            or len(relative_pose) != 3
            or not all(math.isfinite(value) for value in (*pose, *relative_pose))
            or not all(math.isfinite(value) for value in observability)
        ):
            return False
        minor_span, pixel_x, pixel_y, translation_std, yaw_std, _ = (
            observability
        )
        return (
            minor_span >= KEYFRAME_MIN_CROSS_SPAN_M
            and pixel_x >= KEYFRAME_MIN_PIXEL_COVERAGE_X
            and pixel_y >= KEYFRAME_MIN_PIXEL_COVERAGE_Y
            and translation_std <= KEYFRAME_MAX_TRANSLATION_STD_M
            and yaw_std <= KEYFRAME_MAX_YAW_STD_DEG
        )

    @staticmethod
    def _select_cluster_witness(
        candidates: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        """Choose one real registration with a deterministic quality order."""
        return min(
            candidates,
            key=lambda item: (
                -int(
                    str(item.get("evidence_type", "direct_rgbd_geometry"))
                    != "place_rgbd_geometry"
                ),
                -float(item["weight"]),
                -int(item["inliers"]),
                -float(item["inlier_ratio"]),
                float(item["rmse_m"]),
                int(item["keyframe"].frame_index),
                str(item["keyframe"].image_id),
            ),
        )

    @classmethod
    def _is_single_strong_cluster_witness(
        cls,
        candidate: dict[str, Any],
        *,
        min_inliers: int,
        min_ratio: float,
        min_above_floor: int,
    ) -> bool:
        return bool(
            cls._is_observable_cluster_witness(candidate)
            and str(candidate.get("evidence_type", "direct_rgbd_geometry"))
            != "place_rgbd_geometry"
            and int(candidate["inliers"]) >= int(min_inliers)
            and float(candidate["inlier_ratio"]) >= float(min_ratio)
            and int(candidate.get("above_floor_inliers", 0))
            >= int(min_above_floor)
        )

    def localize(
        self,
        image_id: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        camera: dict[str, Any],
        *,
        predicted_pose: Optional[tuple[float, float, float]] = None,
        frame_index: Optional[int] = None,
        min_frame_gap: int = KEYFRAME_MIN_FRAME_GAP,
        min_independent: int = KEYFRAME_MIN_INDEPENDENT,
        allow_single_strong: bool = True,
        single_min_inliers: int = KEYFRAME_SINGLE_MIN_INLIERS,
        single_min_ratio: float = KEYFRAME_SINGLE_MIN_RATIO,
        single_min_above_floor: int = KEYFRAME_SINGLE_MIN_ABOVE_FLOOR,
        max_pose_jump_m: float = KEYFRAME_MAX_POSE_JUMP_M,
        max_pose_yaw_jump_deg: float = KEYFRAME_MAX_POSE_YAW_JUMP_DEG,
        image_features: Optional[ImageFeatures] = None,
        place_candidates: Optional[Iterable[Any]] = None,
        require_cuda_matching: bool = False,
    ) -> Optional[VisualLocalization]:
        """把当前帧定位到关键帧地图；失败时返回 ``None``。

        ``place_candidates`` 只是一组外观召回的历史 ID。它们必须重新经过
        双向描述子匹配和 RGB-D 几何，绝不把召回条目的绝对位姿当作当前位姿。
        """
        self.attempts += 1
        extracted = self._extract(
            rgb,
            depth,
            camera,
            image_features=image_features,
            cache_key=str(image_id),
        )
        if extracted is None:
            self.rejected += 1
            self.last_reason = "current_no_rgbd_features"
            self.last_result = None
            return None
        pixels, descriptors, robot_points = extracted
        depth_array = np.asarray(depth).squeeze()
        image_shape = (
            (int(depth_array.shape[0]), int(depth_array.shape[1]))
            if depth_array.ndim == 2
            else None
        )
        eligible_keyframes = [
            keyframe
            for keyframe in self._keyframes
            if not (
                frame_index is not None
                and int(frame_index) - keyframe.frame_index < int(min_frame_gap)
            )
        ]
        candidate_ids: set[str] = set()
        if place_candidates is not None:
            for item in place_candidates:
                if isinstance(item, str):
                    key = item
                else:
                    key = getattr(item, "keyframe_id", "")
                key = str(key or "").strip()
                if key:
                    candidate_ids.add(key)
        eligible_place_keyframes = [
            keyframe
            for keyframe in self._place_keyframes
            if str(keyframe.image_id) in candidate_ids
            and not (
                frame_index is not None
                and int(frame_index) - keyframe.frame_index < int(min_frame_gap)
            )
            and str(keyframe.appearance_backend or self.feature_backend)
            == str(self.feature_backend)
        ]
        if (
            require_cuda_matching
            and (eligible_keyframes or eligible_place_keyframes)
            and not self.feature_device.startswith("cuda")
        ):
            self.rejected += 1
            self.last_reason = (
                "cuda_place_matching_required"
                if eligible_place_keyframes
                else "cuda_keyframe_matching_required"
            )
            self.last_result = None
            return None
        precomputed: Optional[list[list[_DescriptorMatch]]] = None
        if self.feature_device.startswith("cuda") and eligible_keyframes:
            try:
                precomputed = self._cuda_ratio_matches(
                    eligible_keyframes, descriptors
                )
            except Exception as exc:
                self.matching_backend = "torch_cdist_cuda_failed"
                self.matching_device = str(self.feature_device)
                self.matching_fallback_reason = f"{type(exc).__name__}: {exc}"
                if require_cuda_matching:
                    self.rejected += 1
                    self.last_reason = "cuda_keyframe_matching_failed"
                    self.last_result = None
                    return None
        else:
            self.matching_backend = "opencv_bf_cpu"
            self.matching_device = "cpu"
            self.matching_fallback_reason = ""
        candidates = []
        for index, keyframe in enumerate(eligible_keyframes):
            candidate = self._match_keyframe(
                keyframe,
                pixels,
                descriptors,
                robot_points,
                precomputed_matches=(
                    None if precomputed is None else precomputed[index]
                ),
                camera=camera,
                image_shape=image_shape,
                require_cuda_ransac=bool(require_cuda_matching),
            )
            if candidate is not None:
                candidates.append(candidate)
        place_candidates_geometry: list[dict[str, Any]] = []
        place_geometry_rows: list[dict[str, Any]] = []
        if eligible_place_keyframes:
            appearance_valid = robot_points[:, 2] > SEQUENCE_MIN_FEATURE_HEIGHT_M
            place_pixels = np.ascontiguousarray(
                pixels[appearance_valid], dtype=np.float64
            )
            place_descriptors = np.ascontiguousarray(
                descriptors[appearance_valid], dtype=np.float32
            )
            place_points = np.ascontiguousarray(
                robot_points[appearance_valid], dtype=np.float64
            )
            if len(place_descriptors) >= PLACE_GEOMETRY_MIN_MATCHES:
                place_precomputed: Optional[list[list[_DescriptorMatch]]] = None
                if self.feature_device.startswith("cuda"):
                    try:
                        place_precomputed = self._cuda_mutual_ratio_matches(
                            eligible_place_keyframes, place_descriptors
                        )
                    except Exception as exc:
                        self.matching_fallback_reason = (
                            f"place_cuda_{type(exc).__name__}: {exc}"
                        )
                        if require_cuda_matching:
                            self.rejected += 1
                            self.last_reason = "cuda_place_matching_failed"
                            self.last_result = None
                            return None
                if place_precomputed is None:
                    if require_cuda_matching:
                        self.rejected += 1
                        self.last_reason = "cuda_place_matching_unavailable"
                        self.last_result = None
                        return None
                    place_precomputed = self._mutual_ratio_matches_cpu(
                        eligible_place_keyframes, place_descriptors
                    )
                for index, keyframe in enumerate(eligible_place_keyframes):
                    self.place_geometry_attempts += 1
                    geometry_report: dict[str, Any] = {}
                    candidate = self._match_place_keyframe(
                        keyframe,
                        place_pixels,
                        place_descriptors,
                        place_points,
                        precomputed_matches=place_precomputed[index],
                        report=geometry_report,
                        camera=camera,
                        image_shape=image_shape,
                        require_cuda_ransac=bool(require_cuda_matching),
                    )
                    place_geometry_rows.append(geometry_report)
                    reason = str(geometry_report.get("reason") or "unknown")
                    self.place_geometry_reason_counts[reason] = (
                        int(self.place_geometry_reason_counts.get(reason, 0)) + 1
                    )
                    if candidate is not None:
                        place_candidates_geometry.append(candidate)
                        self.place_geometry_accepted += 1
                        self.place_geometry_last_reason = "accepted"
                    else:
                        self.place_geometry_rejected += 1
                        self.place_geometry_last_reason = "geometry_gate_rejected"
            else:
                self.last_reason = "place_current_above_ground_features_missing"
                self.place_geometry_last_reason = self.last_reason
        self.last_place_geometry_rows = tuple(place_geometry_rows)
        candidates.extend(place_candidates_geometry)
        if not candidates:
            self.rejected += 1
            self.last_reason = "no_keyframe_geometry"
            self.last_result = None
            return None
        clusters = self._clusters(candidates)
        # 以不同时间、不同相机位置关键帧的共同解为首要证据；单个关键帧
        # 只有在调用方明确允许时才可走强证据捷径。
        scored_clusters = []
        for cluster in clusters:
            independent_items = self._independent_keyframe_candidates(cluster)
            if independent_items is None:
                self.rejected += 1
                self.last_reason = "independent_evidence_search_budget_exceeded"
                self.last_result = None
                return None
            independent = len(independent_items)
            translation_span = self._keyframe_translation_span(independent_items)
            place_items = [
                item for item in cluster
                if str(item.get("evidence_type", "direct_rgbd_geometry"))
                == "place_rgbd_geometry"
            ]
            place_independent_items = self._independent_keyframe_candidates(
                place_items,
                min_frame_gap=PLACE_GEOMETRY_INDEPENDENT_GAP,
                min_translation_m=PLACE_GEOMETRY_INDEPENDENT_TRANSLATION_M,
            )
            if place_independent_items is None:
                self.rejected += 1
                self.last_reason = "independent_evidence_search_budget_exceeded"
                self.last_result = None
                return None
            place_independent = len(place_independent_items)
            place_translation_span = self._keyframe_translation_span(
                place_independent_items
            )
            direct_count = len(cluster) - len(place_items)
            weight = sum(float(item["weight"]) for item in cluster)
            strong = max(int(item["inliers"]) for item in cluster)
            ratio = max(float(item["inlier_ratio"]) for item in cluster)
            above = max(int(item.get("above_floor_inliers", 0)) for item in cluster)
            scored_clusters.append({
                "independent": independent,
                "translation_span": translation_span,
                "weight": weight,
                "strongest": strong,
                "best_ratio": ratio,
                "strongest_above": above,
                "place_count": len(place_items),
                "place_independent": place_independent,
                "place_translation_span": place_translation_span,
                "direct_count": direct_count,
                "cluster": cluster,
            })
        valid_clusters = [
            item for item in scored_clusters
            if not item["place_count"]
            or item["direct_count"]
            or int(item["place_count"]) >= 1
        ]
        if not valid_clusters:
            self.rejected += 1
            self.last_reason = "no_independent_place_geometry"
            self.last_result = None
            return None
        ranked_clusters = sorted(
            valid_clusters,
            key=lambda item: (
                int(item["independent"]),
                float(item["weight"]),
                int(item["strongest"]),
            ),
            reverse=True,
        )
        best_cluster = ranked_clusters[0]
        best_support = max(1e-9, float(best_cluster["weight"]))
        for competitor in ranked_clusters[1:]:
            competitor_has_single_strong_witness = bool(
                allow_single_strong
                and int(competitor["independent"]) >= 1
                and any(
                    self._is_single_strong_cluster_witness(
                        item,
                        min_inliers=single_min_inliers,
                        min_ratio=single_min_ratio,
                        min_above_floor=single_min_above_floor,
                    )
                    for item in competitor["cluster"]
                )
            )
            if (
                (
                    int(competitor["independent"])
                    >= max(
                        1,
                        min(
                            int(min_independent),
                            int(best_cluster["independent"]),
                        ),
                    )
                    or competitor_has_single_strong_witness
                )
                and float(competitor["weight"]) / best_support
                >= KEYFRAME_AMBIGUOUS_SECOND_WEIGHT_RATIO
            ):
                self.rejected += 1
                self.last_reason = "ambiguous_metric_pose_clusters"
                self.last_result = None
                return None
        independent = int(best_cluster["independent"])
        translation_span = float(best_cluster["translation_span"])
        cluster = best_cluster["cluster"]
        place_count = int(best_cluster["place_count"])
        place_independent = int(best_cluster["place_independent"])
        required_independent = max(1, int(min_independent))
        observable_witnesses = [
            item for item in cluster
            if self._is_observable_cluster_witness(item)
        ]
        single_strong_witnesses = [
            item for item in observable_witnesses
            if self._is_single_strong_cluster_witness(
                item,
                min_inliers=single_min_inliers,
                min_ratio=single_min_ratio,
                min_above_floor=single_min_above_floor,
            )
        ]
        single_is_strong = bool(
            allow_single_strong
            and independent >= 1
            and single_strong_witnesses
        )
        if place_count and not best_cluster["direct_count"]:
            # 建图调用方通常要求两份历史关键帧；冻结后的定位可先租借
            # 单份已通过 RGB-D 的地点证据，再由跨时间窗口完成第二道门。
            required_independent = max(required_independent, 1)
        if independent < required_independent and not single_is_strong:
            self.rejected += 1
            self.last_reason = "no_independent_consensus"
            self.last_result = None
            return None
        if not observable_witnesses:
            self.rejected += 1
            self.last_reason = "no_observable_cluster_witness"
            self.last_result = None
            return None
        witness_pool = (
            single_strong_witnesses
            if independent < required_independent
            else observable_witnesses
        )
        representative = self._select_cluster_witness(witness_pool)
        x, y, yaw = (float(value) for value in representative["pose"])
        # 预测位姿只作极端跳变保护，不能把合法回环限制在里程附近。
        if predicted_pose is not None:
            jump = math.hypot(x - float(predicted_pose[0]), y - float(predicted_pose[1]))
            yaw_jump = abs(self._yaw_delta(yaw, float(predicted_pose[2])))
            if jump > float(max_pose_jump_m):
                self.rejected += 1
                self.last_reason = "implausible_pose_jump"
                self.last_result = None
                return None
            if yaw_jump > float(max_pose_yaw_jump_deg):
                self.rejected += 1
                self.last_reason = "implausible_yaw_jump"
                self.last_result = None
                return None
        candidate_frames = [item["keyframe"].frame_index for item in cluster]
        oldest_frame = min(candidate_frames)
        result = VisualLocalization(
            x_m=x,
            y_m=y,
            yaw_deg=math.degrees(_wrap_rad(math.radians(yaw))),
            keyframe_id=representative["keyframe"].image_id,
            ratio_matches=int(representative["ratio_matches"]),
            rgbd_pairs=int(representative["rgbd_pairs"]),
            inliers=int(representative["inliers"]),
            inlier_ratio=float(representative["inlier_ratio"]),
            rmse_m=float(representative["rmse_m"]),
            candidate_count=len(candidates),
            independent_candidates=independent,
            keyframe_frame_index=int(representative["keyframe"].frame_index),
            oldest_candidate_frame_index=int(oldest_frame),
            candidate_frame_gap=(
                int(frame_index) - int(oldest_frame)
                if frame_index is not None
                else 0
            ),
            above_floor_inliers=int(representative.get("above_floor_inliers", 0)),
            independent_translation_span_m=translation_span,
            keyframe_to_current_x_m=float(representative["relative_pose"][0]),
            keyframe_to_current_y_m=float(representative["relative_pose"][1]),
            keyframe_to_current_yaw_deg=float(representative["relative_pose"][2]),
            evidence_type=str(
                representative.get("evidence_type", "direct_rgbd_geometry")
            ),
            place_geometry_candidates=place_count,
            place_geometry_independent=place_independent,
            geometry_major_span_m=float(
                representative.get("geometry_major_span_m", 0.0)
            ),
            geometry_minor_span_m=float(
                representative.get("geometry_minor_span_m", 0.0)
            ),
            pixel_coverage_x=float(
                representative.get("pixel_coverage_x", 0.0)
            ),
            pixel_coverage_y=float(
                representative.get("pixel_coverage_y", 0.0)
            ),
            translation_std_m=float(
                representative.get("translation_std_m", math.inf)
            ),
            yaw_std_deg=float(
                representative.get("yaw_std_deg", math.inf)
            ),
            normal_matrix_condition=float(
                representative.get("normal_matrix_condition", math.inf)
            ),
            winning_cluster_candidates=len(cluster),
            winning_cluster_observable_candidates=len(observable_witnesses),
            witness_selection_reason=(
                "single_strong_observable_witness"
                if independent < required_independent
                else "observable_cluster_witness"
            ),
        )
        self.accepted += 1
        self.last_reason = "accepted"
        self.last_result = result
        return result
