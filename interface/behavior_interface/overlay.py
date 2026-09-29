"""画 HUD：xyz 轴 + 文本信息（右下角小坐标系 + 顶部状态条）。"""

from __future__ import annotations

import math
from typing import Dict, Optional  # noqa: F401  (Optional 给 draw_axes_overlay 用)

import cv2
import numpy as np


def _line_aa(img, p0, p1, color, thickness=2):
    cv2.line(img, p0, p1, color, thickness, lineType=cv2.LINE_AA)


def draw_axes_overlay(
    img: np.ndarray,
    yaw_rad: float,
    radius: int = 70,
    margin: int = 30,
    cam_R: Optional[np.ndarray] = None,
) -> np.ndarray:
    """在 img 右下角画**世界系**的 XYZ 三轴投影，跟随相机视角动态变化。

    X=红, Y=绿, Z=蓝。这套轴和 move(x,y) / move_to(x,y) 使用的世界坐标系一致——
    机器人沿屏幕上红轴方向运动时世界 x 增大。

    cam_R: GTA 相机 3x3 旋转矩阵（cam -> world，列分别是相机系 +X/+Y/+Z 在世界中的方向）。
           为 None 时退化为屏幕固定三轴（启动初期 fallback）。
    yaw_rad: 仅用于在指南针下方显示一个 yaw 数字与小箭头，便于读底盘朝向。
    """
    h, w = img.shape[:2]
    cx, cy = w - margin - radius, h - margin - radius

    cv2.rectangle(
        img,
        (cx - radius - 6, cy - radius - 6),
        (cx + radius + 6, cy + radius + 30),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        img,
        (cx - radius - 6, cy - radius - 6),
        (cx + radius + 6, cy + radius + 30),
        (180, 180, 180),
        1,
    )

    arrow_len = radius - 10
    origin = (cx, cy)

    # 把世界 X/Y/Z 单位向量旋到相机系：cam_vec = cam_R.T @ world_vec
    # 相机系：+X = 画面 right, +Y = 画面 up, +Z = 出屏方向（朝向观察者）
    # 屏幕投影：screen_dx = cam_vec.x, screen_dy = -cam_vec.y（图像 y 向下）
    # 颜色（BGR）：X=红, Y=绿, Z=蓝
    axes = [
        ("X", np.array([1.0, 0.0, 0.0]), (40, 40, 235)),
        ("Y", np.array([0.0, 1.0, 0.0]), (60, 200, 60)),
        ("Z", np.array([0.0, 0.0, 1.0]), (235, 160, 40)),
    ]
    if cam_R is None:
        R = np.eye(3)
    else:
        R = np.asarray(cam_R, dtype=np.float64)
    Rt = R.T

    # 先画"朝里"的轴（z<0，被画面平面挡住），再画"朝外"的轴，保证近的盖远的
    projected = []
    for label, vec, color in axes:
        cv = Rt @ vec
        sx = float(cv[0])  # 屏幕 x 方向分量
        sy = float(cv[1])  # 屏幕 y 方向分量（cv 系，未翻转 image y）
        sz = float(cv[2])  # 出屏分量；>0 朝向观察者
        projected.append((label, color, sx, sy, sz))
    projected.sort(key=lambda p: p[4])  # 升序：sz 小（朝里）的先画

    for label, color, sx, sy, sz in projected:
        ex = int(cx + sx * arrow_len)
        ey = int(cy - sy * arrow_len)
        if sz < -0.05:
            # 朝里的轴用更细+暗一点
            dim = tuple(int(c * 0.5) for c in color)
            _line_aa(img, origin, (ex, ey), dim, 1)
        else:
            _line_aa(img, origin, (ex, ey), color, 2)
        cv2.putText(img, label, (ex + 4 if sx >= 0 else ex - 14,
                                 ey + 12 if sy <= 0 else ey - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.circle(img, origin, 3, (255, 255, 255), -1)

    # 底部：底盘 yaw 数值（不画指针避免和世界轴混淆）
    yaw_txt = f"yaw={math.degrees(yaw_rad):+.1f} deg"
    cv2.putText(img, yaw_txt, (cx - radius, cy + radius + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def draw_topbar(
    img: np.ndarray,
    state: Dict,
) -> np.ndarray:
    """顶部黑色条：tick、任务、底盘 (x,y,yaw)、当前 skill、message。"""
    h, w = img.shape[:2]
    bar_h = 56
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)

    task = state.get("task", "?")
    robot = state.get("robot", "?")
    tick = state.get("tick", 0)
    fps = state.get("fps", 0.0)
    base = state.get("base_pose", {})
    bx = base.get("x", 0.0)
    by = base.get("y", 0.0)
    byaw = base.get("yaw_deg", 0.0)

    line1 = f"task={task} robot={robot} tick={tick} fps={fps:.1f}"
    cv2.putText(img, line1, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    line2 = f"base x={bx:+.2f} y={by:+.2f} yaw={byaw:+.1f}deg"
    cv2.putText(img, line2, (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)

    skill = state.get("active_skill")
    if skill:
        info = f"skill: {skill.get('name')} {skill.get('status') or ''}"
        cv2.putText(img, info, (w // 2 - 50, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 230, 60), 1, cv2.LINE_AA)

    return img


def label_image(img: np.ndarray, label: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (200, 24), (0, 0, 0), -1)
    cv2.putText(img, label, (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def make_placeholder(
    width: int,
    height: int,
    text: str,
    bg=(30, 30, 40),
) -> np.ndarray:
    img = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.putText(img, text, (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    return img
