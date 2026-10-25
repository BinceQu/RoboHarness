#!/usr/bin/env python
"""2D 位姿图优化：把回环约束摊回整条轨迹。

扫描匹配只给得出「相邻两处的相对位姿」，一路积下来必然漂。回环说的是
「这两个隔了很久的地方其实是同一处」，位姿图的活儿就是同时满足这两类
约束——把回环处的落差沿着轨迹分摊回去，而不是在接缝上硬掰一下。

节点是 submap 的位姿，边是相对位姿观测。误回环比不回环更致命（会把整
张图撕坏），所以用 Huber 核压住个别离群约束。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# 迭代到位移增量小于这个量就停，再算下去只是在数值噪声里打转
CONVERGE_M = 1e-4
CONVERGE_DEG = 1e-3
MAX_ITERATIONS = 30
# LM 阻尼：位姿图常有欠约束方向（比如只有一条链时的整体旋转）
DAMPING = 1e-6
# Huber 阈值：残差超过它的边按线性而不是平方计权，个别误回环就掀不翻整张图
HUBER_M = 0.30
HUBER_DEG = 12.0


def wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


@dataclass
class PoseEdge:
    """节点 j 在节点 i 局部系里的相对位姿观测。"""

    i: int
    j: int
    dx: float
    dy: float
    dyaw_deg: float
    # 权重就是信息矩阵的对角：越可信越大
    weight_xy: float = 1.0
    weight_yaw: float = 1.0
    # 回环边才过 Huber。里程边是推算出来的，本来就该被完全信任
    robust: bool = False


def _residual(
    poses: np.ndarray, edge: PoseEdge
) -> Tuple[np.ndarray, float, float]:
    """返回 (误差向量, cos θi, sin θi)。误差在节点 i 的局部系里。"""
    xi, yi, ti = poses[edge.i]
    xj, yj, tj = poses[edge.j]
    angle = math.radians(ti)
    cos_i, sin_i = math.cos(angle), math.sin(angle)
    dx, dy = xj - xi, yj - yi
    # 把地图系位移转进 i 的局部系，才好和观测比
    local_x = cos_i * dx + sin_i * dy
    local_y = -sin_i * dx + cos_i * dy
    return (
        np.array([
            local_x - edge.dx,
            local_y - edge.dy,
            wrap_deg(tj - ti - edge.dyaw_deg),
        ]),
        cos_i,
        sin_i,
    )


def _huber_scale(error: np.ndarray, robust: bool) -> float:
    """残差太大就降权，免得一条误回环把整张图拽歪。"""
    if not robust:
        return 1.0
    lin = math.hypot(error[0], error[1])
    scale = 1.0
    if lin > HUBER_M:
        scale = min(scale, HUBER_M / lin)
    ang = abs(error[2])
    if ang > HUBER_DEG:
        scale = min(scale, HUBER_DEG / ang)
    return scale


def optimize_graph(
    poses: Sequence[Sequence[float]],
    edges: Sequence[PoseEdge],
    *,
    fixed: int = 0,
    iterations: int = MAX_ITERATIONS,
) -> np.ndarray:
    """高斯-牛顿求解位姿图，返回优化后的 (N, 3) 位姿数组。

    ``fixed`` 那个节点被钉死——位姿图只约束相对关系，不钉住的话整张图
    可以整体平移旋转，解不唯一。
    """
    result = np.array(poses, dtype=np.float64).reshape(-1, 3)
    n = result.shape[0]
    if n == 0 or not edges:
        return result
    # 角度以度为单位参与求解，雅可比里的 ∂(位移)/∂(角度) 得跟着换算
    deg = math.pi / 180.0

    for _ in range(int(iterations)):
        hessian = np.zeros((3 * n, 3 * n), dtype=np.float64)
        gradient = np.zeros(3 * n, dtype=np.float64)
        for edge in edges:
            error, cos_i, sin_i = _residual(result, edge)
            scale = _huber_scale(error, edge.robust)
            info = np.diag([
                edge.weight_xy * scale,
                edge.weight_xy * scale,
                edge.weight_yaw * scale,
            ])
            dx = result[edge.j][0] - result[edge.i][0]
            dy = result[edge.j][1] - result[edge.i][1]
            jac_i = np.array([
                [-cos_i, -sin_i, (-sin_i * dx + cos_i * dy) * deg],
                [sin_i, -cos_i, (-cos_i * dx - sin_i * dy) * deg],
                [0.0, 0.0, -1.0],
            ])
            jac_j = np.array([
                [cos_i, sin_i, 0.0],
                [-sin_i, cos_i, 0.0],
                [0.0, 0.0, 1.0],
            ])
            bi, bj = 3 * edge.i, 3 * edge.j
            hessian[bi:bi + 3, bi:bi + 3] += jac_i.T @ info @ jac_i
            hessian[bi:bi + 3, bj:bj + 3] += jac_i.T @ info @ jac_j
            hessian[bj:bj + 3, bi:bi + 3] += jac_j.T @ info @ jac_i
            hessian[bj:bj + 3, bj:bj + 3] += jac_j.T @ info @ jac_j
            gradient[bi:bi + 3] += jac_i.T @ info @ error
            gradient[bj:bj + 3] += jac_j.T @ info @ error

        base = 3 * int(fixed)
        hessian[base:base + 3, :] = 0.0
        hessian[:, base:base + 3] = 0.0
        hessian[base:base + 3, base:base + 3] = np.eye(3)
        gradient[base:base + 3] = 0.0
        hessian[np.diag_indices_from(hessian)] += DAMPING

        try:
            step = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        result += step.reshape(-1, 3)
        result[:, 2] = (result[:, 2] + 180.0) % 360.0 - 180.0

        moved = step.reshape(-1, 3)
        if (
            np.abs(moved[:, :2]).max() < CONVERGE_M
            and np.abs(moved[:, 2]).max() < CONVERGE_DEG
        ):
            break
    return result


def total_error(
    poses: Sequence[Sequence[float]], edges: Sequence[PoseEdge]
) -> float:
    """加权残差平方和，用来判断优化有没有真的改善。"""
    grid = np.array(poses, dtype=np.float64).reshape(-1, 3)
    total = 0.0
    for edge in edges:
        error, _, _ = _residual(grid, edge)
        total += (
            edge.weight_xy * (error[0] ** 2 + error[1] ** 2)
            + edge.weight_yaw * error[2] ** 2
        )
    return float(total)


def relative_pose(
    origin: Sequence[float], target: Sequence[float]
) -> Tuple[float, float, float]:
    """target 在 origin 局部系里的位姿——里程边就是这么算出来的。"""
    angle = math.radians(origin[2])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx = float(target[0]) - float(origin[0])
    dy = float(target[1]) - float(origin[1])
    return (
        cos_a * dx + sin_a * dy,
        -sin_a * dx + cos_a * dy,
        wrap_deg(float(target[2]) - float(origin[2])),
    )


def compose_pose(
    origin: Sequence[float], relative: Sequence[float]
) -> Tuple[float, float, float]:
    """把 ``origin`` 和在它局部系表达的 ``relative`` 组合成绝对位姿。"""
    angle = math.radians(float(origin[2]))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx, dy = float(relative[0]), float(relative[1])
    return (
        float(origin[0]) + cos_a * dx - sin_a * dy,
        float(origin[1]) + sin_a * dx + cos_a * dy,
        wrap_deg(float(origin[2]) + float(relative[2])),
    )


def inverse_pose(pose: Sequence[float]) -> Tuple[float, float, float]:
    """返回刚体位姿的 SE(2) 逆变换。"""
    return relative_pose(pose, (0.0, 0.0, 0.0))
