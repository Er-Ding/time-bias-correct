"""直接由地图建立的连续传播函数，全部传播顺序统一为 UE → BS。

长度单位米，角度为全局坐标中的弧度。函数的光滑延拓可供优化计算，
但只有重新计算通过地图检查的路径才能支持最终结果。
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Sequence

import numpy as np

from .diffraction import (
    InteractionSequence, diffraction_edges, rebuild_path, reflection_leg,
    shadow_directions,
)
from .raytrace2d import (
    GeometricPath2D, _backtrack_reflection_points, _segment_visible,
)
from .scene import Scene2D


def wrap_angle_rad(angle: float | np.ndarray):
    """取最短角度差，返回 [-pi, pi)。"""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class ContinuousObservation:
    observation_id: str
    aoa_rad: float
    observed_length_m: float
    angle_scale_rad: float
    length_scale_m: float

    def __post_init__(self):
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise ValueError("观测编号必须是非空字符串")
        values = (self.aoa_rad, self.observed_length_m,
                  self.angle_scale_rad, self.length_scale_m)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("观测和误差尺度必须是有限数")
        if self.angle_scale_rad <= 0 or self.length_scale_m <= 0:
            raise ValueError("角度和长度误差尺度必须为正数")
        object.__setattr__(self, "aoa_rad", float(wrap_angle_rad(self.aoa_rad)))

    def to_dict(self):
        return {
            "observation_id": self.observation_id,
            "aoa_rad": float(self.aoa_rad),
            "observed_length_m": float(self.observed_length_m),
            "angle_scale_rad": float(self.angle_scale_rad),
            "length_scale_m": float(self.length_scale_m),
        }


@dataclass(frozen=True)
class PropagationHypothesis:
    """一个完整传播顺序对应一个函数，不含代表坐标或采样方向。

令 v=A@x+t-anchor，则 L=|v|+fixed_length。无绕射时
    theta=atan2(v_y,v_x)；有固定墙角时 theta=fixed_aoa。
    """

    hypothesis_id: str
    interactions: InteractionSequence
    receiver_m: tuple[float, float]
    affine_image_matrix: tuple[tuple[float, float], tuple[float, float]]
    affine_image_offset: tuple[float, float]
    anchor_m: tuple[float, float]
    fixed_length_m: float = 0.0
    fixed_aoa_rad: float | None = None
    origin: str = "public_map"

    @property
    def diffraction_order(self):
        return sum(kind == "diffraction" for kind, _ in self.interactions)

    @property
    def reflection_order(self):
        return sum(kind == "reflection" for kind, _ in self.interactions)

    @property
    def equivalent_anchor_m(self) -> np.ndarray:
        """把镜像距离写成 |x-equivalent_anchor| 时的固定锚点。"""
        matrix = np.asarray(self.affine_image_matrix)
        return matrix.T @ (np.asarray(self.anchor_m) - self.affine_image_offset)

    def to_dict(self):
        return {
            "hypothesis_id": self.hypothesis_id,
            "interactions_ue_to_bs": [list(item) for item in self.interactions],
            "receiver_m": list(self.receiver_m),
            "affine_image_matrix": [list(row) for row in self.affine_image_matrix],
            "affine_image_offset": list(self.affine_image_offset),
            "anchor_m": list(self.anchor_m),
            "equivalent_anchor_m": self.equivalent_anchor_m.tolist(),
            "fixed_length_m": self.fixed_length_m,
            "fixed_aoa_rad": self.fixed_aoa_rad,
            "origin": self.origin,
            "legal_domain": "map_bounds_and_rebuilt_visible_path_and_diffraction_shadow",
        }


@dataclass(frozen=True)
class PropagationEvaluation:
    length_m: float
    aoa_rad: float
    length_gradient_xy: np.ndarray
    aoa_gradient_xy: np.ndarray
    path: GeometricPath2D | None
    valid: bool | None
    invalid_reason: str | None


def _point(point_m, name: str) -> np.ndarray:
    point = np.asarray(point_m, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} 必须是两个有限坐标")
    return point


def make_hypothesis(
    scene: Scene2D,
    bs_position_m: Sequence[float],
    interactions: InteractionSequence = (),
) -> PropagationHypothesis:
    """从墙面和固定墙角构造函数；固定墙角至 BS 不通则拒绝该分支。"""
    return _make_hypothesis_formula(
        scene, bs_position_m, interactions,
        {wall.wall_id: wall for wall in scene.walls},
        {edge.edge_id: edge for edge in diffraction_edges(scene)},
        validate_fixed_leg=True,
    )


def _make_hypothesis_formula(scene, bs_position_m, interactions, walls, edges,
                             *, validate_fixed_leg):
    """构造器内部先做便宜的解析筛选，再检查固定段几何。"""
    bs = _point(bs_position_m, "BS")
    if not scene.contains(bs):
        raise ValueError("BS 必须位于地图范围内")
    interactions = tuple(tuple(item) for item in interactions)
    if any(len(item) != 2 or item[0] not in {"reflection", "diffraction"}
           for item in interactions):
        raise ValueError("传播交互只支持 reflection 或 diffraction 及其地图编号")
    diffraction_indices = [i for i, (kind, _) in enumerate(interactions)
                           if kind == "diffraction"]
    reflection_ids = [key for kind, key in interactions if kind == "reflection"]
    if len(reflection_ids) > 2 or len(diffraction_indices) > 1:
        raise ValueError("连续模型当前最多支持二次反射和一次绕射")
    if any(key not in walls for key in reflection_ids):
        raise ValueError("传播分支引用了地图中不存在的墙")
    if any(a == b and a[0] == "reflection"
           for a, b in zip(interactions, interactions[1:])):
        raise ValueError("不能在同一面墙上连续反射")

    anchor = bs
    variable_interactions = interactions
    fixed_length = 0.0
    fixed_aoa = None
    if diffraction_indices:
        index = diffraction_indices[0]
        edge = edges.get(interactions[index][1])
        if edge is None:
            raise ValueError("传播分支引用了地图中不存在的绕射边缘")
        anchor = np.asarray(edge.position_m)
        after = tuple(key for _, key in interactions[index + 1:])
        image = anchor.copy()
        for key in after:
            wall = walls[key]
            image = image - 2.0 * np.dot(image - wall.start, wall.normal) * wall.normal
        incoming = image - bs
        fixed_length = float(np.linalg.norm(incoming))
        if fixed_length <= 1e-7:
            raise ValueError("fixed_edge_to_bs_leg_invalid")
        if validate_fixed_leg:
            points = reflection_leg(scene, anchor, bs, after,
                                    source_walls=edge.incident_wall_ids)
            if points is None:
                raise ValueError("fixed_edge_to_bs_leg_invalid")
        fixed_aoa = math.atan2(incoming[1], incoming[0])
        variable_interactions = interactions[:index]

    matrix = np.eye(2)
    offset = np.zeros(2)
    for _, key in variable_interactions:
        wall = walls[key]
        normal = wall.normal
        reflection = np.eye(2) - 2.0 * np.outer(normal, normal)
        shift = 2.0 * np.dot(wall.start, normal) * normal
        matrix = reflection @ matrix
        offset = reflection @ offset + shift
    serialized = json.dumps(interactions, separators=(",", ":"))
    identifier = "los" if not interactions else "h_" + sha256(serialized.encode()).hexdigest()[:20]
    return PropagationHypothesis(
        hypothesis_id=identifier, interactions=interactions,
        receiver_m=tuple(float(value) for value in bs),
        affine_image_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        affine_image_offset=tuple(float(value) for value in offset),
        anchor_m=tuple(float(value) for value in anchor),
        fixed_length_m=fixed_length, fixed_aoa_rad=fixed_aoa,
    )


def _leg_failure(scene, source, receiver, wall_ids, source_walls=(), receiver_walls=()):
    lookup = {wall.wall_id: wall for wall in scene.walls}
    points = _backtrack_reflection_points(source, receiver, [lookup[key] for key in wall_ids])
    if points is None:
        return "reflection_outside_wall_or_wrong_order", None
    nodes = [source, *points, receiver]
    for i, (start, end) in enumerate(zip(nodes[:-1], nodes[1:])):
        if np.linalg.norm(end - start) <= 1e-7:
            return "zero_length_segment", None
        allowed = list(source_walls if i == 0 else (wall_ids[i - 1],))
        allowed.extend(receiver_walls if i == len(wall_ids) else (wall_ids[i],))
        if not _segment_visible(scene, start, end, allowed_endpoint_walls=allowed):
            return "blocked_segment", None
    return None, points


def _path_failure_reason(scene, hypothesis, source):
    if not scene.contains(source):
        return "ue_outside_map"
    receiver = np.asarray(hypothesis.receiver_m)
    interactions = hypothesis.interactions
    indices = [i for i, (kind, _) in enumerate(interactions) if kind == "diffraction"]
    if not indices:
        return _leg_failure(scene, source, receiver, [key for _, key in interactions])[0] or "path_rebuild_failed"
    index = indices[0]
    edge = {edge.edge_id: edge for edge in diffraction_edges(scene)}[interactions[index][1]]
    q = np.asarray(edge.position_m)
    reason, before = _leg_failure(scene, source, q, [key for _, key in interactions[:index]],
                                 receiver_walls=edge.incident_wall_ids)
    if reason:
        return "ue_to_edge_" + reason
    reason, after = _leg_failure(scene, q, receiver, [key for _, key in interactions[index + 1:]],
                                source_walls=edge.incident_wall_ids)
    if reason:
        return "edge_to_bs_" + reason
    previous = before[-1] if before else source
    following = after[0] if after else receiver
    if not shadow_directions(scene, edge, (previous - q) / np.linalg.norm(previous - q),
                             (following - q) / np.linalg.norm(following - q)):
        return "diffraction_outside_shadow_or_degenerate"
    return "path_rebuild_failed"


def evaluate_hypothesis(
    scene: Scene2D,
    hypothesis: PropagationHypothesis,
    position_m: Sequence[float],
    *,
    check_validity: bool = True,
) -> PropagationEvaluation:
    """返回函数及其真实空间导数；valid=None 表示尚未检查几何合法性。

    不合法的物理路径仍保留固定维度的光滑延拓值，使调用方能够回退步长。
    重合锚点的长度导数不存在，明确返回 singular_endpoint，不伪装成合法解。
    """
    source = _point(position_m, "UE")
    matrix = np.asarray(hypothesis.affine_image_matrix)
    vector = matrix @ source + hypothesis.affine_image_offset - np.asarray(hypothesis.anchor_m)
    distance = float(np.linalg.norm(vector))
    singular = distance <= 1e-10
    denominator = max(distance, 1e-10)
    length_gradient = matrix.T @ (vector / denominator)
    if hypothesis.fixed_aoa_rad is None:
        aoa = math.atan2(vector[1], vector[0])
        angle_gradient = np.asarray([-vector[1], vector[0]]) @ matrix / denominator**2
    else:
        aoa = hypothesis.fixed_aoa_rad
        angle_gradient = np.zeros(2)
    path = None
    valid = None
    reason = "singular_endpoint" if singular else None
    if check_validity:
        path = None if singular else rebuild_path(scene, source, hypothesis.receiver_m,
                                                 hypothesis.interactions)
        valid = path is not None
        if not valid and not singular:
            reason = _path_failure_reason(scene, hypothesis, source)
        # A valid path must agree with its image formula; reject any convention drift.
        if valid and (abs(path.length_m - distance - hypothesis.fixed_length_m) > 1e-6
                      or abs(float(wrap_angle_rad(math.radians(path.arrival_aoa_deg) - aoa))) > 1e-7):
            valid = False
            reason = "continuous_formula_path_mismatch"
    return PropagationEvaluation(
        length_m=distance + hypothesis.fixed_length_m, aoa_rad=float(wrap_angle_rad(aoa)),
        length_gradient_xy=length_gradient, aoa_gradient_xy=angle_gradient,
        path=path, valid=valid, invalid_reason=reason,
    )
