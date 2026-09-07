"""把 MUSIC 的 AOA/Delay 样本变成随公共偏差移动的二维候选轨迹。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S
from .scene import Scene2D, WallSegment, ray_segment_intersection, reflect_direction
from .solver import CandidateTrajectory


@dataclass(frozen=True)
class PathObservationSample:
    """一条 MUSIC 观测的一个可复现扰动样本。"""

    observation_id: str
    sample_id: str
    aoa_global_rad: float
    delay_s: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.aoa_global_rad):
            raise ValueError("aoa_global_rad 必须是有限数")
        if not np.isfinite(self.delay_s) or self.delay_s < 0.0:
            raise ValueError("delay_s 必须是有限非负数")
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("weight 必须是有限正数")


@dataclass(frozen=True)
class RawReverseTrajectory:
    """聚类前的反向追踪轨迹；保留每个原始样本的完整几何信息。"""

    observation_id: str
    sample_id: str
    topology_id: str
    anchor_m: tuple[float, float]
    direction: tuple[float, float]
    beta_min_m: float
    beta_max_m: float
    prefix_length_m: float
    endpoint_origin_m: tuple[float, float]
    reflection_wall_ids: tuple[str, ...]
    reflection_points_m: tuple[tuple[float, float], ...]
    observed_aoa_global_rad: float
    observed_delay_s: float
    weight: float

    def point(self, beta_m: float) -> np.ndarray:
        return np.asarray(self.anchor_m) - float(beta_m) * np.asarray(self.direction)


def wrap_angle_rad(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def local_to_global_aoa(local_aoa_rad: float, bs_boresight_rad: float) -> float:
    """将 BS 阵列局部角转换为地图全局角。"""

    return wrap_angle_rad(float(local_aoa_rad) + float(bs_boresight_rad))


def global_to_local_aoa(global_aoa_rad: float, bs_boresight_rad: float) -> float:
    """将地图全局角转换为 BS 阵列局部角。"""

    return wrap_angle_rad(float(global_aoa_rad) - float(bs_boresight_rad))


def _nearest_wall_hit(
    scene: Scene2D,
    origin: np.ndarray,
    direction: np.ndarray,
) -> tuple[float, WallSegment, np.ndarray] | None:
    hits: list[tuple[float, str, WallSegment, np.ndarray]] = []
    for wall in scene.walls:
        hit = ray_segment_intersection(origin, direction, wall, min_distance_m=1e-6)
        if hit is None:
            continue
        distance, _, point = hit
        hits.append((distance, wall.wall_id, wall, point))
    if not hits:
        return None
    distance, _, wall, point = min(hits, key=lambda item: (item[0], item[1]))
    return float(distance), wall, point


def _distance_to_bounds(scene: Scene2D, origin: np.ndarray, direction: np.ndarray) -> float:
    x_min, x_max, y_min, y_max = scene.bounds_m
    distances: list[float] = []
    for coordinate, component, lower, upper in zip(
        origin, direction, (x_min, y_min), (x_max, y_max), strict=True
    ):
        if component > 1e-12:
            distances.append((upper - coordinate) / component)
        elif component < -1e-12:
            distances.append((lower - coordinate) / component)
    positive = [distance for distance in distances if distance > 1e-7]
    return float(min(positive)) if positive else 0.0


def _make_raw_candidate(
    sample: PathObservationSample,
    *,
    origin: np.ndarray,
    direction: np.ndarray,
    prefix_length_m: float,
    free_distance_m: float,
    wall_ids: Sequence[str],
    reflection_points: Sequence[np.ndarray],
    global_beta_interval_m: tuple[float, float],
) -> RawReverseTrajectory | None:
    observed_distance_m = sample.delay_s * SPEED_OF_LIGHT_M_S
    # remaining = observed_distance - prefix - beta，且 UE 必须在下一面墙之前。
    physical_min = observed_distance_m - prefix_length_m - free_distance_m
    physical_max = observed_distance_m - prefix_length_m
    beta_min = max(float(global_beta_interval_m[0]), float(physical_min))
    beta_max = min(float(global_beta_interval_m[1]), float(physical_max))
    if beta_min > beta_max:
        return None
    anchor = origin + (observed_distance_m - prefix_length_m) * direction
    topology_id = "los" if not wall_ids else "-".join(wall_ids)
    return RawReverseTrajectory(
        observation_id=sample.observation_id,
        sample_id=sample.sample_id,
        topology_id=topology_id,
        anchor_m=(float(anchor[0]), float(anchor[1])),
        direction=(float(direction[0]), float(direction[1])),
        beta_min_m=float(beta_min),
        beta_max_m=float(beta_max),
        prefix_length_m=float(prefix_length_m),
        endpoint_origin_m=(float(origin[0]), float(origin[1])),
        reflection_wall_ids=tuple(wall_ids),
        reflection_points_m=tuple((float(p[0]), float(p[1])) for p in reflection_points),
        observed_aoa_global_rad=float(sample.aoa_global_rad),
        observed_delay_s=float(sample.delay_s),
        weight=float(sample.weight),
    )


def reverse_trace_sample(
    scene: Scene2D,
    bs_position_m: Sequence[float],
    sample: PathObservationSample,
    *,
    max_reflections: int,
    beta_interval_m: tuple[float, float],
) -> list[RawReverseTrajectory]:
    """对单个 AOA/Delay 样本生成 0、1、2 次反射解释。"""

    if max_reflections not in (0, 1, 2):
        raise ValueError("第一版只支持最多二次反射")
    if beta_interval_m[0] >= beta_interval_m[1]:
        raise ValueError("beta_interval_m 下界必须小于上界")
    origin = np.asarray(bs_position_m, dtype=float)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("bs_position_m 必须是有限二维坐标")
    direction = np.asarray(
        [math.cos(sample.aoa_global_rad), math.sin(sample.aoa_global_rad)], dtype=float
    )
    direction /= np.linalg.norm(direction)
    prefix_length = 0.0
    wall_ids: list[str] = []
    reflection_points: list[np.ndarray] = []
    candidates: list[RawReverseTrajectory] = []

    for reflection_order in range(max_reflections + 1):
        next_hit = _nearest_wall_hit(scene, origin, direction)
        free_distance = (
            next_hit[0] if next_hit is not None else _distance_to_bounds(scene, origin, direction)
        )
        candidate = _make_raw_candidate(
            sample,
            origin=origin,
            direction=direction,
            prefix_length_m=prefix_length,
            free_distance_m=free_distance,
            wall_ids=wall_ids,
            reflection_points=reflection_points,
            global_beta_interval_m=beta_interval_m,
        )
        if candidate is not None:
            candidates.append(candidate)

        if reflection_order == max_reflections or next_hit is None:
            break
        distance, wall, point = next_hit
        prefix_length += distance
        wall_ids.append(wall.wall_id)
        reflection_points.append(point)
        origin = point
        direction = reflect_direction(direction, wall)

    return candidates


def generate_reverse_candidates(
    scene: Scene2D,
    bs_position_m: Sequence[float],
    samples: Iterable[PathObservationSample],
    *,
    max_reflections: int,
    beta_interval_m: tuple[float, float],
) -> list[RawReverseTrajectory]:
    """对所有 MUSIC 样本执行反向追踪。"""

    raw: list[RawReverseTrajectory] = []
    for sample in samples:
        raw.extend(
            reverse_trace_sample(
                scene,
                bs_position_m,
                sample,
                max_reflections=max_reflections,
                beta_interval_m=beta_interval_m,
            )
        )
    return raw


def _direction_separation_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _minimum_shared_interval_trajectory_distance_m(
    first: RawReverseTrajectory,
    second: RawReverseTrajectory,
) -> float:
    """返回两条轨迹在共同有效 beta 区间内能够达到的最小距离。

    两条轨迹之差仍是 beta 的一次函数，因此平方距离是一个一元二次函数。
    先解析求出最低点，再把 beta 截到共同有效区间即可。区间不相交时返回
    正无穷，表示两个样本不能聚到一起。
    """

    beta_min = max(first.beta_min_m, second.beta_min_m)
    beta_max = min(first.beta_max_m, second.beta_max_m)
    if beta_min > beta_max:
        return float("inf")

    anchor_delta = np.asarray(first.anchor_m) - np.asarray(second.anchor_m)
    direction_delta = np.asarray(first.direction) - np.asarray(second.direction)
    direction_energy = float(np.dot(direction_delta, direction_delta))
    if direction_energy <= np.finfo(float).eps:
        return float(np.linalg.norm(anchor_delta))

    unconstrained_beta = float(
        np.dot(anchor_delta, direction_delta) / direction_energy
    )
    closest_beta = float(np.clip(unconstrained_beta, beta_min, beta_max))
    separation = anchor_delta - closest_beta * direction_delta
    return float(np.linalg.norm(separation))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """计算确定性的加权中位数；恰好各占一半时取两个中间值的均值。"""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    halfway = 0.5 * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, halfway, side="left"))
    if (
        index + 1 < ordered_values.size
        and np.isclose(cumulative[index], halfway, rtol=0.0, atol=1e-14)
    ):
        return float(0.5 * (ordered_values[index] + ordered_values[index + 1]))
    return float(ordered_values[index])


def _connected_components(
    group: Sequence[RawReverseTrajectory],
    *,
    position_radius_m: float,
    direction_radius_deg: float,
) -> list[list[int]]:
    neighbors: list[list[int]] = [[] for _ in group]
    for first in range(len(group)):
        for second in range(first + 1, len(group)):
            trajectory_distance = _minimum_shared_interval_trajectory_distance_m(
                group[first], group[second]
            )
            direction_distance = _direction_separation_deg(
                np.asarray(group[first].direction), np.asarray(group[second].direction)
            )
            if (
                trajectory_distance <= position_radius_m
                and direction_distance <= direction_radius_deg
            ):
                neighbors[first].append(second)
                neighbors[second].append(first)

    components: list[list[int]] = []
    unseen = set(range(len(group)))
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component: list[int] = []
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbor in neighbors[index]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def cluster_reverse_candidates(
    raw_candidates: Sequence[RawReverseTrajectory],
    *,
    position_radius_m: float = 1.5,
    direction_radius_deg: float = 5.0,
) -> list[CandidateTrajectory]:
    """只在同一观测、同一反射结构内部聚类并生成可复现代表轨迹。"""

    if position_radius_m <= 0.0 or direction_radius_deg <= 0.0:
        raise ValueError("聚类半径必须为正数")
    grouped: dict[tuple[str, str], list[RawReverseTrajectory]] = {}
    observation_sample_ids: dict[str, set[str]] = {}
    for candidate in raw_candidates:
        grouped.setdefault(
            (candidate.observation_id, candidate.topology_id), []
        ).append(candidate)
        observation_sample_ids.setdefault(candidate.observation_id, set()).add(
            candidate.sample_id
        )

    representatives: list[CandidateTrajectory] = []
    for (observation_id, topology_id), group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: item.sample_id)
        components = _connected_components(
            group,
            position_radius_m=position_radius_m,
            direction_radius_deg=direction_radius_deg,
        )
        for component_index, component in enumerate(components):
            members = [group[index] for index in component]
            weights = np.asarray([member.weight for member in members], dtype=float)
            weights /= np.sum(weights)
            anchors = np.asarray([member.anchor_m for member in members], dtype=float)
            directions = np.asarray([member.direction for member in members], dtype=float)
            anchor = np.sum(weights[:, None] * anchors, axis=0)
            direction = np.sum(weights[:, None] * directions, axis=0)
            direction /= np.linalg.norm(direction)
            beta_min_values = np.asarray(
                [member.beta_min_m for member in members], dtype=float
            )
            beta_max_values = np.asarray(
                [member.beta_max_m for member in members], dtype=float
            )
            # 扰动样本互为替代，并非必须同时成立的多条证据。上下界各取加权
            # 中位数，可抵抗少量极端样本，又能给出一条明确的代表有效区间。
            beta_min = _weighted_median(beta_min_values, weights)
            beta_max = _weighted_median(beta_max_values, weights)
            observed_delays = [member.observed_delay_s for member in members]
            observed_angles = [member.observed_aoa_global_rad for member in members]
            empirical_frequency = len({member.sample_id for member in members}) / max(
                1, len(observation_sample_ids[observation_id])
            )
            metadata = {
                "topology_id": topology_id,
                "reflection_wall_ids": list(members[0].reflection_wall_ids),
                "raw_count": len(members),
                "source_sample_ids": [member.sample_id for member in members],
                "prefix_length_m": float(
                    np.sum(weights * np.asarray([member.prefix_length_m for member in members]))
                ),
                "endpoint_origin_m": list(members[0].endpoint_origin_m),
                "reflection_points_m": [
                    list(point) for point in members[0].reflection_points_m
                ],
                "observed_delay_range_s": [
                    float(min(observed_delays)),
                    float(max(observed_delays)),
                ],
                "observed_aoa_range_rad": [
                    float(min(observed_angles)),
                    float(max(observed_angles)),
                ],
                "empirical_frequency": float(empirical_frequency),
                "member_beta_min_range_m": [
                    float(np.min(beta_min_values)),
                    float(np.max(beta_min_values)),
                ],
                "member_beta_max_range_m": [
                    float(np.min(beta_max_values)),
                    float(np.max(beta_max_values)),
                ],
                "representative_rule": "weighted_mean_with_weighted_median_endpoints",
                "cluster_distance_rule": "minimum_over_shared_beta_interval",
            }
            representatives.append(
                CandidateTrajectory(
                    observation_id=observation_id,
                    candidate_id=f"{observation_id}:{topology_id}:cluster_{component_index}",
                    anchor_m=anchor,
                    direction=direction,
                    beta_min_m=beta_min,
                    beta_max_m=beta_max,
                    # 簇频率是候选先验，不是位置残差的逆方差。第一版没有校准
                    # 候选协方差，因此所有代表轨迹使用相同拟合权重。
                    weight=1.0,
                    metadata=metadata,
                )
            )
    return representatives


__all__ = [
    "PathObservationSample",
    "RawReverseTrajectory",
    "cluster_reverse_candidates",
    "generate_reverse_candidates",
    "global_to_local_aoa",
    "local_to_global_aoa",
    "reverse_trace_sample",
    "wrap_angle_rad",
]
