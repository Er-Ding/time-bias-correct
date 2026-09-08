"""把 MUSIC 的 AOA/Delay 样本变成随公共偏差移动的二维候选轨迹。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S
from .reverse_compute import VectorizedWallIntersector
from .scene import Scene2D, WallSegment, ray_segment_intersection, reflect_direction
from .solver import CandidateTrajectory


@dataclass(frozen=True)
class PathObservationSample:
    """一条 MUSIC 观测峰附近的一个可复现角度/时延采样。"""

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
    wall_intersector: VectorizedWallIntersector | None = None,
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
        next_hit = (
            _nearest_wall_hit(scene, origin, direction)
            if wall_intersector is None
            else wall_intersector.nearest(origin, direction)
        )
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
    backend: str = "numpy",
    wall_chunk_size: int = 8192,
) -> list[RawReverseTrajectory]:
    """对所有 MUSIC 样本执行反向追踪，复用分块墙求交数据。

    numpy 批量计算每条射线与墙的交点；reference 保留原逐墙实现用于对照。
    两种模式均保持采样顺序、反射顺序和所有候选的完整物理字段。
    """

    if backend not in {"numpy", "reference"}:
        raise ValueError("反向追踪 backend 只能为 numpy 或 reference")
    intersector = (
        VectorizedWallIntersector(scene, wall_chunk_size=wall_chunk_size)
        if backend == "numpy" else None
    )

    raw: list[RawReverseTrajectory] = []
    for sample in samples:
        raw.extend(
            reverse_trace_sample(
                scene,
                bs_position_m,
                sample,
                max_reflections=max_reflections,
                beta_interval_m=beta_interval_m,
                wall_intersector=intersector,
            )
        )
    return raw


def _trajectory_distance_matrices(
    group: Sequence[RawReverseTrajectory],
) -> tuple[np.ndarray, np.ndarray]:
    """批量比较同一 beta 下的轨迹，返回共同区间上的最大距离与方向差。

    两条轨迹的距离是 beta 的凸函数，闭区间最大值必在端点取得。因此
    只需检查两个共同端点，不能以某一个交点处的最小距离代表整个区间。
    """

    anchors = np.asarray([item.anchor_m for item in group], dtype=float)
    directions = np.asarray([item.direction for item in group], dtype=float)
    lower = np.asarray([item.beta_min_m for item in group], dtype=float)
    upper = np.asarray([item.beta_max_m for item in group], dtype=float)
    shared_lower = np.maximum(lower[:, None], lower[None, :])
    shared_upper = np.minimum(upper[:, None], upper[None, :])
    delta_anchor = anchors[:, None, :] - anchors[None, :, :]
    delta_direction = directions[:, None, :] - directions[None, :, :]
    start_distance = np.linalg.norm(
        delta_anchor - shared_lower[:, :, None] * delta_direction, axis=2
    )
    end_distance = np.linalg.norm(
        delta_anchor - shared_upper[:, :, None] * delta_direction, axis=2
    )
    distances = np.maximum(start_distance, end_distance)
    distances[shared_lower > shared_upper] = np.inf
    direction_distances = np.degrees(
        np.arccos(np.clip(directions @ directions.T, -1.0, 1.0))
    )
    np.fill_diagonal(distances, 0.0)
    np.fill_diagonal(direction_distances, 0.0)
    return distances, direction_distances


def _bounded_diameter_clusters(
    distances: np.ndarray,
    direction_distances: np.ndarray,
    *,
    position_radius_m: float,
    direction_radius_deg: float,
) -> list[list[int]]:
    """确定性贪心分组：新成员必须与簇内每个已有成员都满足阈值。

    与单链连通分量不同，A 接近 B、B 接近 C 不足以将 A/B/C 合并。
    多个簇都兼容时，优先加入最大两两距离最小的簇；并列保持稳定顺序。
    """

    compatible = (
        (distances <= position_radius_m)
        & (direction_distances <= direction_radius_deg)
    )
    clusters: list[list[int]] = []
    for index in range(len(distances)):
        options = [
            (float(np.max(distances[index, members])), cluster_index)
            for cluster_index, members in enumerate(clusters)
            if bool(np.all(compatible[index, members]))
        ]
        if options:
            _, cluster_index = min(options)
            clusters[cluster_index].append(index)
        else:
            clusters.append([index])
    return clusters


def cluster_reverse_candidates(
    raw_candidates: Sequence[RawReverseTrajectory],
    *,
    position_radius_m: float = 1.5,
    direction_radius_deg: float = 5.0,
) -> list[CandidateTrajectory]:
    """汇集采样候选，按来源峰/反射结构聚类，再选真实成员作为代表。

    每个簇具有受限的两两轨迹距离；代表为加权距离和最小的真实成员。
    代表的几何、采样观测和有效 beta 区间完整保留，不合成平均反射路径。
    样本频率只供诊断，不能把同一观测的多次采样当作独立测量加权。
    """

    if (
        not np.isfinite(position_radius_m)
        or not np.isfinite(direction_radius_deg)
        or position_radius_m <= 0.0
        or direction_radius_deg <= 0.0
    ):
        raise ValueError("聚类半径必须为有限正数")
    grouped: dict[tuple[str, str], list[RawReverseTrajectory]] = {}
    observation_sample_ids: dict[str, set[str]] = {}
    for candidate in raw_candidates:
        grouped.setdefault((candidate.observation_id, candidate.topology_id), []).append(candidate)
        observation_sample_ids.setdefault(candidate.observation_id, set()).add(candidate.sample_id)

    representatives: list[CandidateTrajectory] = []
    for (observation_id, topology_id), group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: item.sample_id)
        distances, direction_distances = _trajectory_distance_matrices(group)
        components = _bounded_diameter_clusters(
            distances,
            direction_distances,
            position_radius_m=position_radius_m,
            direction_radius_deg=direction_radius_deg,
        )
        for component_index, component in enumerate(components):
            members = [group[index] for index in component]
            weights = np.asarray([member.weight for member in members], dtype=float)
            weights /= np.sum(weights)
            member_distances = distances[np.ix_(component, component)]
            medoid_index = int(np.argmin(member_distances @ weights))
            representative = members[medoid_index]
            beta_min_values = np.asarray([member.beta_min_m for member in members])
            beta_max_values = np.asarray([member.beta_max_m for member in members])
            observed_delays = [member.observed_delay_s for member in members]
            observed_angles = [member.observed_aoa_global_rad for member in members]
            empirical_frequency = len({member.sample_id for member in members}) / max(
                1, len(observation_sample_ids[observation_id])
            )
            metadata = {
                "topology_id": topology_id,
                "reflection_wall_ids": list(representative.reflection_wall_ids),
                "raw_count": len(members),
                "source_sample_ids": [member.sample_id for member in members],
                "representative_sample_id": representative.sample_id,
                "observed_aoa_global_rad": float(representative.observed_aoa_global_rad),
                "observed_delay_s": float(representative.observed_delay_s),
                "prefix_length_m": float(representative.prefix_length_m),
                "endpoint_origin_m": list(representative.endpoint_origin_m),
                "reflection_points_m": [list(point) for point in representative.reflection_points_m],
                "observed_delay_range_s": [float(min(observed_delays)), float(max(observed_delays))],
                "observed_aoa_range_rad": [float(min(observed_angles)), float(max(observed_angles))],
                "empirical_frequency": float(empirical_frequency),
                "empirical_frequency_denominator": "distinct_samples_with_valid_reverse_candidate",
                "observation_valid_sample_count": len(observation_sample_ids[observation_id]),
                "member_beta_min_range_m": [float(np.min(beta_min_values)), float(np.max(beta_min_values))],
                "member_beta_max_range_m": [float(np.min(beta_max_values)), float(np.max(beta_max_values))],
                "shared_beta_interval_m": [float(np.max(beta_min_values)), float(np.min(beta_max_values))],
                "representative_rule": "weighted_medoid_actual_member",
                "cluster_distance_rule": "maximum_over_shared_beta_interval",
                "cluster_linkage_rule": "deterministic_greedy_complete_compatibility",
                "maximum_member_trajectory_distance_m": float(np.max(member_distances)),
                "maximum_member_direction_difference_deg": float(
                    np.max(direction_distances[np.ix_(component, component)])
                ),
                "members": [
                    {
                        "sample_id": member.sample_id,
                        "anchor_m": list(member.anchor_m),
                        "direction": list(member.direction),
                        "beta_min_m": float(member.beta_min_m),
                        "beta_max_m": float(member.beta_max_m),
                        "observed_aoa_global_rad": float(member.observed_aoa_global_rad),
                        "observed_delay_s": float(member.observed_delay_s),
                        "weight": float(member.weight),
                    }
                    for member in members
                ],
            }
            representatives.append(
                CandidateTrajectory(
                    observation_id=observation_id,
                    candidate_id=f"{observation_id}:{topology_id}:cluster_{component_index}",
                    anchor_m=representative.anchor_m,
                    direction=representative.direction,
                    beta_min_m=representative.beta_min_m,
                    beta_max_m=representative.beta_max_m,
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
