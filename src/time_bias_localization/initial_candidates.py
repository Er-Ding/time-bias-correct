"""先生成固定参考偏差下的位置点，点聚类完成后才建立代表轨迹。

初始点只表示 ``c * (observed_delay - reference_bias)`` 的反向追踪终点。
参考偏差是公开的计算约定，不是偏差估计或真值。只在该参考值有效的反射
路径才进入点集；超出场景或反射次数的样本显式记录为拒绝，不能从其他
偏差处的候选轨迹取点来补齐。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from .candidates import PathObservationSample
from .constants import SPEED_OF_LIGHT_M_S
from .reverse_compute import VectorizedWallIntersector
from .scene import Scene2D, WallSegment, ray_segment_intersection, reflect_direction

if TYPE_CHECKING:
    from .solver import CandidateTrajectory


_ENDPOINT_TOLERANCE_M = 1e-7


@dataclass(frozen=True)
class InitialCandidatePoint:
    """一次角度/时延采样在参考偏差下的实际终点，不含候选轨迹。"""

    observation_id: str
    sample_id: str
    topology_id: str
    reference_bias_s: float
    position_m: tuple[float, float]
    reflection_wall_ids: tuple[str, ...]
    reflection_points_m: tuple[tuple[float, float], ...]
    observed_aoa_global_rad: float
    observed_delay_s: float
    prefix_length_m: float
    endpoint_origin_m: tuple[float, float]
    endpoint_direction: tuple[float, float]
    endpoint_free_distance_m: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("position_m", "endpoint_origin_m", "endpoint_direction"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} 必须是有限二维坐标或方向")
        if not np.isclose(np.linalg.norm(self.endpoint_direction), 1.0, atol=1e-8):
            raise ValueError("endpoint_direction 必须是单位方向")
        for name in ("reference_bias_s", "observed_aoa_global_rad", "observed_delay_s",
                     "prefix_length_m", "endpoint_free_distance_m", "weight"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} 必须为有限数")
        if self.observed_delay_s < 0 or self.prefix_length_m < 0:
            raise ValueError("观测时延和反射前缀长度不能为负数")
        if self.endpoint_free_distance_m <= 0 or self.weight <= 0:
            raise ValueError("末段可用长度和权重必须为正数")
        if len(self.reflection_wall_ids) != len(self.reflection_points_m):
            raise ValueError("反射墙数量必须与反射点数量一致")


@dataclass(frozen=True)
class InitialCandidateGenerationResult:
    points: list[InitialCandidatePoint]
    rejected_samples: list[dict[str, Any]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RepresentativeCandidatePoint:
    """点簇的真实成员代表；成员保留实际位置与原始采样来源。"""

    candidate_id: str
    point: InitialCandidatePoint
    members: tuple[InitialCandidatePoint, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.members or self.point not in self.members:
            raise ValueError("代表点必须是非空点簇中的真实成员")
        key = (self.point.observation_id, self.point.reflection_wall_ids, self.point.reference_bias_s)
        if any((member.observation_id, member.reflection_wall_ids, member.reference_bias_s) != key
               for member in self.members):
            raise ValueError("点簇成员必须来自同一观测、反射墙序列和参考 bias")


def _nearest_wall(
    scene: Scene2D, origin: np.ndarray, direction: np.ndarray,
) -> tuple[float, WallSegment, np.ndarray] | None:
    hits = []
    for wall in scene.walls:
        hit = ray_segment_intersection(origin, direction, wall, min_distance_m=1e-6)
        if hit is not None:
            distance, _, point = hit
            hits.append((distance, wall.wall_id, wall, point))
    if not hits:
        return None
    distance, _, wall, point = min(hits, key=lambda row: (row[0], row[1]))
    return float(distance), wall, point


def _distance_to_scene_exit(scene: Scene2D, origin: np.ndarray, direction: np.ndarray) -> float:
    """向外离开边界时返回零，不能跳过零交点继续追踪到地图外的墙。"""

    x_min, x_max, y_min, y_max = scene.bounds_m
    distances = []
    for coordinate, component, lower, upper in zip(
        origin, direction, (x_min, y_min), (x_max, y_max), strict=True
    ):
        if component > 1e-12:
            distances.append((upper - coordinate) / component)
        elif component < -1e-12:
            distances.append((lower - coordinate) / component)
    return float(max(0.0, min(distances))) if distances else 0.0


def generate_initial_candidate_points(
    scene: Scene2D,
    bs_position_m: Sequence[float],
    samples: Iterable[PathObservationSample],
    *,
    reference_bias_s: float = 0.0,
    max_reflections: int = 2,
    backend: str = "numpy",
    wall_chunk_size: int = 8192,
) -> InitialCandidateGenerationResult:
    """每次采样沿反向射线走完参考传播长度，只生成一个初始位置点。

    先后经过的墙只在射线确实到达墙面后计入反射。终点恰落在墙面、地图
    边界或反向段起点（容差 1e-7 米）时拒绝，避免零长末段的歧义。
    numpy 与 reference 只在求交实现上不同；二者均不建立 bias 轨迹。
    """

    if (isinstance(max_reflections, (bool, np.bool_))
            or not isinstance(max_reflections, (int, np.integer))
            or max_reflections not in (0, 1, 2)):
        raise ValueError("初始点反向追踪只支持 0、1、2 次反射")
    if not np.isfinite(reference_bias_s):
        raise ValueError("reference_bias_s 必须为有限数")
    if backend not in {"numpy", "reference"}:
        raise ValueError("初始点反向追踪 backend 只能为 numpy 或 reference")
    bs = np.asarray(bs_position_m, dtype=float)
    if bs.shape != (2,) or not np.all(np.isfinite(bs)) or not scene.contains(bs):
        raise ValueError("bs_position_m 必须是场景内的有限二维坐标")
    intersector = (
        VectorizedWallIntersector(scene, wall_chunk_size=wall_chunk_size)
        if backend == "numpy" else None
    )
    points: list[InitialCandidatePoint] = []
    rejected: list[dict[str, Any]] = []
    sample_counts: dict[str, int] = {}
    accepted_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for sample in samples:
        key = (sample.observation_id, sample.sample_id)
        if key in seen:
            raise ValueError(f"同一观测内的 sample_id 不能重复：{key}")
        seen.add(key)
        sample_counts[sample.observation_id] = sample_counts.get(sample.observation_id, 0) + 1
        target_length = float((sample.delay_s - reference_bias_s) * SPEED_OF_LIGHT_M_S)
        direction = np.asarray([math.cos(sample.aoa_global_rad), math.sin(sample.aoa_global_rad)])
        direction /= np.linalg.norm(direction)
        origin = bs.copy()
        prefix = 0.0
        walls: list[str] = []
        reflection_points: list[tuple[float, float]] = []
        reason = "nonpositive_reference_length" if target_length <= _ENDPOINT_TOLERANCE_M else None
        if reason is None:
            for reflection_order in range(max_reflections + 1):
                hit = (
                    _nearest_wall(scene, origin, direction)
                    if intersector is None else intersector.nearest(origin, direction)
                )
                boundary_distance = _distance_to_scene_exit(scene, origin, direction)
                hit_inside_scene = hit is not None and hit[0] <= boundary_distance + _ENDPOINT_TOLERANCE_M
                free_distance = min(float(hit[0]), boundary_distance) if hit_inside_scene else boundary_distance
                remaining = target_length - prefix
                if remaining <= _ENDPOINT_TOLERANCE_M:
                    reason = "endpoint_on_segment_origin"
                    break
                if remaining < free_distance - _ENDPOINT_TOLERANCE_M:
                    endpoint = origin + remaining * direction
                    if np.linalg.norm(endpoint - bs) <= _ENDPOINT_TOLERANCE_M:
                        reason = "endpoint_at_bs"
                        break
                    point = InitialCandidatePoint(
                        observation_id=sample.observation_id,
                        sample_id=sample.sample_id,
                        topology_id="los" if not walls else "-".join(walls),
                        reference_bias_s=float(reference_bias_s),
                        position_m=(float(endpoint[0]), float(endpoint[1])),
                        reflection_wall_ids=tuple(walls),
                        reflection_points_m=tuple(reflection_points),
                        observed_aoa_global_rad=float(sample.aoa_global_rad),
                        observed_delay_s=float(sample.delay_s),
                        prefix_length_m=float(prefix),
                        endpoint_origin_m=(float(origin[0]), float(origin[1])),
                        endpoint_direction=(float(direction[0]), float(direction[1])),
                        endpoint_free_distance_m=float(free_distance),
                        weight=float(sample.weight),
                    )
                    points.append(point)
                    accepted_counts[sample.observation_id] = accepted_counts.get(sample.observation_id, 0) + 1
                    break
                if remaining <= free_distance + _ENDPOINT_TOLERANCE_M:
                    reason = "endpoint_on_wall" if hit_inside_scene else "endpoint_on_scene_boundary"
                    break
                if not hit_inside_scene:
                    reason = "leaves_scene_before_endpoint"
                    break
                if reflection_order == max_reflections:
                    reason = "exceeds_max_reflections"
                    break
                assert hit is not None
                distance, wall, reflection_point = hit
                prefix += distance
                walls.append(wall.wall_id)
                reflection_points.append((float(reflection_point[0]), float(reflection_point[1])))
                origin = reflection_point
                direction = reflect_direction(direction, wall)
        if reason is not None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            rejected.append({
                "observation_id": sample.observation_id,
                "sample_id": sample.sample_id,
                "reference_bias_s": float(reference_bias_s),
                "observed_aoa_global_rad": float(sample.aoa_global_rad),
                "observed_delay_s": float(sample.delay_s),
                "reference_path_length_m": target_length,
                "completed_prefix_length_m": float(prefix),
                "reflection_wall_ids": list(walls),
                "reason": reason,
            })
    return InitialCandidateGenerationResult(points=points, rejected_samples=rejected, diagnostics={
        "reference_bias_s": float(reference_bias_s),
        "reference_bias_source": "configured_reference_not_estimate_or_ground_truth",
        "input_sample_count": len(seen),
        "initial_point_count": len(points),
        "rejected_sample_count": len(rejected),
        "observation_input_sample_counts": sample_counts,
        "observation_initial_point_counts": accepted_counts,
        "rejection_reason_counts": rejection_counts,
        "max_reflections": int(max_reflections),
        "endpoint_tolerance_m": _ENDPOINT_TOLERANCE_M,
        "endpoint_boundary_policy": "reject_wall_boundary_or_zero_length_endpoints",
        "one_point_per_sample": True,
        "builds_bias_trajectories": False,
        "backend": backend,
    })


@dataclass(frozen=True)
class InitialCandidateClusteringResult:
    """聚类结果及全部点的去向；离群点只记录，不建立代表点。"""

    representatives: list[RepresentativeCandidatePoint]
    noise_points: list[InitialCandidatePoint]
    memberships: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def cluster_initial_candidate_points(
    points: Sequence[InitialCandidatePoint], *, position_radius_m: float = 1.5,
    min_samples: int = 5, return_diagnostics: bool = False,
) -> list[RepresentativeCandidatePoint] | InitialCandidateClusteringResult:
    """同一来源峰、同一墙序列内，对参考位置进行确定性 DBSCAN 聚类。

    ``position_radius_m`` 保留原参数名，但现在表示邻近距离 eps，不限制整簇
    的最大跨度。邻域内点数包含自身；达到 ``min_samples`` 的点为核心点，
    核心点通过相邻关系连接成簇。非核心点只归入最近核心点所在的簇；距离
    并列时按簇最小 sample_id 选择，不能通过边界点连接两个核心簇。

    密度只计等权样本数量，不使用谱值。每簇代表仍为加权距离和最小的真实
    成员；并列按 sample_id。离群点不生成代表，开启 return_diagnostics
    可取得核心点、边界点、离群点及全部成员归属。方向与轨迹不参与聚类。
    """

    if (isinstance(position_radius_m, (bool, np.bool_))
            or not isinstance(position_radius_m, Real)
            or not np.isfinite(position_radius_m) or position_radius_m <= 0):
        raise ValueError("position_radius_m (DBSCAN eps) 必须为有限正数，不能为布尔值")
    if (isinstance(min_samples, (bool, np.bool_))
            or not isinstance(min_samples, (int, np.integer)) or min_samples <= 0):
        raise ValueError("min_samples 必须为正整数，包含点自身，不能为布尔值")
    references = {point.reference_bias_s for point in points}
    if len(references) > 1:
        raise ValueError("同一轮初始点必须使用相同 reference_bias_s")
    grouped: dict[tuple[str, tuple[str, ...]], list[InitialCandidatePoint]] = {}
    valid_samples: dict[str, set[str]] = {}
    seen: set[tuple[str, str]] = set()
    for point in points:
        key = (point.observation_id, point.sample_id)
        if key in seen:
            raise ValueError(f"初始点 sample_id 重复：{key}")
        seen.add(key)
        grouped.setdefault((point.observation_id, point.reflection_wall_ids), []).append(point)
        valid_samples.setdefault(point.observation_id, set()).add(point.sample_id)

    representatives: list[RepresentativeCandidatePoint] = []
    noise_points: list[InitialCandidatePoint] = []
    memberships: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    for (observation_id, wall_ids), group in sorted(grouped.items()):
        group = sorted(group, key=lambda point: point.sample_id)
        positions = np.asarray([point.position_m for point in group])
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        neighbors = distances <= position_radius_m
        neighbor_counts = np.sum(neighbors, axis=1)
        core_mask = neighbor_counts >= min_samples
        labels = np.full(len(group), -1, dtype=int)
        core_components: list[list[int]] = []
        # 只遍历核心点之间的边，边界点不能把两团核心点连接起来。
        for seed in np.flatnonzero(core_mask):
            if labels[seed] != -1:
                continue
            label = len(core_components)
            labels[seed] = label
            pending = [int(seed)]
            component: list[int] = []
            while pending:
                index = pending.pop()
                component.append(index)
                adjacent = np.flatnonzero(neighbors[index] & core_mask & (labels == -1))
                labels[adjacent] = label
                pending.extend(int(item) for item in adjacent)
            core_components.append(sorted(component))
        core_indices = np.flatnonzero(core_mask)
        for index in np.flatnonzero(~core_mask):
            adjacent = core_indices[neighbors[index, core_indices]]
            if len(adjacent):
                # 最小核心距离优先；同距时簇序已由稳定的 sample_id 决定。
                nearest = min(adjacent, key=lambda item: (distances[index, item], labels[item]))
                labels[index] = labels[nearest]

        candidate_ids: dict[int, str] = {}
        for component_index in range(len(core_components)):
            component = np.flatnonzero(labels == component_index)
            members = tuple(group[index] for index in component)
            member_distances = distances[np.ix_(component, component)]
            weights = np.asarray([member.weight for member in members], dtype=float)
            weights /= np.max(weights)
            weights /= np.sum(weights)
            representative = members[int(np.argmin(member_distances @ weights))]
            # 墙编号自身允许含 '-'，不能靠拼接墙编号保证候选 ID 唯一。
            candidate_id = f"{observation_id}:point_cluster_{len(representatives):06d}"
            candidate_ids[component_index] = candidate_id
            core_sample_ids = [group[index].sample_id for index in component if core_mask[index]]
            border_sample_ids = [group[index].sample_id for index in component if not core_mask[index]]
            metadata = {
                "topology_id": representative.topology_id,
                "reflection_wall_ids": list(representative.reflection_wall_ids),
                "raw_count": len(members),
                "source_sample_ids": [member.sample_id for member in members],
                "representative_sample_id": representative.sample_id,
                "reference_bias_s": float(representative.reference_bias_s),
                "initial_position_m": list(representative.position_m),
                "representative_rule": "weighted_medoid_actual_member",
                "cluster_algorithm": "dbscan",
                "cluster_distance_rule": "euclidean_at_reference_bias",
                "cluster_linkage_rule": "density_connected_core_points_with_border_assignment",
                "position_radius_m": float(position_radius_m),
                "eps_m": float(position_radius_m),
                "min_samples": int(min_samples),
                "min_samples_includes_self": True,
                "density_weighting": "equal_sample_counts",
                "core_sample_ids": core_sample_ids,
                "border_sample_ids": border_sample_ids,
                "core_point_count": len(core_sample_ids),
                "border_point_count": len(border_sample_ids),
                "maximum_member_point_distance_m": float(np.max(member_distances)),
                "uses_trajectory_distance": False,
                "uses_direction_threshold": False,
                "observation_valid_sample_count": len(valid_samples[observation_id]),
                "empirical_frequency": len(members) / len(valid_samples[observation_id]),
                "empirical_frequency_denominator": "distinct_samples_with_valid_initial_point",
            }
            representatives.append(RepresentativeCandidatePoint(candidate_id, representative, members, metadata))
        for index, point in enumerate(group):
            label = int(labels[index])
            role = "core" if core_mask[index] else "border" if label >= 0 else "noise"
            if role == "noise":
                noise_points.append(point)
            memberships.append({
                "observation_id": point.observation_id,
                "sample_id": point.sample_id,
                "topology_id": point.topology_id,
                "reflection_wall_ids": list(point.reflection_wall_ids),
                "reference_bias_s": float(point.reference_bias_s),
                "position_m": list(point.position_m),
                "candidate_id": candidate_ids.get(label),
                "role": role,
                "neighbor_count": int(neighbor_counts[index]),
                "is_representative": label >= 0 and point.sample_id == representatives[
                    len(representatives) - len(core_components) + label
                ].point.sample_id,
            })
        group_summaries.append({
            "observation_id": observation_id,
            "reflection_wall_ids": list(wall_ids),
            "reference_bias_s": float(group[0].reference_bias_s),
            "input_point_count": len(group),
            "cluster_count": len(core_components),
            "core_point_count": int(np.count_nonzero(core_mask)),
            "border_point_count": int(np.count_nonzero((labels >= 0) & ~core_mask)),
            "noise_point_count": int(np.count_nonzero(labels < 0)),
            "noise_sample_ids": [group[index].sample_id for index in np.flatnonzero(labels < 0)],
        })
    result = InitialCandidateClusteringResult(representatives, noise_points, memberships, {
        "algorithm": "dbscan",
        "eps_m": float(position_radius_m),
        "position_radius_m": float(position_radius_m),
        "min_samples": int(min_samples),
        "min_samples_includes_self": True,
        "density_weighting": "equal_sample_counts",
        "border_assignment_rule": "nearest_core_then_stable_component_order",
        "grouping_rule": "same_observation_and_full_reflection_wall_sequence",
        "reference_bias_s": float(next(iter(references))) if references else None,
        "input_point_count": len(points),
        "cluster_count": len(representatives),
        "core_point_count": sum(row["role"] == "core" for row in memberships),
        "border_point_count": sum(row["role"] == "border" for row in memberships),
        "noise_point_count": len(noise_points),
        "clustered_point_count": len(points) - len(noise_points),
        "noise_policy": "record_without_representative_or_trajectory",
        "group_summaries": group_summaries,
        "builds_bias_trajectories": False,
    })
    return result if return_diagnostics else representatives


def build_representative_trajectories(
    representatives: Sequence[RepresentativeCandidatePoint],
    beta_interval_m: tuple[float, float],
) -> list[CandidateTrajectory]:
    """仅为聚类后代表建立合法末段上的 ``p(beta)=anchor-beta*direction``。

    轨迹保留该真实成员在参考偏差下选择的固定反射墙序列，不枚举新的墙
    组合。合法区间由代表末段起止位置与全局偏差区间相交得到。
    """

    from .solver import CandidateTrajectory

    bounds = np.asarray(beta_interval_m, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError("beta_interval_m 必须是有限且严格递增的二元区间")
    trajectories: list[CandidateTrajectory] = []
    seen_ids: set[str] = set()
    for representative in representatives:
        point = representative.point
        if representative.candidate_id in seen_ids:
            raise ValueError("代表点 candidate_id 不能重复")
        seen_ids.add(representative.candidate_id)
        reference_beta = point.reference_bias_s * SPEED_OF_LIGHT_M_S
        if not bounds[0] <= reference_beta <= bounds[1]:
            raise ValueError("代表点的参考 bias 必须位于求解区间内")
        direction = np.asarray(point.endpoint_direction)
        position = np.asarray(point.position_m)
        origin = np.asarray(point.endpoint_origin_m)
        remaining = float(np.dot(position - origin, direction))
        if (
            remaining <= _ENDPOINT_TOLERANCE_M
            or remaining >= point.endpoint_free_distance_m - _ENDPOINT_TOLERANCE_M
            or not np.allclose(origin + remaining * direction, position, atol=1e-7, rtol=0)
        ):
            raise ValueError("代表初始点必须位于其已记录的合法末段内部")
        # 从实际代表点开始建立轨迹，不能倒回原始点集批量建立轨迹再筛选。
        anchor = position + reference_beta * direction
        physical_max = reference_beta + remaining
        physical_min = physical_max - point.endpoint_free_distance_m
        metadata = dict(representative.metadata)
        metadata.update({
            "point_cluster_id": representative.candidate_id,
            "initial_position_m": list(point.position_m),
            "reference_bias_s": float(point.reference_bias_s),
            "representative_sample_id": point.sample_id,
            "observed_aoa_global_rad": float(point.observed_aoa_global_rad),
            "observed_delay_s": float(point.observed_delay_s),
            "prefix_length_m": float(point.prefix_length_m),
            "endpoint_origin_m": list(point.endpoint_origin_m),
            "endpoint_free_distance_m": float(point.endpoint_free_distance_m),
            "reflection_wall_ids": list(point.reflection_wall_ids),
            "reflection_points_m": [list(position) for position in point.reflection_points_m],
            "trajectory_source": "cluster_representative_point_only",
            "trajectory_topology_policy": "fixed_to_reference_point_reflection_sequence",
            "initial_point": asdict(point),
            "members": [asdict(member) for member in representative.members],
        })
        trajectories.append(CandidateTrajectory(
            observation_id=point.observation_id,
            candidate_id=representative.candidate_id,
            anchor_m=anchor,
            direction=direction,
            beta_min_m=max(float(bounds[0]), physical_min),
            beta_max_m=min(float(bounds[1]), physical_max),
            weight=1.0,
            metadata=metadata,
        ))
    return trajectories


__all__ = [
    "InitialCandidateClusteringResult",
    "InitialCandidateGenerationResult",
    "InitialCandidatePoint",
    "RepresentativeCandidatePoint",
    "build_representative_trajectories",
    "cluster_initial_candidate_points",
    "generate_initial_candidate_points",
]
