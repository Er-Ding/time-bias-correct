"""在整个允许偏差区间保留反向路径的每个合法末段。

先枚举与传播长度范围相交的物理线段，再给同一观测、同一完整传播顺序
选共同的合法参考偏差。没有共同参考值的成员分组处理。聚类仍使用真实
位置点和原 DBSCAN 规则；不在墙外外推点，也不把偏差采样当成独立观测。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from collections import Counter, defaultdict

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S as C
from .initial_candidates import (
    InitialCandidatePoint, InitialCandidateGenerationResult,
    _distance_to_scene_exit, _nearest_wall,
)
from .reverse_compute import VectorizedWallIntersector
from .scene import reflect_direction
from .timing import stage


_EPS = 1e-7


@dataclass
class _Segment:
    sample: object
    sample_id: str
    origin: np.ndarray
    direction: np.ndarray
    prefix: float
    free: float
    interactions: tuple
    nodes: tuple
    lower: float
    upper: float

    def point(self, bias):
        endpoint = self.origin + (C*(self.sample.delay_s-bias)-self.prefix)*self.direction
        walls = tuple(key for kind,key in self.interactions if kind == "reflection")
        wall_points = tuple(p for (kind,_),p in zip(self.interactions,self.nodes) if kind == "reflection")
        return InitialCandidatePoint(
            observation_id=self.sample.observation_id, sample_id=self.sample_id,
            topology_id=json.dumps(self.interactions, separators=(",", ":")),
            reference_bias_s=float(bias), position_m=tuple(endpoint),
            reflection_wall_ids=walls, reflection_points_m=wall_points,
            observed_aoa_global_rad=self.sample.aoa_global_rad, observed_delay_s=self.sample.delay_s,
            prefix_length_m=self.prefix, endpoint_origin_m=tuple(self.origin),
            endpoint_direction=tuple(self.direction), endpoint_free_distance_m=self.free,
            weight=self.sample.weight, propagation_interactions=self.interactions,
            interaction_points_m=self.nodes, parent_sample_id=self.sample.sample_id,
        )


def _walk(scene, sample, origin, direction, prefix, interactions, nodes, *,
          sample_id, bounds, max_reflections, nearest):
    """沿一条确定方向的射线遍历；每段的有效 b 区间解析计算，非偏差网格。"""
    interactions, nodes = list(interactions), list(nodes)
    maximum_length = C*(sample.delay_s-bounds[0])
    order = sum(kind == "reflection" for kind,_ in interactions)
    for segment_index in range(max_reflections-order+1):
        if prefix >= maximum_length-_EPS:
            break
        hit = nearest(scene, origin, direction)
        boundary = _distance_to_scene_exit(scene, origin, direction)
        inside = hit is not None and hit[0] <= boundary + _EPS
        free = min(float(hit[0]), boundary) if inside else boundary
        if free <= 2*_EPS:
            break
        lower = max(bounds[0], sample.delay_s-(prefix+free-_EPS)/C)
        upper = min(bounds[1], sample.delay_s-(prefix+_EPS)/C)
        if lower < upper:
            yield _Segment(sample, f"{sample_id}:segment_{segment_index:02d}",
                           origin.copy(), direction.copy(), float(prefix), float(free),
                           tuple(interactions), tuple(nodes), float(lower), float(upper))
        if not inside or order == max_reflections or prefix+free >= maximum_length-_EPS:
            break
        distance, wall, point = hit
        if (any(kind == "diffraction" for kind,_ in interactions)
                and min(np.linalg.norm(point-wall.start), np.linalg.norm(point-wall.end)) <= _EPS):
            break
        prefix += distance
        interactions.append(("reflection", wall.wall_id))
        nodes.append(tuple(point))
        origin, direction = point, reflect_direction(direction, wall)
        order += 1


def generate_bias_interval_points(scene, bs_position_m, samples, *, bias_interval_s,
                                  reference_bias_s=0., max_reflections=2, backend="numpy",
                                  wall_chunk_size=8192, max_diffractions=0,
                                  diffraction_directions_per_sample=4,
                                  diffraction_angle_tolerance_deg=3.):
    bounds = np.asarray(bias_interval_s, float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError("bias_interval_s 必须为有限、严格递增的区间")
    if not np.isfinite(reference_bias_s) or not bounds[0] <= reference_bias_s <= bounds[1]:
        raise ValueError("参考偏差必须位于允许区间内")
    if isinstance(max_reflections, bool) or max_reflections not in (0,1,2):
        raise ValueError("当前支持 0、1、2 次反射")
    if isinstance(max_diffractions, bool) or max_diffractions not in (0,1):
        raise ValueError("当前支持 0 或 1 次绕射")
    if backend not in ("numpy", "reference"):
        raise ValueError("全偏差候选后端必须为 numpy 或 reference")
    if (isinstance(diffraction_directions_per_sample, bool)
            or not isinstance(diffraction_directions_per_sample, (int,np.integer))
            or diffraction_directions_per_sample < 1):
        raise ValueError("绕射方向数必须为正整数")
    if not np.isfinite(diffraction_angle_tolerance_deg) or not 0 < diffraction_angle_tolerance_deg < 90:
        raise ValueError("绕射匹配角容差必须介于 0 和 90 度")
    bs = np.asarray(bs_position_m, float)
    if bs.shape != (2,) or not np.all(np.isfinite(bs)) or not scene.contains(bs):
        raise ValueError("BS 必须位于场景内")
    samples = list(samples)
    if len({(s.observation_id,s.sample_id) for s in samples}) != len(samples):
        raise ValueError("同一观测的 sample_id 不能重复")
    intersector = VectorizedWallIntersector(scene, wall_chunk_size=wall_chunk_size) if backend == "numpy" else None
    nearest = _nearest_wall if intersector is None else lambda _scene,o,d: intersector.nearest(o,d)
    segments = []
    with stage("T08_specular"):
        for sample in samples:
            direction = np.array([np.cos(sample.aoa_global_rad), np.sin(sample.aoa_global_rad)])
            segments.extend(_walk(scene, sample, bs, direction, 0., (), (),
                                  sample_id=sample.sample_id, bounds=bounds,
                                  max_reflections=max_reflections, nearest=nearest))
    prefix_count = 0
    if max_diffractions:
        from .diffraction import shadow_directions
        from .diffraction_prefixes import get_diffraction_prefixes
        with stage("T08_prefix_build"):
            prefixes, _cache = get_diffraction_prefixes(scene, bs, max_reflections)
        prefix_count = len(prefixes)
        ordinals = Counter()
        with stage("T08_fan_trace"):
            for sample in samples:
                ordinal = ordinals[sample.observation_id]
                ordinals[sample.observation_id] += 1
                maximum_length = C*(sample.delay_s-bounds[0])
                for prefix_index, (edge,walls,points,length,angle,toward_bs) in enumerate(prefixes):
                    error = (angle-sample.aoa_global_rad+np.pi) % (2*np.pi)-np.pi
                    if abs(error) > math.radians(diffraction_angle_tolerance_deg) or maximum_length <= length+_EPS:
                        continue
                    for branch in range(diffraction_directions_per_sample):
                        fraction = ((ordinal+.5)*.6180339887498949 + branch/diffraction_directions_per_sample) % 1
                        direction = np.array([np.cos(2*np.pi*fraction), np.sin(2*np.pi*fraction)])
                        if not shadow_directions(scene, edge, toward_bs, direction):
                            continue
                        interactions = tuple([*(("reflection",w) for w in walls), ("diffraction",edge.edge_id)])
                        nodes = tuple([*(tuple(p) for p in points), edge.position_m])
                        segments.extend(_walk(scene, sample, np.asarray(edge.position_m), direction,
                            length, interactions, nodes, sample_id=f"{sample.sample_id}:d{prefix_index:05d}:{branch:04d}",
                            bounds=bounds, max_reflections=max_reflections, nearest=nearest))
    grouped = defaultdict(list)
    for segment in segments:
        grouped[(segment.sample.observation_id, segment.interactions)].append(segment)
    points, reference_groups = [], []
    with stage("T08_reference_groups"):
        for (observation, interactions), group in sorted(grouped.items()):
            remaining = sorted(group, key=lambda s:s.sample_id)
            first = True
            while remaining:
                if first and any(s.lower <= reference_bias_s <= s.upper for s in remaining):
                    bias = float(reference_bias_s)
                else:
                    # 已由解析区间保证至少有一段在该值有效；不是通过扫描 b 寻找分支。
                    mids = sorted(((s.lower+s.upper)/2,s.sample_id) for s in remaining)
                    bias = float(mids[len(mids)//2][0])
                selected = [s for s in remaining if s.lower <= bias <= s.upper]
                assert selected
                selected_ids = {s.sample_id for s in selected}
                remaining = [s for s in remaining if s.sample_id not in selected_ids]
                # 偶然落在 BS 的点改用组内另一个共同合法偏差，不丢整段。
                common_lo, common_hi = max(s.lower for s in selected), min(s.upper for s in selected)
                if any(np.linalg.norm(np.asarray(s.point(bias).position_m)-bs) <= _EPS for s in selected):
                    for fraction in (.38196601125, .61803398875, .27182818285):
                        alternate = common_lo+(common_hi-common_lo)*fraction
                        if all(np.linalg.norm(np.asarray(s.point(alternate).position_m)-bs)>_EPS for s in selected):
                            bias = alternate
                            break
                    else:
                        raise ValueError("参考点退化在 BS，无法找到组内合法参考值")
                points.extend(s.point(bias) for s in selected)
                reference_groups.append({"observation_id": observation,
                    "propagation_interactions": [list(i) for i in interactions],
                    "reference_bias_s": bias, "point_count": len(selected),
                    "common_valid_bias_interval_s": [common_lo,common_hi]})
                first = False
    accepted = {(p.observation_id,p.parent_sample_id) for p in points}
    rejected = [{"observation_id":s.observation_id,"sample_id":s.sample_id,
                 "reason":"no_valid_segment_in_allowed_bias_interval"} for s in samples
                if (s.observation_id,s.sample_id) not in accepted]
    return InitialCandidateGenerationResult(points, rejected, {
        "candidate_generation_mode": "full_bias_interval", "bias_interval_s": bounds.tolist(),
        "reference_bias_s": float(reference_bias_s),
        "reference_bias_source": "configured_preference_then_common_valid_segment_intervals",
        "input_sample_count": len(samples), "candidate_count":len(points),
        "initial_point_count":len(points), "segment_count":len(segments),
        "accepted_sample_count":len(accepted), "rejected_sample_count":len(rejected),
        "observation_candidate_counts":dict(Counter(p.observation_id for p in points)),
        "reference_groups":reference_groups, "builds_bias_trajectories":False,
        "segment_coverage": "analytic_intersection_with_entire_allowed_bias_interval",
        "max_reflections":max_reflections,"max_diffractions":max_diffractions,
        "visible_bs_edge_prefix_count":prefix_count,
        "completeness_scope":"sampled_arrival_angles_and_diffraction_directions_within_configured_interaction_limits",
    })
