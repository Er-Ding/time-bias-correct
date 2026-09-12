"""从 BS、公共地图和角度/时延样本建立绕射候选；不读取真值。"""
from __future__ import annotations

import json
import math
import time
import numpy as np

from .constants import SPEED_OF_LIGHT_M_S
from .diffraction import diffraction_edges, shadow_directions
from .scene import reflect_direction
from .diffraction_prefixes import get_diffraction_prefixes
from .reverse_compute import VectorizedWallIntersector
from .timing import stage


def generate_diffraction_points(scene, bs, samples, *, reference_bias_s,
                                max_reflections, directions_per_sample, angle_tolerance_deg):
    from .initial_candidates import InitialCandidatePoint, _distance_to_scene_exit

    if (isinstance(directions_per_sample, (bool, np.bool_))
            or not isinstance(directions_per_sample, (int, np.integer)) or directions_per_sample < 1):
        raise ValueError("绕射方向采样数必须为正整数")
    if not np.isfinite(angle_tolerance_deg) or not 0 < angle_tolerance_deg < 90:
        raise ValueError("绕射角度匹配容差必须介于 0 与 90 度")
    # BS 到边缘的可见反射前缀只计算一次，与未知 UE 和偏差无关。
    with stage("T08_prefix_build"):
        prefixes, prefix_cache_hit = get_diffraction_prefixes(scene, bs, max_reflections)
    intersector = VectorizedWallIntersector(scene)
    points_out = []
    angle_matches = attempted = 0
    counts = {}
    sample_ordinals = {}
    samples = list(samples)
    fan_started = last_progress = time.monotonic()
    with stage("T08_fan_trace"):
        for sample_index, sample in enumerate(samples):
            ordinal = sample_ordinals.get(sample.observation_id, 0)
            sample_ordinals[sample.observation_id] = ordinal + 1
            target = (sample.delay_s - reference_bias_s) * SPEED_OF_LIGHT_M_S
            for prefix_index, (edge, prefix_walls, prefix_points, prefix_length, angle, toward_bs) in enumerate(prefixes):
                angle_error = (angle - sample.aoa_global_rad + np.pi) % (2 * np.pi) - np.pi
                if abs(angle_error) > math.radians(angle_tolerance_deg) or target <= prefix_length + 1e-7:
                    continue
                angle_matches += 1
                for branch in range(directions_per_sample):
                    attempted += 1
                    # 各样本的扇面错开；多次采样不能被当作额外独立观测。
                    fraction = ((ordinal + 0.5) * 0.6180339887498949 + branch / directions_per_sample) % 1.0
                    phi = 2 * np.pi * fraction
                    direction = np.asarray([math.cos(phi), math.sin(phi)])
                    if not shadow_directions(scene, edge, toward_bs, direction):
                        continue
                    origin = np.asarray(edge.position_m)
                    prefix = prefix_length
                    wall_ids = list(prefix_walls)
                    reflection_points = [tuple(p) for p in prefix_points]
                    interactions = [*(('reflection', key) for key in prefix_walls), ('diffraction', edge.edge_id)]
                    interaction_points = [*reflection_points, edge.position_m]
                    for order in range(len(prefix_walls), max_reflections + 1):
                        hit = intersector.nearest(origin, direction)
                        boundary = _distance_to_scene_exit(scene, origin, direction)
                        inside_hit = hit is not None and hit[0] <= boundary + 1e-7
                        free = min(hit[0], boundary) if inside_hit else boundary
                        remaining = target - prefix
                        if 1e-7 < remaining < free - 1e-7:
                            endpoint = origin + remaining * direction
                            if np.linalg.norm(endpoint - bs) <= 1e-7:
                                break
                            topology = json.dumps(interactions, separators=(",", ":"))
                            points_out.append(InitialCandidatePoint(
                                observation_id=sample.observation_id,
                                sample_id=f"{sample.sample_id}:d{prefix_index:05d}:{branch:04d}",
                                parent_sample_id=sample.sample_id, topology_id=topology,
                                reference_bias_s=float(reference_bias_s), position_m=tuple(endpoint),
                                reflection_wall_ids=tuple(wall_ids), reflection_points_m=tuple(reflection_points),
                                observed_aoa_global_rad=float(sample.aoa_global_rad), observed_delay_s=float(sample.delay_s),
                                prefix_length_m=prefix, endpoint_origin_m=tuple(origin),
                                endpoint_direction=tuple(direction), endpoint_free_distance_m=float(free),
                                weight=float(sample.weight), propagation_interactions=tuple(interactions),
                                interaction_points_m=tuple(interaction_points),
                            ))
                            counts[sample.observation_id] = counts.get(sample.observation_id, 0) + 1
                            break
                        if remaining <= free + 1e-7 or not inside_hit or order == max_reflections:
                            break
                        distance, wall, point = hit
                        # 镜面反射不能发生在墙角。
                        if min(np.linalg.norm(point - wall.start), np.linalg.norm(point - wall.end)) <= 1e-7:
                            break
                        prefix += distance
                        wall_ids.append(wall.wall_id)
                        reflection_points.append(tuple(point))
                        interactions.append(("reflection", wall.wall_id))
                        interaction_points.append(tuple(point))
                        origin, direction = point, reflect_direction(direction, wall)
            now = time.monotonic()
            if now - last_progress >= 5:
                print(f"[绕射反向追踪] 已处理 {sample_index + 1}/{len(samples)} 个角度/时延样本，"
                      f"得到 {len(points_out)} 个候选，用时 {now - fan_started:.1f} 秒", flush=True)
                last_progress = now
    return points_out, {
        "model": "2d_vertical_edges_single_shadow_diffraction_with_specular_reflections",
        "amplitude_model": "not_used_in_reverse_candidate_generation",
        "edge_count": len(diffraction_edges(scene)), "visible_bs_edge_prefix_count": len(prefixes),
        "public_prefix_cache_hit": prefix_cache_hit,
        "prefix_geometry": "batched_exact_image_method_with_public_first_wall_visibility",
        "sample_prefix_angle_match_count": angle_matches, "attempted_direction_count": attempted,
        "directions_per_sample": directions_per_sample, "angle_tolerance_deg": float(angle_tolerance_deg),
        "initial_point_count": len(points_out), "observation_point_counts": counts,
        "mechanism_label_source": "map_hypothesis_not_observed_path_ground_truth",
        "angular_sampling": "deterministic_stratified_fan_across_observation_samples",
    }
