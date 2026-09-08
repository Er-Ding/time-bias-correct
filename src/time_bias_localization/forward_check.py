"""不使用真值的已选路径正向几何检查；不替代全路径集合或 CSI 匹配。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Hashable

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S
from .raytrace2d import _backtrack_reflection_points, _build_path, _is_path_visible
from .scene import Scene2D, reflect_direction
from .solver import CandidateTrajectory


def _observation_residuals(
    predicted_angle: float,
    predicted_delay_s: float,
    observed: Mapping[str, float] | None,
) -> dict[str, float] | None:
    if observed is None:
        return None
    angle = observed.get("aoa_global_rad", observed.get("observed_aoa_global_rad"))
    delay = observed.get("delay_s", observed.get("observed_delay_s"))
    if angle is None or delay is None:
        return None
    angle_error = float((predicted_angle - float(angle) + np.pi) % (2 * np.pi) - np.pi)
    delay_error = float(predicted_delay_s - float(delay))
    return {
        "aoa_error_rad": angle_error,
        "aoa_error_deg": float(np.degrees(angle_error)),
        "delay_error_s": delay_error,
        "delay_error_ns": delay_error * 1e9,
    }


def forward_check_solution(
    scene: Scene2D,
    bs_position_m: Sequence[float],
    selected_candidates: Mapping[Hashable, CandidateTrajectory] | Sequence[CandidateTrajectory],
    position_m: Sequence[float],
    beta_m: float,
    *,
    max_reflections: int = 2,
    observed_peaks: Mapping[Hashable, Mapping[str, float]] | None = None,
    validity_tolerance_m: float = 1e-7,
    reflection_tolerance_deg: float = 1e-4,
) -> dict[str, Any]:
    """从估计 UE 到 BS 重建已选拓扑，检查镜面反射、墙段与遮挡。

    墙序列来自代表候选，反射点必须由当前估计位置重新计算，不能照搬反向
    候选的反射点。AOA/时延分别对照代表采样及可选的原始粗网格峰。所有路径
    只使用同一个 beta/c。结果是诊断，既不改解，也不根据残差作科学接受判定。

    计算量与已选拓扑数、场景墙数线性相关；没有枚举所有反射墙组合，因此
    不检查遗漏路径、额外路径、复振幅或 CSI，也不声称全路径集合匹配。
    """

    if max_reflections not in (0, 1, 2):
        raise ValueError("正向检查仅支持直射及最多二次镜面反射")
    source = np.asarray(position_m, dtype=float)
    receiver = np.asarray(bs_position_m, dtype=float)
    if (
        source.shape != (2,)
        or receiver.shape != (2,)
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(receiver))
        or not np.isfinite(beta_m)
    ):
        raise ValueError("UE、BS 和 beta 必须为有限二维坐标及有限偏差")
    if (
        not np.isfinite(validity_tolerance_m)
        or validity_tolerance_m < 0
        or not np.isfinite(reflection_tolerance_deg)
        or reflection_tolerance_deg < 0
    ):
        raise ValueError("几何检查容差必须为有限非负数")
    candidates = (
        list(selected_candidates.values())
        if isinstance(selected_candidates, Mapping)
        else list(selected_candidates)
    )
    wall_lookup = {wall.wall_id: wall for wall in scene.walls}
    bias_s = float(beta_m / SPEED_OF_LIGHT_M_S)
    inside_scene = bool(scene.contains(source) and scene.contains(receiver))
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        metadata = candidate.metadata
        # 候选的序列从 BS 反向出发；正向传播顺序必须反转。
        wall_ids = list(reversed(metadata.get("reflection_wall_ids", [])))
        reasons: list[str] = []
        beta_valid = bool(candidate.is_valid(beta_m, validity_tolerance_m))
        if not beta_valid:
            reasons.append("beta_outside_candidate_interval")
        if not inside_scene:
            reasons.append("endpoint_outside_scene_bounds")
        if "reflection_wall_ids" not in metadata:
            reasons.append("missing_reflection_topology")
        if len(wall_ids) > max_reflections:
            reasons.append("reflection_order_exceeds_limit")
        if any(wall_id not in wall_lookup for wall_id in wall_ids):
            reasons.append("reflection_wall_not_found")
        if any(first == second for first, second in zip(wall_ids, wall_ids[1:])):
            reasons.append("consecutive_identical_reflection_walls")
        row: dict[str, Any] = {
            "observation_id": str(candidate.observation_id),
            "candidate_id": str(candidate.candidate_id),
            "representative_sample_id": metadata.get("representative_sample_id"),
            "reflection_wall_ids": wall_ids,
            "beta_in_candidate_interval": beta_valid,
            "trajectory_position_residual_m": float(np.linalg.norm(source - candidate.point(beta_m))),
            "visible": False,
            "specular_reflection_error_deg": None,
            "path_nodes_m": None,
            "prediction": None,
            "sample_residuals": None,
            "original_peak_residuals": None,
        }
        # 区间越界仍可算几何残差；缺墙/非法端点等则没有可解释的路径。
        if not any(reason != "beta_outside_candidate_interval" for reason in reasons):
            walls = [wall_lookup[wall_id] for wall_id in wall_ids]
            points = _backtrack_reflection_points(source, receiver, walls)
            if points is None:
                reasons.append("no_specular_path_inside_selected_wall_segments")
            else:
                nodes = np.asarray([source, *points, receiver], dtype=float)
                segment_vectors = np.diff(nodes, axis=0)
                segment_lengths = np.linalg.norm(segment_vectors, axis=1)
                row["path_nodes_m"] = nodes.tolist()
                if np.any(segment_lengths <= 1e-9):
                    reasons.append("zero_length_path_segment")
                else:
                    directions = segment_vectors / segment_lengths[:, None]
                    reflection_errors = [
                        float(np.degrees(np.arccos(np.clip(
                            np.dot(reflect_direction(directions[index], wall), directions[index + 1]),
                            -1.0, 1.0,
                        ))))
                        for index, wall in enumerate(walls)
                    ]
                    row["specular_reflection_error_deg"] = reflection_errors
                    if any(error > reflection_tolerance_deg for error in reflection_errors):
                        reasons.append("specular_reflection_law_mismatch")
                    visible = _is_path_visible(scene, source, receiver, walls, points)
                    row["visible"] = bool(visible)
                    if not visible:
                        reasons.append("path_blocked_by_scene_wall")
                    path = _build_path(source, receiver, walls, points)
                    predicted_angle = float(np.deg2rad(path.arrival_aoa_deg))
                    predicted_delay = float(path.delay_s + bias_s)
                    row["prediction"] = {
                        "aoa_global_rad": predicted_angle,
                        "aoa_global_deg": float(path.arrival_aoa_deg),
                        "geometric_delay_s": float(path.delay_s),
                        "predicted_observed_delay_s": predicted_delay,
                        "length_m": float(path.length_m),
                    }
                    row["sample_residuals"] = _observation_residuals(
                        predicted_angle, predicted_delay, metadata
                    )
                    original_peak = (
                        observed_peaks.get(candidate.observation_id)
                        if observed_peaks is not None else None
                    )
                    row["original_peak_residuals"] = _observation_residuals(
                        predicted_angle, predicted_delay, original_peak
                    )
        row["valid"] = not reasons
        row["failure_reasons"] = reasons
        rows.append(row)
    return {
        "scope": "selected_topologies_2d_specular",
        "checks_all_scene_path_topologies": False,
        "checks_csi_reconstruction": False,
        "changes_solver_result": False,
        "uses_ground_truth": False,
        "position_m": source.tolist(),
        "bs_position_m": receiver.tolist(),
        "beta_m": float(beta_m),
        "bias_s": bias_s,
        "selected_path_count": len(rows),
        "valid_selected_path_count": sum(row["valid"] for row in rows),
        "all_selected_paths_valid": bool(rows and all(row["valid"] for row in rows)),
        "residual_sign": "prediction_minus_observation",
        "paths": rows,
    }


__all__ = ["forward_check_solution"]
