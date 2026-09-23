"""直接从观测与地图进入连续模型；不读取候选点、代表、真值或旧解。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4
import json
import math
import time

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S as C
from .continuous_config import continuous_settings, HYPOTHESIS_SETTING_KEYS, OBSERVATION_SETTING_KEYS
from .config import validate_localization_config
from .provenance import artifact_record, capture_file, exclusive_output_root_lock, localization_config_snapshot
from .scene import Scene2D
from .timing import mark, stage


WORKFLOW = "music_continuous_propagation_v1"


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_continuous_problem(config, scene, nominal_peaks, source_indices, bs_position_m, bs_boresight_rad):
    from .propagation_model import ContinuousObservation
    from .propagation_hypotheses import build_hypothesis_bank
    from .continuous_solver import ContinuousSolverConfig

    settings = continuous_settings(config["localization"].get("continuous", {}))
    if len(nominal_peaks) != len(source_indices) or len(set(source_indices)) != len(source_indices):
        raise ValueError("正式峰与来源编号必须一一对应且编号不重复")
    from .amplitude_weighting import amplitude_scale_factors, amplitude_weighting_settings
    weighting = amplitude_weighting_settings(config["localization"].get("amplitude_weighting"))
    amplitudes = [getattr(p, "spectrum_value", None) for p in nominal_peaks]
    if weighting["enabled"] and any(value is None for value in amplitudes):
        raise ValueError("启用 amplitude_weighting 时每条观测必须带有 MUSIC 谱值；"
                         "冻结观测缺少 spectrum_value")
    factors = amplitude_scale_factors(
        [1.0 if value is None else float(value) for value in amplitudes], weighting)
    base_angle_rad, base_length_m = math.radians(settings["angle_scale_deg"]), settings["length_scale_m"]
    observations = tuple(ContinuousObservation(
        observation_id=f"music_path_{int(i):02d}",
        aoa_rad=(float(p.aoa_rad) + float(bs_boresight_rad) + np.pi) % (2*np.pi) - np.pi,
        observed_length_m=C * float(p.delay_s),
        angle_scale_rad=base_angle_rad * float(factor),
        length_scale_m=base_length_m * float(factor),
    ) for i, p, factor in zip(source_indices, nominal_peaks, factors, strict=True))
    loc = config["localization"]
    beta_bounds = (C * float(loc["bias_min_s"]), C * float(loc["bias_max_s"]))
    with stage("C01_continuous_functions"):
        bank = build_hypothesis_bank(scene, bs_position_m, observations,
            max_reflections=int(config["scene"]["max_reflections"]),
            max_diffractions=int(config["scene"].get("max_diffractions", 0)),
            diffraction_position=config["scene"].get("diffraction_position", "any"),
            max_hypotheses=settings["max_hypotheses"],
            max_enumerated_sequences=settings["max_enumerated_sequences"],
            aoa_gate_rad=math.radians(settings["aoa_gate_deg"]),
            beta_interval_m=beta_bounds, length_gate_sigma=settings["length_gate_sigma"])
    xmin, xmax, ymin, ymax = scene.bounds_m
    options = {k: v for k, v in settings.items()
               if k not in HYPOTHESIS_SETTING_KEYS | OBSERVATION_SETTING_KEYS}
    solver_config = ContinuousSolverConfig(
        xy_bounds_m=((xmin, xmax), (ymin, ymax)), bias_bounds_m=beta_bounds,
        speed_of_light_mps=C, seed=int(config["project"]["random_seed"]), **options)
    return bank, solver_config


def run_continuous_from_peaks(config, scene, nominal_peaks, source_indices,
                              bs_position_m, bs_boresight_rad, *, run_id=None,
                              artifact_callback=None):
    """CSI 主流程与冻结观测对照共用入口；没有旧方法输出参数。"""
    from .continuous_solver import solve_continuous_position_and_bias

    began = time.perf_counter()
    bank, solver_config = build_continuous_problem(config, scene, nominal_peaks, source_indices,
                                                  bs_position_m, bs_boresight_rad)
    payloads = {
        "continuous_observations": {
            "workflow": WORKFLOW, "observations": [asdict(o) for o in bank.observations],
            "source": "original_refined_music_peaks", "uses_ground_truth": False,
            "uncertainty_scales_calibrated": False, "generated_position_samples": 0,
        },
        "propagation_hypotheses": {
            "hypotheses": [asdict(h) for h in bank.hypotheses],
            "observation_hypothesis_indices": bank.observation_hypothesis_indices,
            "search_report": bank.search_report,
            "source": "map_and_observations_directly", "interaction_order": "UE_to_BS",
        },
    }
    if artifact_callback is not None:
        artifact_callback(_json_safe(payloads))
    mark("continuous_functions_available", count=len(bank.hypotheses), observations=len(bank.observations))
    with stage("C02_continuous_optimization"):
        solution = solve_continuous_position_and_bias(bank, solver_config)
    raw = solution.to_dict()
    payloads["continuous_search"] = raw
    if artifact_callback is not None:
        artifact_callback(_json_safe({"continuous_search": raw}))
    status_map = {"insufficient_constraints": "unlocalizable", "search_exhausted": "solver_budget_exhausted",
                  "no_valid_branches": "unlocalizable", "geometry_failed": "geometry_failed"}
    status = status_map.get(raw["status"], raw["status"])
    best = raw.get("best_candidate") or {}
    # 求解器失败或多解时，只把候选保留在诊断，不能当成位置输出。
    position = raw.get("position_m") if status == "success" else None
    beta = raw.get("beta_m") if status == "success" else None
    selected = raw.get("selected_paths", [])
    diagnostics = dict(raw.get("diagnostics", {}))
    diagnostics.update({
        "effective_settings": continuous_settings(config["localization"].get("continuous", {})),
        "continuous_solver_status": raw["status"], "hypothesis_search": bank.search_report,
        "nominal_music_peak_count": len(nominal_peaks), "hypothesis_count": len(bank.hypotheses),
        "spectrum_sample_count": 0, "spatial_clustering_performed": False,
        "representative_selection_performed": False, "legacy_ransac_used": False,
        "covariance_calibrated": False, "truth_was_loaded": False,
        "acceptance_scope": "selected_physical_paths_and_original_observations; global_uniqueness_not_proven",
    })
    forward = _forward_check(bank, solver_config, position, beta, selected)
    if status == "success" and not forward["all_selected_paths_valid"]:
        status, position, beta = "geometry_failed", None, None
    covariance = diagnostics.get("joint_covariance", best.get("covariance"))
    if isinstance(covariance, dict):
        covariance = covariance.get("matrix_xy_beta_m2")
    sigma = np.asarray(covariance)[:2, :2] if covariance is not None and np.asarray(covariance).shape == (3, 3) else None
    result = {
        "schema_version": 5, "workflow": WORKFLOW, "localization_run_id": run_id or str(uuid4()),
        "status": status, "reason": ("selected_path_validation_failed" if status == "geometry_failed"
                                      else raw["status"] if status != "success" else None),
        "output_type": "continuous_propagation_estimate" if position is not None else "no_unique_position",
        "mu_m": position, "distance_bias_m": beta, "clock_bias_s": None if beta is None else beta/C,
        "sigma_m2": sigma if position is not None else None,
        "central_solution": {"mu_m": position, "distance_bias_m": beta,
                             "clock_bias_s": None if beta is None else beta/C, "sigma_m2": sigma},
        "selected_paths": selected, "alternatives": raw.get("alternatives", []),
        "best_candidate_for_diagnostics_only": raw.get("best_candidate"),
        "diagnostics": diagnostics, "forward_check": forward,
        "scientific_validation_status": "not_validated",
    }
    payloads["forward_check"] = forward
    if artifact_callback is not None:
        artifact_callback(_json_safe({"forward_check": forward}))
    diagnostics["continuous_elapsed_s"] = time.perf_counter() - began
    if position is not None:
        mark("position_available", mu_m=position, clock_bias_s=beta/C)
    mark("checked_complete", status=status)
    return _json_safe(result), _json_safe(payloads), bank, solver_config


MIN_ANGLE_BRANCH_OBSERVATIONS = 2
ANGLE_BRANCH_SELECTION_RULE = (
    "lowest_objective_position_branch_then_cross_branch_ambiguity_check")


@dataclass(frozen=True)
class AngleBranchSelection:
    result: dict[str, Any] | None
    payloads: dict[str, Any]
    report: dict[str, Any]
    bank: Any = None
    solver_config: Any = None


def angle_branch_choices(groups):
    """每个歧义组选一个成员的全部组合；无组时返回唯一的空组合。"""
    return [tuple(choice) for choice in product(*groups)] if groups else [()]


def branch_kept_positions(peak_count, groups, choice):
    """按原始峰序升序给出该分支保留的峰位置；组内未被选中的峰全部丢弃。"""
    if len(choice) != len(groups):
        raise ValueError("角度分支必须为每个歧义组给出一个选择")
    for members, chosen in zip(groups, choice):
        if chosen not in members:
            raise ValueError("角度分支只能选择该歧义组内的峰")
    dropped = set()
    for members, chosen in zip(groups, choice):
        dropped.update(index for index in members if index != chosen)
    return [index for index in range(peak_count) if index not in dropped]


def run_continuous_with_angle_branches(config, scene, peaks, source_indices, groups,
                                       bs_position_m, bs_boresight_rad, *, run_id=None,
                                       artifact_callback=None):
    """枚举镜像角度分支，比较有效解并保留跨分支多解状态。

    每个歧义组是一份角度不可分辨的测量，只保留一个成员参与求解。各分支的
    观测条数相同；启用幅值加权时，各分支的尺度可能不同，代价比较仅作启发式。
    不使用真值，不采样 UE 位置；只有代表分支的产物会被发布，其余分支保留摘要。
    """
    if len(peaks) != len(source_indices):
        raise ValueError("正式峰与来源编号必须一一对应")
    choices = angle_branch_choices(groups)
    branches, outcomes = [], {}
    for branch_index, choice in enumerate(choices):
        kept = branch_kept_positions(len(peaks), groups, choice)
        record = {"branch_index": branch_index, "choice_peak_positions": list(choice),
                  "kept_peak_positions": kept,
                  "kept_observation_ids": [f"music_path_{int(source_indices[index]):02d}"
                                           for index in kept],
                  "observation_count": len(kept),
                  "status": None, "mu_m": None, "objective": None,
                  "matched_observation_count": None, "physical_rank": None,
                  "skipped_reason": None}
        if len(kept) < MIN_ANGLE_BRANCH_OBSERVATIONS:
            record["skipped_reason"] = "insufficient_music_peaks_after_ambiguity_branch"
            branches.append(record)
            continue
        result, payloads, bank, solver_config = run_continuous_from_peaks(
            config, scene, [peaks[index] for index in kept],
            [source_indices[index] for index in kept],
            bs_position_m, bs_boresight_rad, run_id=run_id)
        best = result.get("best_candidate_for_diagnostics_only") or {}
        record.update(status=result["status"], mu_m=result["mu_m"],
                      objective=best.get("objective"),
                      matched_observation_count=best.get("matched_observation_count"),
                      physical_rank=best.get("physical_rank"))
        branches.append(record)
        outcomes[branch_index] = AngleBranchSelection(result, payloads, {}, bank, solver_config)
    located = [record for record in branches if record["mu_m"] is not None]
    scored = [record for record in branches if record["objective"] is not None]
    pool = located or scored or [record for record in branches if record["status"] is not None]
    selected = (min(pool, key=lambda record: (
        math.inf if record["objective"] is None else record["objective"], record["branch_index"]))
                if pool else None)
    if located:
        failure_reason = None
    elif selected is not None:
        failure_reason = (selected["skipped_reason"]
                          or outcomes[selected["branch_index"]].result.get("reason"))
    else:
        failure_reason = branches[0]["skipped_reason"] if branches else "no_angle_branch"
    weighted = (config.get("localization", {}).get("amplitude_weighting") or {}).get("enabled", False)
    report = {"policy": "enumerate_branches", "branch_count": len(branches),
              "ambiguity_group_count": len(groups),
              "ambiguity_groups": [list(members) for members in groups],
              "observation_count": branches[0]["observation_count"] if branches else 0,
              "selection_rule": ANGLE_BRANCH_SELECTION_RULE, "branches": branches,
              "selected_branch": None if selected is None else selected["branch_index"],
              "all_branches_failed": not located, "failure_reason": failure_reason,
              "rejected_branch_count": sum(record["status"] is not None
                                           for record in branches) - len(located),
              "uses_truth": False, "uses_position_sampling": False,
              "branch_objectives_comparable": not weighted,
              "objective_comparability": ("branch_dependent_amplitude_scales_heuristic_only" if weighted else
                                          "same_observation_count_and_fixed_residual_scales")}
    if selected is None:
        return AngleBranchSelection(None, {}, report)
    selection = outcomes[selected["branch_index"]]
    competitors = []
    if located:
        settings = continuous_settings(config.get("localization", {}).get("continuous", {}))
        for index, outcome in outcomes.items():
            if index == selected["branch_index"] or outcome.result["status"] not in {"success", "ambiguous"}:
                continue
            candidates = [outcome.result.get("best_candidate_for_diagnostics_only"),
                          *outcome.result.get("alternatives", [])]
            for candidate in candidates:
                if not candidate or not candidate.get("acceptable", False):
                    continue
                distinct = (np.linalg.norm(np.subtract(candidate["position_m"], selection.result["mu_m"]))
                            >= settings["distinct_position_m"] or
                            abs(candidate["beta_m"] - selection.result["distance_bias_m"])
                            >= settings["distinct_bias_m"])
                # 复用求解器已有门限；加权尺度不可比时，不按代价淘汰另一个有效位置。
                if distinct and (weighted or candidate["objective"] <= selected["objective"]
                                  + settings["ambiguity_cost_tolerance"]):
                    competitors.append({**candidate, "angle_branch_index": index})
        if competitors:
            selection.result.update(status="ambiguous", reason="multiple_angle_branches_fit",
                output_type="no_unique_position", mu_m=None, distance_bias_m=None,
                clock_bias_s=None, sigma_m2=None, selected_paths=[])
            selection.result["central_solution"] = dict.fromkeys(
                ("mu_m", "distance_bias_m", "clock_bias_s", "sigma_m2"))
            selection.result["alternatives"] = (competitors + selection.result.get("alternatives", []))[
                :settings["max_alternatives"]]
            for forward in (selection.result.get("forward_check"), selection.payloads.get("forward_check")):
                if forward is not None:
                    forward["diagnostic_candidate_only"] = True
            report["failure_reason"] = "multiple_angle_branches_fit"
    report.update(cross_branch_competitor_count=len(competitors),
                  cross_branch_ambiguity=bool(competitors),
                  cross_branch_rule=("all_distinct_valid_solutions_when_scales_differ" if weighted else
                                     "existing_solver_cost_position_and_bias_tolerances"))
    selection.result["diagnostics"]["angle_branch_search"] = report
    mark("angle_branch_selected", branch=selected["branch_index"],
         branches=len(branches), status=selection.result["status"])
    if artifact_callback is not None:
        artifact_callback(_json_safe(selection.payloads))
    return AngleBranchSelection(selection.result, selection.payloads, report,
                                selection.bank, selection.solver_config)


def _forward_check(bank, solver_config, position, beta, selected):
    """独立重建所选传播顺序，检查每条原始观测；不使用代表专属区间。"""
    from .propagation_model import evaluate_hypothesis

    rows = []
    obs_lookup = {o.observation_id: o for o in bank.observations}
    h_lookup = {h.hypothesis_id: h for h in bank.hypotheses}
    selections = list(selected.values()) if isinstance(selected, dict) else selected
    seen, seen_observations = set(), set()
    if position is not None and beta is not None:
        for item in selections:
            oid, hid = item["observation_id"], item["hypothesis_id"]
            observation, hypothesis = obs_lookup[oid], h_lookup[hid]
            ev = evaluate_hypothesis(bank.scene, hypothesis, position, check_validity=True)
            angle = (ev.aoa_rad-observation.aoa_rad+np.pi) % (2*np.pi)-np.pi
            distance = ev.length_m+beta-observation.observed_length_m
            norm = float(np.hypot(angle/observation.angle_scale_rad, distance/observation.length_scale_m))
            reasons = []
            if not ev.valid:
                reasons.append(ev.invalid_reason or "invalid_geometry")
            if not math.isfinite(norm) or norm > solver_config.max_residual_norm:
                reasons.append("original_observation_residual_exceeds_limit")
            if hid in seen:
                reasons.append("same_physical_path_used_twice")
            if oid in seen_observations:
                reasons.append("same_observation_used_twice")
            seen.add(hid)
            seen_observations.add(oid)
            rows.append({"observation_id": oid, "hypothesis_id": hid, "valid": not reasons,
                "failure_reasons": reasons, "propagation_interactions": hypothesis.interactions,
                "path_nodes_m": None if ev.path is None else ev.path.nodes,
                "prediction": {"aoa_global_rad": ev.aoa_rad, "length_m": ev.length_m,
                               "predicted_observed_delay_s": (ev.length_m+beta)/C},
                "original_peak_residuals": {"aoa_error_rad": angle, "aoa_error_deg": math.degrees(angle),
                    "delay_error_s": distance/C, "delay_error_ns": distance/C*1e9,
                    "length_error_m": distance, "standardized_norm": norm}})
    return _json_safe({"scope": "selected_continuous_paths_and_original_music_observations",
        "uses_ground_truth": False, "checks_all_scene_path_topologies": False,
        "checks_csi_reconstruction": False, "changes_solver_result": True,
        "position_m": position, "beta_m": beta, "paths": rows,
        "selected_path_count": len(rows), "valid_selected_path_count": sum(r["valid"] for r in rows),
        "all_selected_paths_valid": bool(rows) and all(r["valid"] for r in rows)})


def localize_saved_music(config, *, scene_json, music_peaks_json, output_root,
                         source_online_input=None, source_manifest=None, return_problem=False):
    """冻结观测对照；校验来源后调用同一个连续求解入口。禁止覆盖已有运行。"""
    from .data import build_subcarrier_frequencies, load_online_measurement_bytes
    from .observation_screen import screen_music_observation
    from .visualization import write_json

    validate_localization_config(config)
    if config["localization"].get("solver_method") != "continuous":
        raise ValueError("冻结观测入口仅供连续模型使用")
    root = Path(output_root).resolve()
    source_paths = [Path(scene_json).resolve(), Path(music_peaks_json).resolve()]
    if source_online_input is not None:
        source_paths.append(Path(source_online_input).resolve())
    if source_manifest is not None:
        source_paths.append(Path(source_manifest).resolve())
    if any(p == root or root in p.parents for p in source_paths):
        raise ValueError("连续模型输出目录不能包含只读输入")
    with exclusive_output_root_lock(root):
        output = root / "localization"
        if output.exists():
            raise FileExistsError(f"拒绝覆盖已有定位目录：{output}")
        captures = {str(p): capture_file(p) for p in source_paths}
        scene_capture = captures[str(Path(scene_json).resolve())]
        peaks_capture = captures[str(Path(music_peaks_json).resolve())]
        scene = Scene2D.from_dict(json.loads(scene_capture.data))
        document = json.loads(peaks_capture.data)
        if source_manifest is not None:
            manifest = json.loads(captures[str(Path(source_manifest).resolve())].data)
            record = manifest.get("artifacts", {}).get("music_peaks")
            if not record or record["sha256"] != peaks_capture.sha256:
                raise ValueError("冻结 MUSIC 文件与来源清单哈希不一致")
            scene_record = manifest.get("inputs", {}).get("scene", manifest.get("inputs", {}).get("scene_json"))
            if not scene_record or scene_record["sha256"] != scene_capture.sha256:
                raise ValueError("冻结地图与 MUSIC 来源清单不一致")
            if source_online_input is not None:
                online_capture = captures[str(Path(source_online_input).resolve())]
                online_record = manifest.get("inputs", {}).get("online_measurement")
                if not online_record or online_record["sha256"] != online_capture.sha256:
                    raise ValueError("冻结 CSI 与 MUSIC 来源清单不一致")
        radio = config["radio"]
        bs = np.asarray(radio["bs_position_m"], dtype=float)
        boresight = math.radians(float(radio["bs_boresight_deg"]))
        if not np.allclose(scene.bounds_m, config["scene"]["bounds_m"], rtol=0, atol=1e-8):
            raise ValueError("冻结地图范围与定位配置不一致")
        measurement = None
        if source_online_input is not None:
            c = captures[str(Path(source_online_input).resolve())]
            measurement = load_online_measurement_bytes(c.data, source_path=c.path)
            if not np.allclose(bs, measurement.bs_position_m, rtol=0, atol=1e-8):
                raise ValueError("冻结 CSI 的 BS 位置与配置不一致")
            if abs((boresight-measurement.bs_boresight_rad+np.pi)%(2*np.pi)-np.pi) > 1e-8:
                raise ValueError("冻结 CSI 的阵列朝向与配置不一致")
        # 在线产物的 nominal 是最终选中的峰；原筛选编号只对选择前的完整列表有效。
        before_branches = "nominal_before_angle_branches" in document
        peak_rows = document.get("nominal_before_angle_branches", document.get("nominal", []))
        index_key = ("nominal_source_indices_before_angle_branches" if before_branches
                     else "nominal_source_indices")
        indices = document.get(index_key, list(range(len(peak_rows))))
        if len(indices) != len(peak_rows):
            raise ValueError("冻结正式峰必须带有对应来源编号")
        screen_settings = config["music"].get("observation_screen", {})
        screening = document.get("observation_screen") or {}
        if (screen_settings.get("enabled", False) and screening.get("excluded_pairs")
                and screening.get("peak_count", len(peak_rows)) != len(peak_rows)):
            raise ValueError("冻结峰数量与原观测筛选编号不一致；缺少分支选择前的完整峰，不能重新枚举")
        peaks = [SimpleNamespace(aoa_rad=float(p["aoa_rad"]), delay_s=float(p["delay_s"]),
                                 spectrum_value=(None if p.get("spectrum_value") is None
                                                 else float(p["spectrum_value"])))
                 for p in peak_rows]
        screening = screen_music_observation(peaks, settings=screen_settings,
            num_antennas=(measurement.csi_observed.shape[-2] if measurement is not None
                          else int(radio["num_bs_antennas"])),
            carrier_frequency_hz=(measurement.carrier_frequency_hz if measurement is not None
                                  else float(radio["carrier_hz"])),
            antenna_spacing_m=(measurement.antenna_spacing_m if measurement is not None
                               else C / float(radio["carrier_hz"]) * float(radio["antenna_spacing_wavelength"])),
            frequencies_hz=(measurement.subcarrier_frequencies_hz if measurement is not None
                            else build_subcarrier_frequencies(bandwidth_hz=float(radio["bandwidth_hz"]),
                                                              num_subcarriers=int(radio["num_subcarriers"]))))
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "localization_config.json", localization_config_snapshot(config))

        def publish_available(payloads):
            for name, payload in payloads.items():
                write_json(output / f"{name}.json", payload)

        policy = str(screen_settings.get("ambiguity_policy", "exclude_sample"))
        groups = screening["ambiguity_groups"] if policy == "enumerate_branches" else []
        if screening["excluded"] and policy == "exclude_sample":
            result = {"schema_version": 5, "workflow": WORKFLOW,
                      "localization_run_id": str(uuid4()), "status": "excluded_observation",
                      "reason": "near_identical_music_responses",
                      "output_type": "excluded_by_observation_screen", "mu_m": None,
                      "distance_bias_m": None, "clock_bias_s": None, "sigma_m2": None,
                      "diagnostics": {"truth_was_loaded": False},
                      "scientific_validation_status": "not_validated"}
            bank = solver_config = None
        elif groups:
            selection = run_continuous_with_angle_branches(
                config, scene, peaks, indices, groups, bs, boresight,
                artifact_callback=publish_available)
            result, bank, solver_config = selection.result, selection.bank, selection.solver_config
            if result is None:
                result = {"schema_version": 5, "workflow": WORKFLOW,
                          "localization_run_id": str(uuid4()), "status": "unlocalizable",
                          "reason": selection.report["failure_reason"],
                          "output_type": "no_unique_position", "mu_m": None,
                          "distance_bias_m": None, "clock_bias_s": None, "sigma_m2": None,
                          "diagnostics": {"angle_branch_search": selection.report,
                                          "truth_was_loaded": False},
                          "scientific_validation_status": "not_validated"}
        else:
            result, _, bank, solver_config = run_continuous_from_peaks(
                config, scene, peaks, indices, bs, boresight, artifact_callback=publish_available)
        result.setdefault("diagnostics", {})["observation_screen"] = screening
        write_json(output / "localization_result.json", result)
        frozen_manifest = {"schema_version": 1, "workflow": WORKFLOW,
            "localization_run_id": result["localization_run_id"], "truth_was_loaded": False,
            "source_binding_verified": source_manifest is not None,
            "inputs": {str(p): {"path": str(p), "sha256": captures[str(p)].sha256} for p in source_paths},
            "artifacts": {p.stem: artifact_record(p) for p in sorted(output.glob("*.json"))},
            "scientific_validation_status": "not_validated"}
        write_json(output / "frozen_input_manifest.json", frozen_manifest)
    return (result, bank, solver_config) if return_problem else result
