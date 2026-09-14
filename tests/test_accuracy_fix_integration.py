from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest

from time_bias_localization.config import DEFAULT_CONFIG, localization_config_view
from time_bias_localization.pipeline import prepare_scene, generate_data, localize
from time_bias_localization.timing import collect_timings


def configuration(root, path_count):
    config = deepcopy(DEFAULT_CONFIG)
    config["output"]["root"] = str(root)
    config["music"]["num_paths"] = path_count
    config["music"]["path_detection"] = dict(enabled=True, max_paths=3,
        false_alarm_probability=.05, calibration_trials=127, max_refine_evaluations=45)
    config["music"]["spectrum_sampling"].update(samples_per_peak=16,local_grid_points_per_axis=9)
    config["localization"].update(candidate_bias_mode="full_interval",require_identifiable_solution=True)
    return config


def test_single_real_path_returns_unlocalizable_before_mc_or_rt(tmp_path, monkeypatch):
    import time_bias_localization.pipeline as pipeline
    config = configuration(tmp_path,1)
    scene = prepare_scene(config,tmp_path)
    bundle = generate_data(config,scene_json=scene["scene_json"],output_root=tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("只有一条可靠路径时不应进入 MC 或 RT")
    monkeypatch.setattr(pipeline,"sample_music_spectrum",forbidden)
    with collect_timings() as timing:
        result = localize(localization_config_view(config),scene_json=scene["scene_json"],
                          online_input=bundle["online_npz"],output_root=tmp_path)
    assert result["status"] == "unlocalizable"
    assert result["reason"] == "insufficient_reliable_paths"
    assert result["mu_m"] is None
    assert result["diagnostics"]["path_detection"]["accepted_path_count"] == 1
    progress = json.loads(Path(result["progress_path"]).read_text())
    assert progress["status"] == "unlocalizable" and progress["truth_was_loaded"] is False
    assert "online_unavailable" in timing.marks
    assert "initial_candidates_available" not in timing.marks


def test_accepted_csi_centers_survive_music_mc_and_result_publication(tmp_path):
    config = configuration(tmp_path,3)
    scene = prepare_scene(config,tmp_path)
    bundle = generate_data(config,scene_json=scene["scene_json"],output_root=tmp_path)
    result = localize(localization_config_view(config),scene_json=scene["scene_json"],
                      online_input=bundle["online_npz"],output_root=tmp_path)
    assert result.get("status") == "success", result.get("reason")
    peaks = json.loads((tmp_path/"localization/music_peaks.json").read_text())
    accepted = peaks["path_detection"]["accepted_peaks"]
    assert len(peaks["nominal"]) == len(accepted) == 3
    for nominal, expected in zip(peaks["nominal"],accepted):
        assert nominal["aoa_rad"] == expected["aoa_rad"]
        assert nominal["delay_s"] == expected["delay_s"]
    assert result["diagnostics"]["music_signal_subspace_rank"] == 3
    assert result["diagnostics"]["initial_candidate_generation"]["candidate_generation_mode"] == "full_bias_interval"


@pytest.mark.parametrize("status,reason", [
    ("unlocalizable", "insufficient_independent_physical_constraints"),
    ("detection_incomplete", "path_detection_budget_exhausted"),
    ("solver_budget_exhausted", "solver_pair_budget_exhausted"),
])
def test_unavailable_trial_does_not_reuse_rejected_candidate_coordinate(tmp_path, status, reason):
    from time_bias_localization.boundary_experiment import trial_record
    from time_bias_localization.provenance import file_sha256
    path = tmp_path/"truth.npz"
    np.savez(path,ue_position_m=[1.,2.],clock_bias_s=0.)
    point = dict(cohort="pilot",ue_id="x",noise_seeds=[1],mc_seeds=[2],channel_category="diffraction_only",has_diffraction=True)
    observation = dict(truth_npz=str(path),truth_sha256=file_sha256(path),input_sha256="csi")
    payload = dict(status=status,position_m=None,clock_bias_s=None,reason=reason,
        timings=dict(marks={"csi_map_ready":0.,"position_available":1.,"online_unavailable":2.},
                     mark_data={"position_available":{"mu_m":[100.,200.],"clock_bias_s":1e-9}}))
    row = trial_record(point,0,"coverage",payload,observation,tmp_path)
    assert row["position_error_m"] is None and row["clock_bias_error_ns"] is None
    assert row["localization_seconds"] == 2.
    assert row["stop_reason"] == reason
    assert row["unlocalizable_reason"] == (reason if status == "unlocalizable" else None)


def test_path_budget_stops_before_mc_without_claiming_single_path(tmp_path, monkeypatch):
    import time_bias_localization.pipeline as pipeline
    config = configuration(tmp_path, 3)
    config["music"]["path_detection"]["max_paths"] = 1
    scene = prepare_scene(config, tmp_path)
    bundle = generate_data(config, scene_json=scene["scene_json"], output_root=tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("路径检测预算耗尽时不应进入 MC 或 RT")
    monkeypatch.setattr(pipeline, "sample_music_spectrum", forbidden)
    result = localize(localization_config_view(config), scene_json=scene["scene_json"],
                      online_input=bundle["online_npz"], output_root=tmp_path)
    assert result["status"] == "detection_incomplete"
    assert result["reason"] == "path_detection_budget_exhausted"
    assert result["diagnostics"]["path_detection"]["accepted_path_count"] == 1
    assert result["output_type"] == "no_position_computation_incomplete"
    progress = json.loads(Path(result["progress_path"]).read_text())
    assert progress["status"] == "detection_incomplete"
    assert progress["truth_was_loaded"] is False
    assert result["mu_m"] is None


def test_candidate_pair_budget_has_distinct_online_result(tmp_path):
    config = configuration(tmp_path, 3)
    config["localization"]["max_seed_pairs"] = 1
    scene = prepare_scene(config, tmp_path)
    bundle = generate_data(config, scene_json=scene["scene_json"], output_root=tmp_path)
    result = localize(localization_config_view(config), scene_json=scene["scene_json"],
                      online_input=bundle["online_npz"], output_root=tmp_path)
    assert result["status"] == "solver_budget_exhausted"
    assert result["reason"] == "solver_pair_budget_exhausted"
    assert result["output_type"] == "no_position_computation_incomplete"
    assert result["diagnostics"]["candidate_pair_count"] > 1
    assert result["diagnostics"]["max_seed_pairs"] == 1
    assert result["diagnostics"]["pair_search_started"] is False
    progress = json.loads(Path(result["progress_path"]).read_text())
    assert progress["status"] == "solver_budget_exhausted"
    assert progress["truth_was_loaded"] is False
    assert result["mu_m"] is None
