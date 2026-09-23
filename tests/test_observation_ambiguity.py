"""镜像角歧义组必须按组枚举角度分支，而不是停掉整份观测。"""
import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from time_bias_localization.config import (
    DEFAULT_CONFIG, localization_config_view, validate_config,
)
from time_bias_localization.constants import SPEED_OF_LIGHT_M_S as C
from time_bias_localization.continuous_pipeline import (
    angle_branch_choices, branch_kept_positions, localize_saved_music,
    run_continuous_with_angle_branches,
)
from time_bias_localization.diffraction import enumerate_paths
from time_bias_localization.observation_screen import ambiguity_groups, screen_music_observation
from time_bias_localization.raytrace2d import enumerate_specular_paths
from time_bias_localization.scene import Scene2D, make_synthetic_room
from time_bias_localization.signal import MusicPeak2D


def screen(angles, delays, **settings):
    return screen_music_observation(
        [SimpleNamespace(aoa_rad=math.radians(angle), delay_s=delay)
         for angle, delay in zip(angles, delays)],
        num_antennas=12, antenna_spacing_m=C / 3.5e9 / 2, carrier_frequency_hz=3.5e9,
        frequencies_hz=np.arange(512) * 400e6 / 512,
        settings={"enabled": True, "max_response_correlation": 0.995, **settings})


def pairs(*items):
    return [{"first_peak_index": first, "second_peak_index": second} for first, second in items]


def branch_config():
    config = deepcopy(DEFAULT_CONFIG)
    config["localization"].update(
        solver_method="continuous", bias_min_s=-20e-9, bias_max_s=20e-9,
        continuous={"max_hypotheses": 256, "max_enumerated_sequences": 4096, "max_starts": 32,
                    "max_seed_combinations": 2048, "max_iterations": 40, "aoa_gate_deg": 5.0})
    return config


def test_ambiguity_groups_merge_shared_peaks_and_stay_ordered():
    assert ambiguity_groups([], 3) == []
    assert ambiguity_groups(pairs((0, 1)), 3) == [[0, 1]]
    assert ambiguity_groups(pairs((0, 1), (2, 3)), 4) == [[0, 1], [2, 3]]
    assert ambiguity_groups(pairs((0, 1), (1, 2)), 4) == [[0, 1, 2]]
    assert ambiguity_groups(pairs((3, 4), (0, 1)), 5) == [[0, 1], [3, 4]]


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_ambiguity_groups_reject_bad_peak_counts(count):
    with pytest.raises(ValueError):
        ambiguity_groups([], count)


def test_ambiguity_groups_reject_members_outside_the_peak_list():
    with pytest.raises(ValueError):
        ambiguity_groups(pairs((0, 2)), 2)


def test_branch_choices_and_kept_positions_follow_original_peak_order():
    assert angle_branch_choices([]) == [()]
    assert angle_branch_choices([[0, 1]]) == [(0,), (1,)]
    assert angle_branch_choices([[0, 1], [2, 3]]) == [(0, 2), (0, 3), (1, 2), (1, 3)]
    assert branch_kept_positions(4, [[0, 1]], (1,)) == [1, 2, 3]
    assert branch_kept_positions(4, [[0, 1]], (0,)) == [0, 2, 3]
    assert branch_kept_positions(4, [[0, 1], [2, 3]], (1, 2)) == [1, 2]
    with pytest.raises(ValueError):
        branch_kept_positions(4, [[0, 1]], ())
    with pytest.raises(ValueError):
        branch_kept_positions(4, [[0, 1]], (2,))


def test_screen_publishes_groups_instead_of_only_a_sample_verdict():
    report = screen([-86.5, 89.0, 10.0], [625.125e-9] * 3)
    assert report["excluded"] and report["ambiguity_policy"] == "exclude_sample"
    assert report["ambiguity_groups"] == [[0, 1]] and report["branch_count"] == 2
    spread = report["ambiguity_group_spreads"][0]
    assert spread["peak_indices"] == [0, 1] and spread["delay_spread_ns"] == 0.0
    assert 170 < spread["angle_spread_deg"] <= 180
    assert report["excluded_meaning"] == "at_least_one_pair_triggered_not_sample_verdict"
    clean = screen([0.0, 45.0], [625.125e-9] * 2)
    assert not clean["excluded"] and clean["ambiguity_groups"] == [] and clean["branch_count"] == 1
    disabled = screen([-86.5, 89.0], [625.125e-9] * 2, enabled=False)
    assert disabled["ambiguity_groups"] == [] and disabled["branch_count"] == 1


def test_screen_rejects_unknown_ambiguity_policy():
    with pytest.raises(ValueError):
        screen([0.0, 45.0], [1e-7, 2e-7], ambiguity_policy="unknown")


@pytest.mark.parametrize("policy,method", [("exclude_sample", "exhaustive"),
                                           ("enumerate_branches", "continuous")])
def test_config_accepts_supported_ambiguity_policies(policy, method):
    config = deepcopy(DEFAULT_CONFIG)
    config["music"]["observation_screen"]["ambiguity_policy"] = policy
    config["localization"]["solver_method"] = method
    validate_config(config)


def test_config_rejects_unknown_policy_and_branches_without_continuous_solver():
    config = deepcopy(DEFAULT_CONFIG)
    config["music"]["observation_screen"]["ambiguity_policy"] = "unknown"
    with pytest.raises(ValueError):
        validate_config(config)
    legacy = deepcopy(DEFAULT_CONFIG)
    legacy["music"]["observation_screen"]["ambiguity_policy"] = "enumerate_branches"
    legacy["localization"]["solver_method"] = "exhaustive"
    with pytest.raises(ValueError):
        validate_config(legacy)


def test_mirror_branch_search_keeps_the_explanation_that_fits_the_map():
    scene, bs, ue = make_synthetic_room(), (2.0, 3.0), (12.0, 9.5)
    paths = enumerate_paths(scene, ue, bs, max_reflections=1)
    assert len(paths) >= 2
    peaks = [SimpleNamespace(aoa_rad=math.pi - math.radians(paths[0].arrival_aoa_deg),
                             delay_s=paths[0].length_m / C)]
    peaks += [SimpleNamespace(aoa_rad=math.radians(path.arrival_aoa_deg),
                              delay_s=path.length_m / C) for path in paths[:2]]
    selection = run_continuous_with_angle_branches(branch_config(), scene, peaks, [0, 1, 2],
                                                   [[0, 1]], bs, 0.0)
    report = selection.report
    assert report["branch_count"] == 2 and report["ambiguity_groups"] == [[0, 1]]
    assert report["branches"][0]["kept_peak_positions"] == [0, 2]
    assert report["branches"][1]["kept_peak_positions"] == [1, 2]
    assert report["branches"][1]["objective"] < report["branches"][0]["objective"]
    assert not report["all_branches_failed"] and report["selected_branch"] == 1
    assert selection.result["diagnostics"]["angle_branch_search"]["selected_branch"] == 1
    assert np.allclose(selection.result["mu_m"], ue, atol=1e-3)
    assert report["branches"][1]["kept_observation_ids"] == ["music_path_01", "music_path_02"]


def test_branch_search_reports_failure_when_every_choice_leaves_one_peak():
    scene, bs, ue = make_synthetic_room(), (2.0, 3.0), (12.0, 9.5)
    angle = math.radians(enumerate_paths(scene, ue, bs, max_reflections=1)[0].arrival_aoa_deg)
    length = enumerate_paths(scene, ue, bs, max_reflections=1)[0].length_m
    peaks = [SimpleNamespace(aoa_rad=angle, delay_s=length / C),
             SimpleNamespace(aoa_rad=math.pi - angle, delay_s=length / C)]
    selection = run_continuous_with_angle_branches(branch_config(), scene, peaks, [0, 1],
                                                   [[0, 1]], bs, 0.0)
    assert selection.result is None and selection.payloads == {}
    assert selection.report["all_branches_failed"]
    assert selection.report["failure_reason"] == "insufficient_music_peaks_after_ambiguity_branch"
    assert all(record["skipped_reason"] for record in selection.report["branches"])


def test_branch_search_preserves_failure_without_a_scored_candidate(monkeypatch):
    import time_bias_localization.continuous_pipeline as continuous
    result = {"status": "solver_budget_exhausted", "reason": "search_exhausted",
              "mu_m": None, "diagnostics": {}}
    monkeypatch.setattr(continuous, "run_continuous_from_peaks",
                        lambda *args, **kwargs: (deepcopy(result), {}, None, None))
    selection = run_continuous_with_angle_branches(
        {}, None, [None] * 3, [0, 1, 2], [[0, 1]], (0, 0), 0)
    assert selection.result["status"] == "solver_budget_exhausted"
    assert selection.report["failure_reason"] == "search_exhausted"
    assert selection.report["rejected_branch_count"] == 2
    weighted = run_continuous_with_angle_branches(
        {"localization": {"amplitude_weighting": {"enabled": True}}},
        None, [None] * 3, [0, 1, 2], [[0, 1]], (0, 0), 0)
    assert not weighted.report["branch_objectives_comparable"]


@pytest.mark.parametrize("position,beta,cost,weighted,other_status,expected", [
    ([5., 0.], 0., .1, False, "success", "ambiguous"),
    ([0., 0.], 1., .1, False, "success", "ambiguous"),
    ([.01, 0.], 0., .1, False, "success", "success"),
    ([5., 0.], 0., 10., False, "success", "success"),
    ([5., 0.], 0., 10., True, "success", "ambiguous"),
    ([5., 0.], 0., .1, False, "ambiguous", "ambiguous"),
])
def test_distinct_angle_branch_solutions_keep_ambiguity(
        monkeypatch, position, beta, cost, weighted, other_status, expected):
    import time_bias_localization.continuous_pipeline as continuous
    results = []
    for xy, bias, objective, status in [([0., 0.], 0., 0., "success"),
                                        (position, beta, cost, other_status)]:
        candidate = {"position_m": xy, "beta_m": bias, "objective": objective, "acceptable": True}
        results.append({"status": status, "mu_m": xy if status == "success" else None,
                        "distance_bias_m": bias, "diagnostics": {},
                        "best_candidate_for_diagnostics_only": candidate})
    monkeypatch.setattr(continuous, "run_continuous_from_peaks",
                        lambda *args, **kwargs: (results.pop(0), {}, None, None))
    selection = run_continuous_with_angle_branches(
        {"localization": {"amplitude_weighting": {"enabled": weighted}}},
        None, [None] * 3, [0, 1, 2], [[0, 1]], (0, 0), 0)
    assert selection.result["status"] == expected
    assert selection.report["cross_branch_ambiguity"] == (expected == "ambiguous")
    if expected == "ambiguous":
        assert selection.result["mu_m"] is None and selection.result["distance_bias_m"] is None
        assert selection.result["reason"] == "multiple_angle_branches_fit"
        assert selection.result["alternatives"][0]["angle_branch_index"] == 1


def room_config(root):
    config = deepcopy(DEFAULT_CONFIG)
    config["scene"].update(max_reflections=1, max_diffractions=0)
    config["simulation"]["path_amplitudes"] = [1.0, 0.8, 0.6]
    config["radio"].update(bs_boresight_deg=0.0, snr_db=80.0)
    config["music"]["spectrum_sampling"].update(samples_per_peak=4, local_grid_points_per_axis=9)
    config["music"]["observation_screen"].update(enabled=True,
                                                ambiguity_policy="enumerate_branches")
    config["localization"].update(solver_method="continuous", require_identifiable_solution=True,
        continuous={"max_hypotheses": 256, "max_enumerated_sequences": 4096, "max_starts": 32,
                    "max_seed_combinations": 2048, "max_iterations": 40, "aoa_gate_deg": 5.0})
    config["output"]["root"] = str(root)
    return config


def mirror_peak_rows(paths, count):
    """一个镜像幽灵峰 + count 条真实路径峰；幽灵与第一条真实峰构成歧义对。"""
    rows = [{"aoa_rad": math.pi - math.radians(paths[0].arrival_aoa_deg),
             "delay_s": paths[0].length_m / C}]
    rows += [{"aoa_rad": math.radians(path.arrival_aoa_deg), "delay_s": path.length_m / C}
             for path in paths[:count]]
    return rows


def test_saved_music_with_ambiguity_groups_selects_one_branch(tmp_path, monkeypatch):
    scene = make_synthetic_room()
    bs, ue = (2.0, 7.0), np.array([14.0, 4.0])
    paths = enumerate_specular_paths(scene, ue, bs, max_reflections=1)
    assert len(paths) >= 2
    rows = mirror_peak_rows(paths, 2)
    source = tmp_path / "source"
    source.mkdir()
    scene_path = source / "scene.json"
    scene_path.write_text(json.dumps(scene.to_dict()))
    peaks_path = source / "music_peaks.json"
    peaks_path.write_text(json.dumps({
        "nominal": rows, "nominal_source_indices": list(range(len(rows))),
        "observation_screen": {"excluded": True,
                               "excluded_pairs": [{"first_peak_index": 0,
                                                   "second_peak_index": 1}],
                               "ambiguity_groups": [[0, 1]]}}))
    config = localization_config_view(room_config(tmp_path / "cfg"))
    assert config["music"]["observation_screen"]["ambiguity_policy"] == "enumerate_branches"
    result = localize_saved_music(config, scene_json=scene_path, music_peaks_json=peaks_path,
                                  output_root=tmp_path / "out")
    report = result["diagnostics"]["angle_branch_search"]
    assert report["branch_count"] == 2 and report["selected_branch"] == 1
    assert report["branches"][0]["kept_peak_positions"] == [0, 2]
    assert report["branches"][1]["objective"] < report["branches"][0]["objective"]
    assert result["status"] == "success"
    assert np.allclose(result["mu_m"], ue, atol=1e-3)
    observations = json.loads((tmp_path / "out/localization/continuous_observations.json").read_text())
    assert [row["observation_id"] for row in observations["observations"]] == \
        ["music_path_01", "music_path_02"]
    config["music"]["observation_screen"]["ambiguity_policy"] = "exclude_sample"
    excluded = localize_saved_music(config, scene_json=scene_path, music_peaks_json=peaks_path,
                                    output_root=tmp_path / "screen_excluded")
    assert excluded["status"] == "excluded_observation" and excluded["mu_m"] is None
    # 关闭筛选必须关闭分支，不能仅因旧文件留有筛选记录就继续枚举。
    import time_bias_localization.continuous_pipeline as continuous
    config["music"]["observation_screen"]["ambiguity_policy"] = "enumerate_branches"
    config["music"]["observation_screen"]["enabled"] = False
    monkeypatch.setattr(continuous, "run_continuous_with_angle_branches",
                        lambda *args, **kwargs: pytest.fail("已关闭的筛选仍在枚举角度分支"))
    monkeypatch.setattr(continuous, "run_continuous_from_peaks", lambda *args, **kwargs: (
        {"localization_run_id": "disabled-screen", "status": "unlocalizable"}, {}, None, None))
    localize_saved_music(config, scene_json=scene_path, music_peaks_json=peaks_path,
                         output_root=tmp_path / "screen_disabled")


def test_online_mirror_pair_keeps_only_the_selected_branch_peaks(tmp_path, monkeypatch):
    import time_bias_localization.pipeline as pipeline
    config = room_config(tmp_path)
    scene = pipeline.prepare_scene(config, tmp_path)
    bundle = pipeline.generate_data(config, scene_json=scene["scene_json"], output_root=tmp_path)
    model = Scene2D.from_dict(json.loads(Path(scene["scene_json"]).read_text()))
    paths = enumerate_specular_paths(model, config["simulation"]["ue_position_m"],
                                     config["simulation"]["bs_position_m"], max_reflections=1)
    assert len(paths) >= 2
    rows = mirror_peak_rows(paths, 2)
    peaks = [MusicPeak2D(row["aoa_rad"], row["delay_s"], 1.0 + index, 0, 0)
             for index, row in enumerate(rows)]
    original = pipeline.refine_music_peaks

    def inject(*args, **kwargs):
        return replace(original(*args, **kwargs), refined_peaks=peaks,
                       refined_peak_source_indices=list(range(len(peaks))))

    monkeypatch.setattr(pipeline, "refine_music_peaks", inject)
    result = pipeline.localize(localization_config_view(config), scene_json=scene["scene_json"],
                               online_input=bundle["online_npz"], output_root=tmp_path)
    report = result["diagnostics"]["angle_branch_search"]
    assert report["branch_count"] == 2 and report["selected_branch"] == 1
    assert result["status"] == "success", result
    assert np.linalg.norm(np.asarray(result["mu_m"]) - config["simulation"]["ue_position_m"]) < 1e-3
    saved = json.loads((tmp_path / "localization/music_peaks.json").read_text())
    assert saved["nominal_source_indices"] == [1, 2]
    assert [row["aoa_rad"] for row in saved["nominal"]] == [row["aoa_rad"] for row in rows[1:]]
    assert saved["observation_screen"]["ambiguity_groups"] == [[0, 1]]
    assert len(saved["nominal_before_angle_branches"]) == 3
    assert saved["nominal_source_indices_before_angle_branches"] == [0, 1, 2]
    replay = localize_saved_music(localization_config_view(config),
        scene_json=scene["scene_json"], music_peaks_json=tmp_path / "localization/music_peaks.json",
        source_online_input=bundle["online_npz"],
        source_manifest=tmp_path / "localization/localization_manifest.json",
        output_root=tmp_path / "replay")
    assert replay["status"] == result["status"]
    assert replay["diagnostics"]["angle_branch_search"]["selected_branch"] == 1
    np.testing.assert_allclose(replay["mu_m"], result["mu_m"], atol=1e-6)
    # 旧产物已经丢失未选中的角度，必须拒绝使用选前编号重放选后的列表。
    saved.pop("nominal_before_angle_branches")
    saved.pop("nominal_source_indices_before_angle_branches")
    incomplete_path = tmp_path / "old_incomplete_music.json"
    incomplete_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="缺少分支选择前"):
        localize_saved_music(localization_config_view(config), scene_json=scene["scene_json"],
            music_peaks_json=incomplete_path, output_root=tmp_path / "incomplete_replay")


@pytest.mark.parametrize("status", ["solver_budget_exhausted", "ambiguous", "geometry_failed"])
def test_online_branch_failure_keeps_solver_status(tmp_path, monkeypatch, status):
    import time_bias_localization.pipeline as pipeline
    import time_bias_localization.continuous_pipeline as continuous
    config = room_config(tmp_path)
    scene = pipeline.prepare_scene(config, tmp_path)
    bundle = pipeline.generate_data(config, scene_json=scene["scene_json"], output_root=tmp_path)
    original = pipeline.refine_music_peaks

    def inject(*args, **kwargs):
        peaks = [MusicPeak2D(math.radians(angle), delay, 1., 0, 0)
                 for angle, delay in [(-89., 1e-7), (89., 1e-7), (0., 2e-7)]]
        return replace(original(*args, **kwargs), refined_peaks=peaks,
                       refined_peak_source_indices=[0, 1, 2])

    monkeypatch.setattr(pipeline, "refine_music_peaks", inject)
    monkeypatch.setattr(continuous, "run_continuous_from_peaks", lambda *args, **kwargs: (
        {"workflow": continuous.WORKFLOW, "status": status, "reason": status,
         "mu_m": None, "diagnostics": {}}, {}, None, None))
    result = pipeline.localize(localization_config_view(config), scene_json=scene["scene_json"],
                               online_input=bundle["online_npz"], output_root=tmp_path)
    assert result["status"] == status
    assert result["reason"] == status
    assert result["diagnostics"]["angle_branch_search"]["all_branches_failed"]


def test_amplitude_weighting_scales_noise_by_spectrum_value():
    from time_bias_localization.amplitude_weighting import (
        amplitude_scale_factors, amplitude_weighting_settings,
    )
    settings = {"enabled": True, "exponent": 1.0, "reference": "maximum",
                "maximum_factor": 10.0}
    factors = amplitude_scale_factors([100.0, 50.0, 10.0], settings)
    assert factors[0] == pytest.approx(1.0)
    assert factors[1] == pytest.approx(2.0)
    assert factors[2] == pytest.approx(10.0)
    assert list(amplitude_scale_factors([100.0, 50.0], {"enabled": False})) == [1.0, 1.0]
    capped = amplitude_scale_factors([1000.0, 1.0], {**settings, "maximum_factor": 5.0})
    assert capped[1] == pytest.approx(5.0)
    halved = amplitude_scale_factors([100.0, 25.0], {**settings, "exponent": 0.5})
    assert halved[1] == pytest.approx(2.0)
    for bad in ({"enabled": True, "reference": "mean"}, {"enabled": True, "maximum_factor": 0.5},
                {"enabled": True, "exponent": -1.0}, {"unknown": 1}, {"enabled": 1}):
        with pytest.raises(ValueError):
            amplitude_weighting_settings(bad)
    with pytest.raises(ValueError):
        amplitude_scale_factors([1.0, 0.0], settings)


def test_amplitude_weighting_changes_observation_scales_only_when_enabled(tmp_path):
    scene = make_synthetic_room()
    bs, ue = (2.0, 7.0), np.array([14.0, 4.0])
    paths = enumerate_specular_paths(scene, ue, bs, max_reflections=1)
    peaks = [SimpleNamespace(aoa_rad=math.radians(path.arrival_aoa_deg),
                             delay_s=path.length_m / C, spectrum_value=100.0 / (i + 1))
             for i, path in enumerate(paths[:2])]
    from time_bias_localization.continuous_pipeline import build_continuous_problem
    base = branch_config()
    base["localization"]["amplitude_weighting"] = {"enabled": False}
    off, _ = build_continuous_problem(base, scene, peaks, [0, 1], bs, 0.0)
    on = branch_config()
    on["localization"]["amplitude_weighting"] = {"enabled": True, "exponent": 1.0,
                                                "reference": "maximum",
                                                "maximum_factor": 10.0}
    weighted, _ = build_continuous_problem(on, scene, peaks, [0, 1], bs, 0.0)
    assert off.observations[0].angle_scale_rad == pytest.approx(off.observations[1].angle_scale_rad)
    assert weighted.observations[0].angle_scale_rad < weighted.observations[1].angle_scale_rad
    assert weighted.observations[1].angle_scale_rad == pytest.approx(
        2.0 * weighted.observations[0].angle_scale_rad)
    # 长度尺度按同一倍数缩放，两者比值保持不变。
    ratio_off = off.observations[0].length_scale_m / off.observations[0].angle_scale_rad
    ratio_on = weighted.observations[1].length_scale_m / weighted.observations[1].angle_scale_rad
    assert ratio_off == pytest.approx(ratio_on)
