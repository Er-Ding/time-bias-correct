"""已知几何、伪观测、数量预算和无真值筛选的行为检查。"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from time_bias_localization.constants import SPEED_OF_LIGHT_M_S as C
from time_bias_localization.config import DEFAULT_CONFIG, validate_config, localization_config_view
from time_bias_localization.initial_candidates import InitialCandidatePoint, cluster_initial_candidate_points
from time_bias_localization.observation_screen import screen_music_observation
from time_bias_localization.solver import CandidateTrajectory, SolverConfig, SolverError, RansacSearchError, solve_position_and_bias


def trajectory(obs, angle, point=(3., 4.), beta=7., candidate_id="center", metadata=None):
    direction = np.array([np.cos(angle), np.sin(angle)])
    return CandidateTrajectory(obs, candidate_id, np.asarray(point) + beta * direction, direction,
                               2., 10., metadata=metadata or {})


def test_ransac_recovers_position_and_bias_ignoring_false_observation_and_pair_limit():
    good = [trajectory(str(i), a) for i, a in enumerate([0., .9, 2., 3.4])]
    false = [trajectory("false", .7, point=(130., 90.), candidate_id=f"false{i:04d}") for i in range(300)]
    config = SolverConfig(method="ransac", ransac_max_trials=128, ransac_seed=12, max_seed_pairs=1)
    result = solve_position_and_bias(good + false, config)
    np.testing.assert_allclose(result.mu, [3., 4.], atol=1e-8)
    assert result.beta == pytest.approx(7., abs=1e-8)
    assert set(result.selected_candidates) == {"0", "1", "2", "3"}
    assert result.diagnostics.unused_observations == ("false",)
    assert result.diagnostics.search["draw_count"] == 128
    assert result.diagnostics.search["inlier_observation_count"] == 4
    again = solve_position_and_bias(list(reversed(good + false)), config)
    np.testing.assert_array_equal(again.mu, result.mu)
    assert again.selected_candidate_ids == result.selected_candidate_ids


def test_two_specular_observations_suffice_and_bias_respects_common_interval():
    candidates = [trajectory("a", .1), trajectory("b", 2.3)]
    result = solve_position_and_bias(candidates, SolverConfig(method="ransac", ransac_max_trials=8))
    np.testing.assert_allclose(result.mu, [3., 4.], atol=1e-8)
    assert result.beta == pytest.approx(7.)
    assert all(c.is_valid(result.beta) for c in result.selected_candidates.values())


def diffracted(obs, center):
    angle = np.arctan2(4.-center[1], 3.-center[0])
    return trajectory(obs, angle, metadata={"propagation_interactions": [("diffraction", obs)],
                                          "interaction_points_m": [center]})


def test_diffraction_directions_do_not_manufacture_independent_constraints():
    candidates = [diffracted("a", [-10., 0.]), diffracted("b", [10., 0.])]
    config = SolverConfig(method="ransac", ransac_max_trials=12)
    with pytest.raises(SolverError, match="两个独立距离约束"):
        solve_position_and_bias(candidates, config)
    candidates.append(diffracted("c", [0., 15.]))
    result = solve_position_and_bias(candidates, config)
    np.testing.assert_allclose(result.mu, [3., 4.], atol=1e-8)
    assert result.beta == pytest.approx(7.)
    assert result.diagnostics.search["sample_size_counts"] == {"2": 0, "3": 12}


def test_no_common_bias_is_search_failure_with_finite_budget():
    candidates = [replace(trajectory("a", 0.), beta_min_m=2., beta_max_m=3.),
                  replace(trajectory("b", 1.), beta_min_m=5., beta_max_m=6.)]
    with pytest.raises(RansacSearchError) as error:
        solve_position_and_bias(candidates, SolverConfig(method="ransac", ransac_max_trials=7))
    assert error.value.diagnostics["draw_count"] == 7
    assert error.value.diagnostics["valid_hypothesis_count"] == 0


def arc(i, phi, edge="corner", observation="obs"):
    direction = (np.cos(phi), np.sin(phi))
    return InitialCandidatePoint(observation, f"s{i:03d}", edge, 0., tuple(10*np.asarray(direction)), (), (),
        0., 15./C, 5., (0., 0.), direction, 40.,
        propagation_interactions=(("diffraction", edge),), interaction_points_m=((0., 0.),))


def test_representative_caps_keep_actual_members_and_report_lost_coverage():
    points = [arc(i, i*.1) for i in range(12)] + [arc(i+20, i*.1, edge="other") for i in range(12)]
    options = dict(min_samples=2, beta_interval_m=(-2., 2.), return_diagnostics=True,
                   diffraction_cluster_representative_max=3, diffraction_representative_max=5)
    result = cluster_initial_candidate_points(points, **options)
    assert len(result.representatives) == 5
    assert max(Counter(r.metadata["point_cluster_id"] for r in result.representatives).values()) <= 3
    assert all(r.point in points and r.point in r.members for r in result.representatives)
    assert not any(r.metadata["all_member_valid_bias_intervals_covered"] for r in result.representatives)
    assert sum(m["is_representative"] for m in result.memberships) == 5
    assert result.diagnostics["cluster_count"] == 2
    again = cluster_initial_candidate_points(list(reversed(points)), **options)
    assert [r.point.sample_id for r in again.representatives] == [r.point.sample_id for r in result.representatives]


def test_observation_cap_also_bounds_many_clusters_and_records_dropped_branches():
    points = [arc(i, 0., edge=f"edge{i}") for i in range(7)]
    result = cluster_initial_candidate_points(points, min_samples=1, beta_interval_m=(-2., 2.),
        diffraction_representative_max=3, return_diagnostics=True)
    assert len(result.representatives) == 3
    assert result.diagnostics["cluster_count"] == 7
    budget = result.diagnostics["diffraction_representative_budget"]["observations"][0]
    assert len(budget["dropped_cluster_ids"]) == 4
    assert budget["before_count"] == 7 and budget["after_count"] == 3


def test_cap_does_not_remove_specular_candidates_or_combine_observation_budgets():
    points = [arc(i, i*.1) for i in range(10)]
    points += [replace(p, observation_id="second") for p in points]
    points += [replace(arc(40, 0.), observation_id="specular", propagation_interactions=(), interaction_points_m=())]
    result = cluster_initial_candidate_points(points, min_samples=1, beta_interval_m=(-2., 2.),
        diffraction_representative_max=2, return_diagnostics=True)
    assert Counter(r.point.observation_id for r in result.representatives) == {"obs":2, "second":2, "specular":1}


def screen(angles, delays, **settings):
    return screen_music_observation([SimpleNamespace(aoa_rad=np.deg2rad(a), delay_s=t) for a,t in zip(angles,delays)],
        num_antennas=12, antenna_spacing_m=C/3.5e9/2, carrier_frequency_hz=3.5e9,
        frequencies_hz=np.arange(512)*400e6/512,
        settings={"enabled":True,"max_response_correlation":.995, **settings})


def test_screen_uses_both_angle_and_delay_without_position_or_noise_truth():
    assert screen([-86.5,89.], [625.125e-9]*2)["excluded"]
    assert not screen([-86.5,89.], [625.125e-9, 645.125e-9])["excluded"]
    assert not screen([0.,45.], [625.125e-9]*2)["excluded"]
    assert not screen([-86.5,89.], [625.125e-9]*2, enabled=False)["excluded"]


@pytest.mark.parametrize("key,value", [("diffraction_representative_max",0), ("diffraction_cluster_representative_max",True),
    ("solver_method","unknown"), ("ransac",{"max_trials":0})])
def test_invalid_public_budgets_rejected(key,value):
    config=deepcopy(DEFAULT_CONFIG)
    config["localization"][key]=value
    with pytest.raises(ValueError):
        validate_config(config)


def test_screen_exclusion_is_published_before_reverse_rt_without_truth(tmp_path, monkeypatch):
    import time_bias_localization.pipeline as pipeline
    config=deepcopy(DEFAULT_CONFIG)
    config["music"]["observation_screen"]["enabled"]=True
    config["music"]["spectrum_sampling"].update(samples_per_peak=4, local_grid_points_per_axis=9)
    config["localization"].update(candidate_bias_mode="full_interval", require_identifiable_solution=True)
    config["output"]["root"]=str(tmp_path)
    scene=pipeline.prepare_scene(config,tmp_path)
    bundle=pipeline.generate_data(config,scene_json=scene["scene_json"],output_root=tmp_path)
    Path(bundle["truth_npz"]).unlink()
    Path(bundle["truth_json"]).unlink()
    original=pipeline.sample_music_spectrum
    def alias(*args,**kwargs):
        result=original(*args,**kwargs)
        peaks=list(result.refined_peaks)
        assert len(peaks)>=2
        peaks[0]=replace(peaks[0],aoa_rad=np.deg2rad(-86.5),delay_s=625.125e-9)
        peaks[1]=replace(peaks[1],aoa_rad=np.deg2rad(89.),delay_s=625.125e-9)
        return replace(result,refined_peaks=peaks)
    monkeypatch.setattr(pipeline,"sample_music_spectrum",alias)
    import time_bias_localization.bias_interval_candidates as generation
    monkeypatch.setattr(generation,"generate_bias_interval_points",lambda *a,**kw:pytest.fail("排除后不应进入 RT"))
    result=pipeline.localize(localization_config_view(config),scene_json=scene["scene_json"],
        online_input=bundle["online_npz"],output_root=tmp_path)
    assert result["status"] == "excluded_observation"
    assert result["mu_m"] is None
    assert result["diagnostics"]["observation_screen"]["excluded_pairs"]
    assert json.loads(Path(result["progress_path"]).read_text())["truth_was_loaded"] is False
