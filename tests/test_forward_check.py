from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from time_bias_localization.candidates import (
    PathObservationSample,
    cluster_reverse_candidates,
    generate_reverse_candidates,
)
from time_bias_localization.constants import SPEED_OF_LIGHT_M_S
from time_bias_localization.forward_check import forward_check_solution
from time_bias_localization.raytrace2d import enumerate_specular_paths
from time_bias_localization.scene import WallSegment, make_synthetic_room


def _reference_paths():
    scene = make_synthetic_room()
    bs = np.array([2.0, 7.0])
    ue = np.array([14.0, 4.0])
    beta = 5.0
    paths = enumerate_specular_paths(scene, ue, bs, max_reflections=2)
    samples = [
        PathObservationSample(
            f"path-{index}", f"sample-{index}", np.deg2rad(path.arrival_aoa_deg),
            path.delay_s + beta / SPEED_OF_LIGHT_M_S,
        )
        for index, path in enumerate(paths)
    ]
    raw = generate_reverse_candidates(
        scene, bs, samples, max_reflections=2, beta_interval_m=(-20.0, 20.0),
    )
    selected = [
        candidate for candidate in cluster_reverse_candidates(raw)
        if candidate.is_valid(beta) and np.linalg.norm(candidate.point(beta) - ue) < 1e-7
    ]
    assert len(selected) == len(paths)
    peaks = {
        sample.observation_id: {"aoa_global_rad": sample.aoa_global_rad,
                                "delay_s": sample.delay_s}
        for sample in samples
    }
    return scene, bs, ue, beta, paths, selected, peaks


def test_forward_check_reconstructs_direct_and_reflected_paths_with_one_common_bias():
    scene, bs, ue, beta, paths, selected, peaks = _reference_paths()
    report = forward_check_solution(scene, bs, selected, ue, beta, observed_peaks=peaks)
    assert report["all_selected_paths_valid"]
    assert report["scope"] == "selected_topologies_2d_specular"
    assert not report["checks_all_scene_path_topologies"]
    assert not report["checks_csi_reconstruction"]
    assert {len(row["reflection_wall_ids"]) for row in report["paths"]} == {0, 1, 2}
    for row in report["paths"]:
        path = paths[int(row["observation_id"].split("-")[1])]
        np.testing.assert_allclose(row["path_nodes_m"], path.nodes, atol=1e-8)
        assert row["prediction"]["length_m"] == pytest.approx(path.length_m)
        assert row["prediction"]["predicted_observed_delay_s"] == pytest.approx(
            path.delay_s + beta / SPEED_OF_LIGHT_M_S
        )
        assert abs(row["sample_residuals"]["aoa_error_deg"]) < 1e-8
        assert abs(row["sample_residuals"]["delay_error_ns"]) < 1e-6
        assert row["sample_residuals"] == row["original_peak_residuals"]
    json.dumps(report, allow_nan=False)


def test_changed_bias_changes_every_predicted_observed_delay_by_same_amount():
    scene, bs, ue, beta, _, selected, peaks = _reference_paths()
    shifted = forward_check_solution(scene, bs, selected, ue, beta + 0.3, observed_peaks=peaks)
    for row in shifted["paths"]:
        assert row["sample_residuals"]["delay_error_ns"] == pytest.approx(
            0.3 / SPEED_OF_LIGHT_M_S * 1e9, abs=1e-6
        )
        assert abs(row["sample_residuals"]["aoa_error_deg"]) < 1e-8


def test_forward_check_recomputes_reflections_instead_of_reusing_candidate_points():
    scene, bs, ue, beta, _, selected, peaks = _reference_paths()
    # 元数据中缓存的旧交点故意损坏；正向重算不能使用它。
    selected = [replace(item, metadata={**item.metadata, "reflection_points_m": [[999.0, 999.0]]})
                for item in selected]
    moved_ue = ue + np.array([0.1, 0.1])
    report = forward_check_solution(scene, bs, selected, moved_ue, beta, observed_peaks=peaks)
    assert report["all_selected_paths_valid"]
    assert any(abs(row["sample_residuals"]["aoa_error_deg"]) > 0.01 for row in report["paths"])
    for row in report["paths"]:
        np.testing.assert_allclose(row["path_nodes_m"][0], moved_ue)
        assert np.max(np.abs(row["path_nodes_m"])) < 999.0


def test_forward_check_detects_occluded_selected_direct_path():
    scene, bs, ue, beta, _, selected, _ = _reference_paths()
    direct = next(item for item in selected if not item.metadata["reflection_wall_ids"])
    blocked = replace(scene, walls=scene.walls + (WallSegment("blocking", (8.0, 1.0), (8.0, 9.0)),))
    report = forward_check_solution(blocked, bs, [direct], ue, beta)
    assert not report["all_selected_paths_valid"]
    assert report["paths"][0]["prediction"] is not None
    assert "path_blocked_by_scene_wall" in report["paths"][0]["failure_reasons"]


def test_forward_check_distinguishes_sample_from_original_peak_residual():
    scene, bs, ue, beta, _, selected, peaks = _reference_paths()
    peaks = {key: {"aoa_global_rad": value["aoa_global_rad"] + 0.02,
                   "delay_s": value["delay_s"] + 2e-9}
             for key, value in peaks.items()}
    report = forward_check_solution(scene, bs, selected, ue, beta, observed_peaks=peaks)
    for row in report["paths"]:
        assert abs(row["sample_residuals"]["delay_error_ns"]) < 1e-6
        assert row["original_peak_residuals"]["delay_error_ns"] == pytest.approx(-2.0)
        assert row["original_peak_residuals"]["aoa_error_rad"] == pytest.approx(-0.02)


def test_missing_topology_does_not_silently_assume_los():
    scene, bs, ue, beta, _, selected, _ = _reference_paths()
    candidate = replace(selected[0], metadata={})
    report = forward_check_solution(scene, bs, [candidate], ue, beta)
    assert not report["all_selected_paths_valid"]
    assert "missing_reflection_topology" in report["paths"][0]["failure_reasons"]
