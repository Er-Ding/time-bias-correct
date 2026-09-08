"""绘图只忽略 Sionna 自动对象名称；真实几何差异继续拒绝。"""

from copy import deepcopy
from pathlib import Path

import pytest

from time_bias_localization import step_visualization, visualization
from time_bias_localization.provenance import artifact_record


def _scene():
    return {
        "bounds_m": [-100., 160., -80., 180.],
        "fixed_height_m": 1.5,
        "source": "sionna_exported_triangle_mesh",
        "bev_resolution_m": .2,
        "walls": [
            {"wall_id": "sionna_0009_016094",
             "start_m": [163.75474548339844, -7.0246992111206055],
             "end_m": [133.072225365476, 4.286075073661673],
             "source_object": "no-name-3"},
            {"wall_id": "sionna_0010_026953",
             "start_m": [-64.17579650878906, -46.73434829711914],
             "end_m": [-73.69845256587539, -73.66147245402058],
             "source_object": "no-name-4"},
        ],
    }


def test_equivalence_ignores_only_source_names_and_existing_pixel_setting():
    first = _scene()
    original = deepcopy(first)
    second = deepcopy(first)
    second["walls"][0]["source_object"] = "no-name-7"
    second["walls"][1]["source_object"] = "no-name-8"
    second["bev_resolution_m"] = 1.
    assert visualization._same_scene_geometry(first, second)
    assert first == original
    assert second["walls"][0]["wall_id"] == first["walls"][0]["wall_id"]


@pytest.mark.parametrize("difference", [
    "coordinate", "wall_id", "missing_wall", "extra_wall", "bounds", "height",
    "source", "wall_property",
])
def test_equivalence_rejects_geometry_identity_or_physical_property_changes(difference):
    first = _scene()
    second = deepcopy(first)
    if difference == "coordinate":
        # 不因默认 allclose 的相对容差而放过细小坐标改变。
        second["walls"][0]["start_m"][0] += 1e-10
    elif difference == "wall_id":
        second["walls"][0]["wall_id"] = "different_wall"
    elif difference == "missing_wall":
        second["walls"].pop()
    elif difference == "extra_wall":
        second["walls"].append(deepcopy(first["walls"][0]))
    elif difference == "bounds":
        second["bounds_m"][0] += .01
    elif difference == "height":
        second["fixed_height_m"] += .001
    elif difference == "source":
        second["source"] = "another_scene_source"
    elif difference == "wall_property":
        second["walls"][0]["reflection_coefficient"] = .3
    assert not visualization._same_scene_geometry(first, second)


class _SummaryReached(Exception):
    """停止在已验证结果汇总的边界，避免此测试额外渲染几十张图。"""


def test_report_accepts_worker_failed_and_keeps_it_in_total_denominator(tmp_path, monkeypatch):
    scene_path = tmp_path / "scene.json"
    visualization.write_json(scene_path, _scene())
    experiment = tmp_path / "experiment"
    visualization.write_json(experiment / "experiment_plan.json", {
        "points": [{"ue_id": "UE001", "position_m": [40., 50.]}],
        "noise_repeats": 3, "scene": artifact_record(scene_path),
        "bs_position_m": [20., 50.], "bs_boresight_rad": -.5,
        "clock_bias_s": 25e-9,
    })
    visualization.write_json(experiment / "UE001/repeat_000/attempt.json", {
        "status": "worker_failed", "error": "worker exited unexpectedly",
    })
    visualization.write_json(experiment / "UE001/repeat_001/attempt.json", {
        "status": "success", "run_id": "test-run",
    })
    renamed_scene = _scene()
    renamed_scene["walls"][0]["source_object"] = "no-name-7"
    run = {
        "true": [40., 50.], "bs": [20., 50.], "boresight": -.5,
        "scene": renamed_scene, "sources": [],
        "result": {"localization_run_id": "test-run", "mu_m": [40.5, 50.],
                   "clock_bias_s": 26e-9},
        "metrics": {"true_clock_bias_s": 25e-9, "localization_error_m": .5,
                    "clock_bias_error_ns": 1.},
    }
    monkeypatch.setattr(visualization, "_plotting", lambda: None)
    monkeypatch.setattr(visualization, "load_run", lambda root: deepcopy(run))

    def capture_summary(plt, rows, summary, output):
        assert [row["status"] for row in rows] == ["worker_failed", "success", "pending"]
        assert summary["planned_count"] == 3
        assert summary["failed_count"] == 1
        assert summary["solved_count"] == 1
        assert summary["pending_count"] == 1
        assert summary["within_1m_fraction_all"] == 1 / 3
        assert summary["position_median_m"] == .5
        raise _SummaryReached

    monkeypatch.setattr(step_visualization, "export_summary", capture_summary)
    with pytest.raises(_SummaryReached):
        visualization.create_report(tmp_path / "report", experiment=experiment)
