"""计时要保留失败、避免父子步骤重复相加，并覆盖真实定位调用链。"""

from copy import deepcopy
import json

import numpy as np
import pytest

from time_bias_localization.config import DEFAULT_CONFIG, localization_config_view
from time_bias_localization.pipeline import generate_data, localize, prepare_scene
from time_bias_localization.timing import collect_timings, current_timings, mark, stage


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_nested_timings_account_without_double_counting():
    clock = FakeClock()
    with collect_timings(clock=clock) as recorder:
        mark("csi_map_ready")
        with stage("parent"):
            clock.advance(2.0)
            with stage("child"):
                clock.advance(3.0)
            clock.advance(1.0)
        clock.advance(4.0)
        mark("position_available", mu_m=[1.0, 2.0])
    result = recorder.to_dict()
    assert result["elapsed_s"] == 10.0
    assert result["recorded_exclusive_s"] == 6.0
    assert result["unattributed_s"] == 4.0
    parent, child = result["events"]
    assert parent["elapsed_s"] == 6.0
    assert parent["exclusive_s"] == 3.0
    assert child["elapsed_s"] == child["exclusive_s"] == 3.0
    assert child["parent_event_id"] == parent["event_id"]
    assert result["marks"] == {"csi_map_ready": 0.0, "position_available": 10.0}
    assert result["mark_data"]["position_available"]["mu_m"] == [1.0, 2.0]
    json.dumps(result, allow_nan=False)


def test_failure_keeps_elapsed_and_unexecuted_stage_is_absent():
    clock = FakeClock()
    with pytest.raises(RuntimeError, match="deliberate"):
        with collect_timings(clock=clock) as recorder:
            with stage("solver"):
                with stage("seeds"):
                    clock.advance(1.5)
                    raise RuntimeError("deliberate")
            with stage("not_reached"):
                pass
    result = recorder.to_dict()
    assert result["status"] == "failed"
    assert result["elapsed_s"] == result["recorded_exclusive_s"] == 1.5
    assert all(row["status"] == "failed" for row in result["events"])
    assert all(row["failed_count"] == 1 for row in result["stages"])
    assert {row["name"] for row in result["events"]} == {"solver", "seeds"}
    assert current_timings() is None


def test_timing_contexts_restore_and_are_opt_in():
    with stage("not_collected"):
        mark("not_collected")
    assert current_timings() is None
    with collect_timings() as outer:
        with stage("outer_first"):
            pass
        with collect_timings() as inner:
            with stage("inner"):
                pass
        with stage("outer_second"):
            pass
    assert [row["name"] for row in outer.events] == ["outer_first", "outer_second"]
    assert [row["name"] for row in inner.events] == ["inner"]


def test_sync_before_and_after_includes_execution_wait_only():
    clock = FakeClock()
    sync_calls = []

    def synchronize():
        sync_calls.append(clock())
        clock.advance(2.0)

    with collect_timings(clock=clock, synchronize=synchronize) as recorder:
        with stage("gpu"):
            clock.advance(3.0)
    row = recorder.to_dict()["events"][0]
    assert sync_calls == [0.0, 5.0]
    assert row["start_s"] == 2.0
    assert row["elapsed_s"] == 5.0
    assert recorder.to_dict()["unattributed_s"] == 2.0


def test_sync_failure_preserves_original_error_and_elapsed():
    clock = FakeClock()
    calls = 0

    def synchronize():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("sync failed")

    with pytest.raises(ValueError, match="original"):
        with collect_timings(clock=clock, synchronize=synchronize) as recorder:
            with stage("gpu"):
                clock.advance(1.0)
                raise ValueError("original")
    event = recorder.to_dict()["events"][0]
    assert event["status"] == "failed"
    assert event["elapsed_s"] == 1.0
    assert event["error_type"] == "ValueError"
    assert "sync failed" in event["synchronization_error"]


def test_atomic_snapshot_records_running_stage_then_failure(tmp_path):
    snapshot = tmp_path / "timing.json"
    with pytest.raises(RuntimeError, match="stop"):
        with collect_timings(snapshot_path=snapshot) as recorder:
            mark("csi_map_ready")
            with stage("solver"):
                running = json.loads(snapshot.read_text())
                assert running["events"][0]["status"] == "running"
                assert running["events"][0]["elapsed_s"] is None
                raise RuntimeError("stop")
    final = json.loads(snapshot.read_text())
    assert final["status"] == "failed"
    assert final["events"][0]["status"] == "failed"
    assert final["events"][0]["elapsed_s"] >= 0
    assert recorder.to_dict()["snapshot_write_count"] == 4
    assert not list(tmp_path.glob(".timing.json.*"))


def _input_bundle(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    config["music"]["spectrum_sampling"].update(samples_per_peak=8, local_grid_points_per_axis=9)
    root = tmp_path / "input"
    scene = prepare_scene(config, root)
    data = generate_data(config, scene_json=scene["scene_json"], output_root=root)
    return localization_config_view(config), scene, data


def test_real_pipeline_timings_cover_boundaries_and_preserve_numerics(tmp_path):
    config, scene, data = _input_bundle(tmp_path)
    args = dict(scene_json=scene["scene_json"], online_input=data["online_npz"],
                generation_manifest=data["generation_manifest"])
    plain = localize(config, **args, output_root=tmp_path / "plain")
    with collect_timings() as recorder:
        timed = localize(config, **args, output_root=tmp_path / "timed")
    np.testing.assert_array_equal(timed["mu_m"], plain["mu_m"])
    assert timed["clock_bias_s"] == plain["clock_bias_s"]
    report = recorder.to_dict()
    names = {row["name"] for row in report["stages"]}
    assert {"T01_covariance", "T02_subspace", "T03_coarse_spectrum", "T04_coarse_peaks",
            "T05_fine_spectrum", "T06_fine_peaks", "T07_feature_sampling", "T08_reverse_rt",
            "T11_trajectories", "T12_solver", "T12_seed_generation", "T12_seed_scoring",
            "T12_iterations", "T13_online_checks", "artifact_publication"} <= names
    boundaries = report["marks"]
    assert 0 <= boundaries["csi_map_ready"] < boundaries["position_available"]
    assert boundaries["position_available"] < boundaries["checked_complete"]
    assert boundaries["checked_complete"] < boundaries["files_published"] <= report["elapsed_s"]
    assert report["recorded_exclusive_s"] <= report["elapsed_s"]
    assert all(row["status"] == "complete" for row in report["events"])
    np.testing.assert_array_equal(report["mark_data"]["position_available"]["mu_m"], timed["mu_m"])
    assert "stage_timings_s" in timed["diagnostics"]


def test_pipeline_failure_boundary_precedes_failure_output(tmp_path, monkeypatch):
    import time_bias_localization.pipeline as pipeline

    config, scene, data = _input_bundle(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("intentional covariance failure")

    monkeypatch.setattr(pipeline, "solve_position_and_bias", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        with collect_timings() as recorder:
            localize(config, scene_json=scene["scene_json"], online_input=data["online_npz"],
                     generation_manifest=data["generation_manifest"], output_root=tmp_path / "failed")
    report = recorder.to_dict()
    assert report["status"] == "failed"
    assert "position_available" not in report["marks"]
    assert "checked_complete" not in report["marks"]
    failure_event = next(row for row in report["events"] if row["name"] == "T12_solver")
    publication = next(row for row in report["events"] if row["name"] == "failure_artifact_publication")
    assert failure_event["status"] == "failed"
    assert report["marks"]["online_failed"] <= publication["start_s"]
