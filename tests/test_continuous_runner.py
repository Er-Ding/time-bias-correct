"""冻结对照保留总分母、隔离真值，并且不能把评估错误改成在线求解失败。"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


def runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_continuous_comparison.py"
    spec = importlib.util.spec_from_file_location("continuous_comparison_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_keeps_excluded_insufficient_and_old_failed_requests(tmp_path):
    runner = runner_module()
    cohort = tmp_path / "pilot"
    (cohort / "report").mkdir(parents=True)
    observations = [{"online_npz": f"/public/online/{i}.npz", "scene_json": "/public/map.json",
                     "truth_npz": f"/evaluation/{i}.npz"} for i in range(4)]
    (cohort / "plan.json").write_text(json.dumps({"points": [{"ue_id": "UE01", "observations": observations}]}))
    rows = [{"ue_id": "UE01", "repeat_index": i, "strategy": "coverage",
             "result_dir": str(tmp_path / f"old{i}"), "status": status, "stop_reason": reason}
            for i, (status, reason) in enumerate([
                ("success", ""), ("excluded_observation", "near_identical_music_responses"),
                ("unlocalizable", "insufficient_music_peaks"), ("failed", "no_representatives")])]
    with (cohort / "report" / "trials.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    jobs = runner.load_inventory(tmp_path, "pilot", "coverage")
    assert len(jobs) == 4
    assert [job["frozen_screen_status"] for job in jobs] == [
        None, "excluded_observation", "insufficient_observations", None]
    # 没有旧代表的请求仍然可进入正式新求解器。
    assert jobs[3]["baseline"]["historical_status"] == "failed"
    assert "truth_npz" not in jobs[0]
    summary = runner.summarize([
        {"status": "ambiguous", "attempted": True},
        {"status": "pending", "attempted": False},
        {"status": "excluded_observation", "frozen_screen_status": "excluded_observation"},
        {"status": "insufficient_observations", "frozen_screen_status": "insufficient_observations"},
    ], planned=4, completed=True)
    assert summary["planned_observation_count"] == 4
    assert summary["pending_count"] == 1
    assert summary["full_frozen_cohort_finished"] is False
    assert summary["success_rate_total_denominator"] == 0


def test_frozen_hash_mismatch_fails_and_no_numeric_result_does_not_open_truth(tmp_path):
    runner = runner_module()
    source = tmp_path / "frozen.json"
    source.write_text("original")
    record = runner.artifact(source)
    source.write_text("modified")
    with pytest.raises(ValueError, match="冻结输入内容已经改变"):
        runner.artifact(source, record["sha256"])
    assert runner.evaluate_numeric_result({"mu_m": None}, {
        "truth_npz": "/this/file/must/not/be/opened", "truth_sha256": "fake"}) == {
        "status": "no_numeric_output"}


def test_numeric_evaluation_does_not_change_solver_result(tmp_path):
    runner = runner_module()
    truth = tmp_path / "truth.npz"
    np.savez(truth, ue_position_m=np.array([1., 2.]), clock_bias_s=np.array(2e-8))
    result = {"mu_m": [4., 6.], "clock_bias_s": 3e-8, "status": "ambiguous"}
    evaluation = runner.evaluate_numeric_result(result, {
        "truth_npz": str(truth), "truth_sha256": runner.file_sha256(truth)})
    assert evaluation["position_error_m"] == pytest.approx(5.)
    assert evaluation["clock_bias_error_ns"] == pytest.approx(10.)
    assert result["status"] == "ambiguous"


def test_runner_preserves_ambiguous_status_and_pending_denominator_on_evaluation_failure(tmp_path, monkeypatch):
    runner = runner_module()
    source = tmp_path / "old"
    (source / "pilot" / "report").mkdir(parents=True)
    for name in ("plan.json", "report/trials.csv"):
        (source / "pilot" / name).write_text("frozen source")
    config_path = tmp_path / "public.yaml"
    config_path.write_text("public config")
    jobs = [{"ue_id": "UE01", "repeat_index": i, "frozen_screen_status": None,
             "evaluation_input": {"truth_npz": "/evaluation-only"},
             "baseline": {"result_dir": str(source / f"missing-result-{i}")}}
            for i in range(2)]
    monkeypatch.setattr(runner, "load_inventory", lambda *_: jobs)
    monkeypatch.setattr(runner, "source_fingerprint", lambda: {"unchanged": True})
    monkeypatch.setattr(runner, "load_localization_config", lambda _: {"localization": {"solver_method": "continuous"}})
    records = {key: {"path": f"/public/{key}", "sha256": "frozen"}
               for key in ("scene", "music_peaks", "online_input", "manifest")}
    monkeypatch.setattr(runner, "prepare_frozen_input", lambda *_: ({"public": True}, records))
    calls = []
    pipeline = ModuleType("time_bias_localization.continuous_pipeline")

    def online(config, **kwargs):
        calls.append(kwargs)
        assert config == {"public": True}
        assert not any("truth" in key for key in kwargs)
        assert kwargs["return_problem"] is True
        return ({"status": "ambiguous", "mu_m": [3., 4.], "clock_bias_s": 2e-8}, None, None)

    pipeline.localize_saved_music = online
    monkeypatch.setitem(sys.modules, pipeline.__name__, pipeline)

    def broken_evaluation(*_):
        raise ValueError("evaluation truth unavailable")

    monkeypatch.setattr(runner, "evaluate_numeric_result", broken_evaluation)
    output = tmp_path / "new"
    code = runner.main(["--source", str(source), "--config", str(config_path),
                        "--output", str(output), "--limit", "1"])
    rows = json.loads((output / "trials.json").read_text())
    assert code == 1
    assert len(calls) == 1
    assert rows[0]["status"] == "ambiguous"
    assert rows[0]["evaluation_status"] == "evaluation_failed"
    assert rows[1]["status"] == "pending"
    assert json.loads((output / "summary.json").read_text())["planned_observation_count"] == 2
