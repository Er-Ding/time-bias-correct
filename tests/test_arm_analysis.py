"""未结束、部分分组的对照实验必须保留计划分母，且可独立生成报告。"""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("have_result", [False, True])
def test_partial_arm_report_preserves_pending_and_selected_arms(tmp_path, monkeypatch, have_result):
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_arm_comparison.py"
    spec = importlib.util.spec_from_file_location("arm_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, output = tmp_path / "report", tmp_path / "analysis"
    source.mkdir()
    plan = {"arms": ["M"], "arm_definitions": {"B0": {}, "M": {}, "W": {}},
            "selection": {"excluded": ["SAMPLE_1", "SAMPLE_2"], "success_with_boundary": []},
            "records": {sid: {"true_position_m": [1., 2.], "true_clock_bias_ns": 0.}
                        for sid in ("SAMPLE_1", "SAMPLE_2")}}
    (source / "plan.json").write_text(json.dumps(plan))
    if have_result:
        (source / "results.json").write_text(json.dumps({"results": [{
            "group": "excluded", "arm": "M", "sample_id": "SAMPLE_1", "status": "success",
            "mu_m": [1., 2.], "clock_bias_s": 0., "processing_seconds": 2., "process_threads": 1}]}))
    monkeypatch.setattr("sys.argv", [str(path), "--report", str(source), "--output", str(output)])
    module.main()
    summary = json.loads((output / "analysis.json").read_text())
    row = summary["per_group_arm"]["excluded"]["M"]
    assert row["tasks"] == 2 and row["completed"] == int(have_result)
    assert row["pending"] == row["status_counts"]["pending"] == 2 - int(have_result)
    assert row["success_rate"] == (0.5 if have_result else 0.)
    assert list(summary["arms"]) == ["M"] and summary["paired_comparisons"] == []
    assert summary["per_group_arm"]["success_with_boundary"]["M"]["median_seconds"] is None
    assert "未完成" in (output / "analysis.md").read_text()
