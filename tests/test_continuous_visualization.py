"""冻结连续结果必须可查看；不依赖旧清单、唯一解或评价真值。"""
from pathlib import Path

import numpy as np
import pytest

from time_bias_localization.constants import SPEED_OF_LIGHT_M_S as C
from time_bias_localization.provenance import artifact_record
from time_bias_localization.scene import make_synthetic_room
from time_bias_localization.visualization import (
    create_report, load_frozen_continuous_run, load_run, read_json, write_json,
)


def _frozen_result(tmp_path, status="success", evaluation=False):
    root = tmp_path / status
    folder = root / "localization"
    folder.mkdir(parents=True)
    scene_path, peaks_path = tmp_path / "scene.json", tmp_path / "peaks.json"
    scene = make_synthetic_room((-10., 10., -10., 10.)).to_dict()
    write_json(scene_path, scene)
    write_json(peaks_path, {"nominal": [{"aoa_rad": .4, "delay_s": 12/C}],
                            "nominal_source_indices": [2]})
    best = {"position_m": [1., 2.], "beta_m": 1., "selected_paths": [],
            "acceptable": status != "solver_budget_exhausted"}
    result = {"workflow": "music_continuous_propagation_v1", "localization_run_id": "run_frozen",
              "status": status, "mu_m": [1., 2.] if status == "success" else None,
              "clock_bias_s": 1/C if status == "success" else None,
              "distance_bias_m": 1. if status == "success" else None,
              "sigma_m2": [[1., 0.], [0., 1.]] if status == "success" else None,
              "selected_paths": [], "best_candidate_for_diagnostics_only": best,
              "alternatives": [{"position_m": [3., 4.], "beta_m": 2., "selected_paths": []}]
                              if status == "ambiguous" else [],
              "diagnostics": {"hypothesis_search": {"budget_exhausted": status == "solver_budget_exhausted",
                                                      "unsearched_sequences": 12 if status == "solver_budget_exhausted" else 0}}}
    config = {"radio": {"bs_position_m": [0., 0.], "bs_boresight_deg": 0.},
              "scene": {"bounds_m": scene["bounds_m"]}}
    payloads = {"localization_result": result, "localization_config": {"resolved_config": config},
                "continuous_observations": {"observations": [{"observation_id": "music_path_02",
                    "aoa_rad": .4, "observed_length_m": 12., "angle_scale_rad": .1, "length_scale_m": 1.}]},
                "propagation_hypotheses": {"hypotheses": [], "search_report": {}},
                "continuous_search": {"status": status}, "forward_check": {"paths": []}}
    for name, document in payloads.items():
        write_json(folder / f"{name}.json", document)
    manifest = {"schema_version": 1, "workflow": result["workflow"],
                "localization_run_id": result["localization_run_id"], "source_binding_verified": False,
                "inputs": {str(path): artifact_record(path) for path in (scene_path, peaks_path)},
                "artifacts": {path.stem: artifact_record(path) for path in folder.glob("*.json")}}
    write_json(folder / "frozen_input_manifest.json", manifest)
    if evaluation:
        truth_path = tmp_path / "truth.npz"
        np.savez(truth_path, ue_position_m=np.array([0., 0.]), clock_bias_s=np.array(2e-8))
        write_json(root / "evaluation/metrics.json", {"status": "evaluated_after_online_solve",
                   "truth_input": artifact_record(truth_path), "position_error_m": np.sqrt(5),
                   "clock_bias_error_ns": abs(1/C - 2e-8) * 1e9})
    return root


@pytest.mark.parametrize("status", ["success", "ambiguous", "solver_budget_exhausted"])
def test_load_run_recognizes_frozen_format_without_legacy_manifest_or_truth(tmp_path, status):
    root = _frozen_result(tmp_path, status)
    run = load_run(root)
    assert run["result"]["status"] == status
    assert run["true"] is None
    assert run["metrics"] == {}
    assert Path(run["artifacts"]["result"]).name == "localization_result.json"
    assert Path(run["artifacts"]["music_peaks"]).name == "peaks.json"
    assert run["result"]["mu_m"] is None if status != "success" else run["result"]["mu_m"] == [1., 2.]


def test_frozen_loader_rejects_changed_source_and_result_id(tmp_path):
    root = _frozen_result(tmp_path)
    (tmp_path / "peaks.json").write_text("changed")
    with pytest.raises(ValueError, match="指纹"):
        load_run(root)
    other = _frozen_result(tmp_path / "other")
    result_path = other / "localization/localization_result.json"
    document = read_json(result_path)
    document["localization_run_id"] = "unrelated"
    write_json(result_path, document)
    manifest_path = other / "localization/frozen_input_manifest.json"
    manifest = read_json(manifest_path)
    manifest["artifacts"]["localization_result"] = artifact_record(result_path)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="运行编号"):
        load_run(other)


def test_optional_evaluation_recomputes_signed_error_and_can_be_disabled(tmp_path, monkeypatch):
    root = _frozen_result(tmp_path, evaluation=True)
    run = load_run(root)
    assert run["true"] == [0., 0.]
    assert run["metrics"]["localization_error_m"] == pytest.approx(np.sqrt(5))
    assert run["metrics"]["clock_bias_error_ns"] < 0
    assert run["evaluation_binding"] == "explicit_hashed_truth_only"
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("禁止打开评价真值"))
    skipped = load_frozen_continuous_run(root, include_evaluation=False)
    assert skipped["true"] is None
    assert skipped["metrics"] == {}


def test_optional_evaluation_rejects_metric_mismatch(tmp_path):
    root = _frozen_result(tmp_path, evaluation=True)
    metrics_path = root / "evaluation/metrics.json"
    document = read_json(metrics_path)
    document["position_error_m"] = 0.
    write_json(metrics_path, document)
    with pytest.raises(ValueError, match="重新计算"):
        load_run(root)


@pytest.mark.parametrize("status", ["success", "ambiguous", "solver_budget_exhausted"])
def test_full_frozen_report_exports_diagnostic_candidates_without_truth(tmp_path, monkeypatch, status):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import time_bias_localization.visualization as visualization
    import time_bias_localization.step_visualization as steps
    root = _frozen_result(tmp_path, status)
    monkeypatch.setattr(visualization, "_plotting", lambda: plt)
    # Exercise actual axes/path creation without font/rendering dependencies.
    monkeypatch.setattr(steps, "_save", lambda _plt, fig, directory, name: plt.close(fig))
    report = create_report(tmp_path / "report", run_roots=[root], skip_evaluation=True)
    row = read_json(report / "results.json")[0]
    assert row["status"] == status
    display = read_json(report / "run_0001/03_optimization/displayed_candidates.json")
    assert display["formal_position_available"] == (status == "success")
    assert display["truth_available"] is False
    assert len(display["candidates"]) == (2 if status == "ambiguous" else 1)
    assert (report / "run_0001/01_observations/peaks.json").is_file()
    assert (report / "run_0001/03_optimization/localization_result.json").is_file()
    assert (report / "run_0001/04_validation/forward_check.json").is_file()
    assert read_json(report / "report_manifest.json")["solver_rerun"] is False


def test_comparison_report_preserves_pending_and_ambiguous_in_denominator(tmp_path):
    root = _frozen_result(tmp_path, "ambiguous")
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    write_json(comparison / "comparison_plan.json", {"planned_observation_count": 2})
    write_json(comparison / "trials.json", [
        {"ue_id": "UE01", "repeat_index": 0, "status": "ambiguous", "result_dir": str(root)},
        {"ue_id": "UE01", "repeat_index": 1, "status": "pending"}])
    report = create_report(tmp_path / "summary_report", experiment=comparison, summary_only=True,
                           skip_evaluation=True)
    summary = read_json(report / "summary.json")
    assert summary["recorded_request_count"] == 2
    assert summary["readable_result_count"] == 1
    assert summary["status_counts"] == {"ambiguous": 1, "pending": 1}
    assert summary["numeric_output_count"] == 0
