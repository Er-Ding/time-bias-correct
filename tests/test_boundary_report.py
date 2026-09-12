"""冻结实验分母、独立误差评价、阶段计时和配对输入的一致性。"""

import csv
import json

import numpy as np
import pytest

from time_bias_localization.boundary_report import create_boundary_report
import time_bias_localization.boundary_report as report


def _point(ue_id="ue001", cohort="pilot", repeats=5, position=(2., 3.)):
    return dict(ue_id=ue_id, cohort=cohort, position_m=list(position),
                noise_seeds=list(range(repeats)), mc_seeds=list(range(100, 100 + repeats)))


def _record(repeat=0, strategy="single", **values):
    return dict(cohort="pilot", ue_id="ue001", repeat_index=repeat, strategy=strategy,
                status=values.pop("status", "success"), **values)


def _rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_all_frozen_requests_remain_denominator_and_diagnostics_do_not_filter_error(tmp_path):
    records = [
        _record(position_error_m=.2, forward_valid=False, identifiable=False, localization_seconds=2.),
        _record(1, position_error_m=3., forward_valid=True, identifiable=True, localization_seconds=4.),
        _record(2, status="timeout", localization_seconds=8.),
        _record(3, status="localization_failed", localization_seconds=1.),
        _record(strategy="coverage", position_error_m=.1, forward_valid=True, localization_seconds=3.),
    ]
    result = create_boundary_report(tmp_path, records, [_point()], metadata={"plots": False})
    single = next(group for group in result["groups"] if group["strategy"] == "single")
    assert result["planned_count"] == 10
    assert single["planned_count"] == 5
    assert single["numeric_output_count"] == 2
    assert single["output_fraction_all_planned"] == .4
    assert single["status_counts"]["pending"] == 1
    assert single["position_error_m"]["mean"] == pytest.approx(1.6)
    assert single["position_error_m"]["rmse"] == pytest.approx(np.sqrt((.2 ** 2 + 3. ** 2) / 2))
    assert single["thresholds_m"][0]["fraction_all_planned"] == .2
    assert single["thresholds_m"][-1]["fraction_all_planned"] == .4
    assert single["forward_valid_fraction_checked"] == .5
    assert single["forward_valid_fraction_all_planned"] == .2
    assert single["unidentifiable_count"] == 1
    timing = single["latency_seconds"]["localization_seconds"]
    assert timing["all_measured_requests"]["mean"] == 3.75
    assert timing["all_measured_requests"]["count"] == 4
    assert timing["numeric_output_requests"]["mean"] == 3.
    trials = _rows(tmp_path / "trials.csv")
    assert len(trials) == 10
    timeout = next(row for row in trials if row["status"] == "timeout")
    assert timeout["position_error_m"] == ""
    assert timeout["within_5m"] == "False"
    assert len(_rows(tmp_path / "per_ue.csv")) == 2
    assert "NaN" not in (tmp_path / "summary.json").read_text()


def test_repeated_stage_events_sum_per_request_and_failed_stage_has_separate_denominator(tmp_path):
    def stage(elapsed, status="completed", name="T05_fine_spectrum"):
        return dict(name=name, elapsed_s=elapsed, exclusive_s=elapsed / 2., status=status)
    records = [
        _record(stage_timings=[stage(.01), stage(.03)]),
        _record(1, stage_timings=[stage(.1)]),
        _record(2, status="localization_failed", stage_timings=[stage(.2, "failed")]),
    ]
    result = create_boundary_report(tmp_path, records, [_point()], metadata={"plots": False})
    timing = next(row for row in result["stage_summary"] if row["strategy"] == "single" and row["name"] == "T05_fine_spectrum")
    assert timing["planned_count"] == 5
    assert timing["executed_count"] == 3
    assert timing["complete_count"] == 2
    assert timing["failed_count"] == 1
    assert timing["not_executed_count"] == 2
    assert timing["event_count"] == 4
    assert timing["complete_elapsed_ms_mean"] == 70.
    assert timing["complete_exclusive_ms_mean"] == 35.
    assert timing["failed_elapsed_ms_mean"] == 200.
    absent = next(row for row in result["stage_summary"] if row["strategy"] == "single" and row["name"] == "T12_solver")
    assert absent["not_executed_count"] == 5
    assert absent["complete_elapsed_ms_count"] == 0
    assert absent["complete_elapsed_ms_mean"] is None
    stages = _rows(tmp_path / "stages.csv")
    assert all(row["elapsed_s"] == "" for row in stages if row["status"] == "not_executed")


def test_paired_comparison_uses_both_outputs_keeps_one_sided_failures_and_input_hashes(tmp_path):
    records = [
        _record(position_error_m=2., localization_seconds=1., input_sha256="same0"),
        _record(strategy="coverage", position_error_m=.5, localization_seconds=3., input_sha256="same0"),
        _record(1, status="localization_failed", input_sha256="same1"),
        _record(1, "coverage", position_error_m=4., localization_seconds=8., input_sha256="same1"),
    ]
    result = create_boundary_report(tmp_path, records, [_point(repeats=3)], metadata={"plots": False})
    comparison = result["paired_comparison"][0]
    assert comparison["planned_pair_count"] == 3
    assert comparison["both_output_count"] == 1
    assert comparison["coverage_only_output_count"] == 1
    assert comparison["neither_output_count"] == 1
    assert comparison["input_hash_verified_count"] == 2
    assert comparison["paired_deltas"]["position_error_m"]["mean"] == -1.5
    assert comparison["paired_deltas"]["localization_seconds"]["mean"] == 2.
    records[-1]["input_sha256"] = "different-observation"
    with pytest.raises(ValueError, match="相同观测"):
        create_boundary_report(tmp_path / "bad", records, [_point(repeats=3)], metadata={"plots": False})
    assert not (tmp_path / "bad" / "summary.json").exists()


def test_hard_killed_running_stage_retains_parent_and_unknown_duration(tmp_path):
    record = _record(status="timeout", stage_timings=[
        dict(name="T12_solver", event_id=8, parent_event_id=0, depth=1,
             elapsed_s=None, exclusive_s=None, start_s=.1, end_s=None, status="running"),
        dict(name="T12_seed_generation", event_id=9, parent_event_id=8, depth=2,
             elapsed_s=.01, exclusive_s=.01, start_s=.2, end_s=.21, status="complete"),
    ])
    result = create_boundary_report(tmp_path, [record], [_point(repeats=1)], metadata={"plots": False})
    solver = next(row for row in result["stage_summary"] if row["name"] == "T12_solver" and row["strategy"] == "single")
    assert solver["executed_count"] == 1
    assert solver["complete_count"] == 0
    assert solver["failed_count"] == 0
    assert solver["incomplete_count"] == 1
    assert solver["incomplete_elapsed_ms_mean"] is None
    events = _rows(tmp_path / "stages.csv")
    child = next(row for row in events if row["name"] == "T12_seed_generation" and row["strategy"] == "single")
    assert child["parent_event_id"] == "8"
    assert child["event_id"] == "9"


def test_channel_groups_keep_planned_failures_show_diffraction_pairs_without_reweighting_main(tmp_path):
    diffraction = {**_point(repeats=2), "channel_category": "diffraction_only", "has_diffraction": True}
    reflection = {**_point("ue002", repeats=2), "channel_category": "reflection_without_los", "has_diffraction": False}

    def stage(duration):
        return [dict(name="T12_solver", elapsed_s=duration, exclusive_s=duration, status="complete")]

    records = [
        _record(position_error_m=3., localization_seconds=.02, stage_timings=stage(.02)),
        _record(strategy="coverage", position_error_m=1., localization_seconds=.04, stage_timings=stage(.04)),
        _record(1, status="localization_failed"),
        _record(1, "coverage", position_error_m=2., localization_seconds=.06, stage_timings=stage(.06)),
        {**_record(position_error_m=.1, localization_seconds=.1, stage_timings=stage(.1)), "ue_id": "ue002"},
        {**_record(strategy="coverage", position_error_m=.1, localization_seconds=.1, stage_timings=stage(.1)), "ue_id": "ue002"},
    ]
    result = create_boundary_report(tmp_path, records, [diffraction, reflection], metadata={"plots": False})
    main = next(row for row in result["groups"] if row["strategy"] == "single")
    assert main["planned_count"] == 4
    assert main["position_error_m"]["mean"] == pytest.approx(1.55)
    groups = result["channel_groups"]
    d_single = next(row for row in groups if row["group_field"] == "has_diffraction"
                    and row["group_value"] is True and row["strategy"] == "single")
    assert d_single["planned_count"] == 2
    assert d_single["numeric_output_count"] == 1
    assert d_single["status_counts"]["localization_failed"] == 1
    assert d_single["position_error_m"]["mean"] == 3.
    assert d_single["latency_seconds"]["localization_seconds"]["all_measured_requests"]["mean"] == .02
    r_single = next(row for row in groups if row["group_field"] == "channel_category"
                    and row["group_value"] == "reflection_without_los" and row["strategy"] == "single")
    assert r_single["planned_count"] == 2
    assert r_single["status_counts"]["pending"] == 1
    pairs = result["channel_paired_comparison"]
    paired = next(row for row in pairs if row["group_field"] == "has_diffraction" and row["group_value"] is True)
    assert paired["planned_pair_count"] == 2
    assert paired["both_output_count"] == 1
    assert paired["coverage_only_output_count"] == 1
    assert paired["position_error_m_delta_mean"] == -2.
    assert paired["localization_seconds_delta_mean"] == .02
    timed = next(row for row in result["channel_stage_summary"] if row["group_field"] == "has_diffraction"
                 and row["group_value"] is True and row["name"] == "T12_solver" and row["strategy"] == "coverage")
    assert timed["complete_elapsed_ms_mean"] == 50.
    assert timed["complete_count"] == 2
    trials = _rows(tmp_path / "trials.csv")
    pending = next(row for row in trials if row["status"] == "pending")
    assert pending["has_diffraction"] == "False"
    assert pending["channel_category"] == "reflection_without_los"
    assert _rows(tmp_path / "summary" / "precision_by_channel.csv")
    assert _rows(tmp_path / "summary" / "timing_by_channel.csv")
    assert _rows(tmp_path / "summary" / "paired_by_channel.csv")


def test_channel_labels_from_old_records_propagate_to_pending_and_conflicts_fail(tmp_path):
    source = _record(position_error_m=1., channel_category="los", has_diffraction=True)
    result = create_boundary_report(tmp_path, [source], [_point(repeats=2)], metadata={"plots": False})
    assert all(row["planned_count"] == 2 for row in result["channel_groups"])
    assert not any(row["group_value"] == "unknown" for row in result["channel_groups"])
    conflict = _record(strategy="coverage", has_diffraction=False)
    with pytest.raises(ValueError, match="标签不一致"):
        create_boundary_report(tmp_path / "bad_records", [source, conflict], [_point(repeats=2)], metadata={"plots": False})
    frozen = {**_point(repeats=2), "has_diffraction": False}
    with pytest.raises(ValueError, match="冻结 UE"):
        create_boundary_report(tmp_path / "bad_frozen", [source], [frozen], metadata={"plots": False})


@pytest.mark.parametrize("bad", ["duplicate", "unplanned", "seeds", "negative_time", "invalid_stage"])
def test_reject_records_that_change_plan_or_corrupt_timings(tmp_path, bad):
    records = [_record(position_error_m=.2)]
    if bad == "duplicate":
        records.append(dict(records[0]))
    elif bad == "unplanned":
        records[0]["ue_id"] = "ue999"
    elif bad == "seeds":
        records[0]["noise_seed"] = 42
    elif bad == "negative_time":
        records[0]["processing_seconds"] = -1.
    else:
        records[0]["stage_timings"] = [dict(name="T01_covariance", elapsed_s=1., exclusive_s=2.)]
    with pytest.raises(ValueError):
        create_boundary_report(tmp_path, records, [_point()], metadata={"plots": False})


def test_pilot_formal_separated_nonfinite_outputs_missing_time_and_raw_log_preserved(tmp_path):
    points = [_point(repeats=1), _point("formal001", "formal", repeats=1)]
    records = [_record(position_error_m=float("nan"))]
    raw = tmp_path / "task.log"
    raw.write_text("原始日志不覆盖\n")
    result = create_boundary_report(tmp_path, records, points, metadata={"plots": False})
    assert len(result["groups"]) == 4
    group = next(item for item in result["groups"] if item["cohort"] == "pilot" and item["strategy"] == "single")
    assert group["numeric_output_count"] == 0
    assert group["position_error_m"]["mean"] is None
    assert group["latency_seconds"]["processing_seconds"]["missing_executed_count"] == 1
    assert raw.read_text() == "原始日志不覆盖\n"
    assert json.loads((tmp_path / "summary.json").read_text())["planned_count"] == 4


def test_figures_distinguish_conditional_cdf_from_all_planned_and_show_failed_positions(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = {}
    monkeypatch.setattr(report, "_plotting", lambda: plt)
    monkeypatch.setattr(report, "_save", lambda _, fig, folder, name: figures.__setitem__(name, fig))
    records = [_record(position_error_m=2., localization_seconds=.5, representative_count=4., stage_timings=[
        dict(name="T12_solver", elapsed_s=.4, exclusive_s=.1, status="completed"),
        dict(name="T12_iterations", elapsed_s=.3, exclusive_s=.3, status="completed")])]
    result = create_boundary_report(tmp_path, records, [_point(repeats=2)])
    try:
        assert result["plot_status"] == "complete"
        cdf = figures["pilot_error_cdf"]
        assert cdf.axes[0].lines[0].get_ydata()[-1] == 1.
        assert cdf.axes[1].lines[0].get_ydata()[-1] == .5
        stage = figures["pilot_stage_times"]
        # 主阶段图不把求解器总阶段与迭代子阶段重复相加。
        assert len(stage.axes[0].get_yticklabels()) == 1
        assert stage.axes[0].patches[0].get_width() == 400.
        spatial = figures["pilot_spatial_error"]
        assert any(collection.get_label() == "无可用数值（失败或待运行）"
                   for axis in spatial.axes for collection in axis.collections)
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_unbalanced_cache_excludes_only_paired_timing_not_accuracy_or_total_latency(tmp_path):
    records = [
        _record(position_error_m=2., localization_seconds=10., execution_state='cold', pair_cache_comparable=False),
        _record(strategy='coverage', position_error_m=1., localization_seconds=3., execution_state='reused', pair_cache_comparable=False),
        _record(1, status='timeout', localization_seconds=12., execution_state='cold', pair_cache_comparable=True),
        _record(1, strategy='coverage', status='timeout', localization_seconds=15., execution_state='cold', pair_cache_comparable=True),
    ]
    result = create_boundary_report(tmp_path, records, [_point(repeats=2)], metadata={'plots': False})
    pair = result['paired_comparison'][0]
    assert pair['paired_deltas']['position_error_m']['mean'] == -1.
    assert pair['paired_deltas']['localization_seconds']['count'] == 0
    assert pair['paired_latency_all_requests']['localization_seconds']['mean'] == 3.
    assert pair['cache_noncomparable_pair_count'] == 1
    single = next(row for row in result['groups'] if row['strategy'] == 'single')
    assert single['latency_seconds']['localization_seconds']['all_measured_requests']['mean'] == 11.
    assert single['execution_state_counts'] == {'cold': 2}
    assert single['latency_by_execution_state']['cold']['localization_seconds']['count'] == 2
