"""独立评估侧的边界实验汇总；失败、待运行和未执行阶段均保留分母。"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np

from .visualization import _plotting, _save, write_json


STAGE_LABELS = {
    "T01_covariance": "CSI 整理与协方差",
    "T02_subspace": "子空间计算",
    "T03_coarse_spectrum": "粗 MUSIC 谱",
    "T04_coarse_peaks": "粗谱找峰",
    "T05_fine_spectrum": "局部细 MUSIC 谱",
    "T06_fine_peaks": "细谱找峰与去重",
    "T07_feature_sampling": "观测特征采样",
    "T08_reverse_rt": "反向 RT",
    "T09_dbscan": "DBSCAN 聚类",
    "T10_representatives": "代表点选择",
    "T11_trajectories": "代表轨迹构建",
    "T12_solver": "位置和偏差求解",
    "T13_online_checks": "在线几何诊断",
}
SUBSTAGE_LABELS = {
    "T08_specular": "反向 RT：直射和镜面反射分支",
    "T08_diffraction": "反向 RT：绕射分支",
    "T12_seed_generation": "求解：初值候选对生成",
    "T12_seed_scoring": "求解：初值评分",
    "T12_iterations": "求解：候选选择和迭代",
    "T12_result": "求解：结果与条件诊断",
}
THRESHOLDS_M = (0.5, 1.0, 2.0, 5.0)
_SUCCESS_STAGE_STATES = {"success", "completed", "complete", "ok"}
_TRIAL_STATUSES = {"success", "localization_failed", "timeout", "data_failed", "pending"}
_KEYS = ("cohort", "ue_id", "repeat_index", "strategy")
_CHANNEL_FIELDS = ("channel_category", "has_diffraction")
_CHANNEL_LABELS = {
    "los": "有直射路径", "reflection_without_los": "无直射，有纯反射路径",
    "diffraction_only": "仅含绕射路径可达", "none": "未发现有效路径", "unknown": "类别未知",
}
_NUMERIC_FIELDS = (
    "position_error_m", "clock_bias_error_ns", "processing_seconds",
    "localization_seconds", "checked_seconds", "initial_count", "cluster_count",
    "representative_count", "diffraction_representative_count",
)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _stats(values: Iterable[Any], *, rmse: bool = False) -> dict[str, Any]:
    numbers = [number for value in values if (number := _number(value)) is not None]
    data = np.asarray(numbers, dtype=float)
    result = {"count": len(numbers), "mean": None, "std": None, "median": None,
              "p90": None, "p95": None, "max": None}
    if len(data):
        result.update(mean=float(np.mean(data)), std=float(np.std(data)),
                      median=float(np.median(data)), p90=float(np.quantile(data, .9)),
                      p95=float(np.quantile(data, .95)), max=float(np.max(data)))
    if rmse:
        result["rmse"] = float(np.sqrt(np.mean(data ** 2))) if len(data) else None
    return result


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _trial_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[key] for key in _KEYS)


def _repeat_count(point: dict[str, Any], metadata: dict[str, Any]) -> int:
    lengths = [len(point[key]) for key in ("noise_seeds", "mc_seeds") if key in point]
    if lengths and len(set(lengths)) != 1:
        raise ValueError(f"UE {point['ue_id']} 的噪声与 MC 种子数量不同")
    value = lengths[0] if lengths else metadata.get("noise_repeats", 5)
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError("每个 UE 的重复次数必须是正整数")
    return int(value)


def _materialize_trials(records, points, metadata):
    """按冻结清单补齐待运行项，不允许重复项或清单外记录悄悄改变分母。"""
    strategies = list(metadata.get("strategies", ["single", "coverage"]))
    if not strategies or len(set(strategies)) != len(strategies):
        raise ValueError("代表策略清单不能为空或包含重复值")
    planned = {}
    point_keys = set()
    for point in points:
        point_key = (point["cohort"], point["ue_id"])
        if point_key in point_keys:
            raise ValueError(f"冻结 UE 清单包含重复位置编号：{point_key}")
        point_keys.add(point_key)
        position = np.asarray(point["position_m"], dtype=float)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError(f"UE {point['ue_id']} 的二维真值坐标无效")
        for repeat in range(_repeat_count(point, metadata)):
            for strategy in strategies:
                trial = dict(cohort=point["cohort"], ue_id=point["ue_id"],
                             repeat_index=repeat, strategy=strategy, status="pending",
                             true_x_m=float(position[0]), true_y_m=float(position[1]),
                             noise_seed=point.get("noise_seeds", [None] * (repeat + 1))[repeat],
                             mc_seed=point.get("mc_seeds", [None] * (repeat + 1))[repeat])
                for field in _NUMERIC_FIELDS:
                    trial[field] = None
                trial.update(forward_valid=None, identifiable=None, stage_timings=[],
                             **{field: point.get(field) for field in _CHANNEL_FIELDS})
                planned[_trial_key(trial)] = trial
    seen = set()
    for source in records:
        key = _trial_key(source)
        if key not in planned:
            raise ValueError(f"定位记录不在冻结的实验清单内：{key}")
        if key in seen:
            raise ValueError(f"同一计划尝试有多条记录，请先明确选用的运行编号：{key}")
        seen.add(key)
        trial = planned[key]
        for seed in ("noise_seed", "mc_seed"):
            if source.get(seed) is not None and trial[seed] is not None and source[seed] != trial[seed]:
                raise ValueError(f"定位记录的 {seed} 与冻结清单不一致：{key}")
        for field in _CHANNEL_FIELDS:
            if source.get(field) is not None and trial[field] is not None and source[field] != trial[field]:
                raise ValueError(f"定位记录的 {field} 与冻结 UE 的信道类别不一致：{key}")
        frozen_coordinates = (trial["true_x_m"], trial["true_y_m"])
        frozen_channel = {field: trial[field] for field in _CHANNEL_FIELDS if trial[field] is not None}
        trial.update(source)
        trial["true_x_m"], trial["true_y_m"] = frozen_coordinates
        trial.update(frozen_channel)
        if trial["status"] not in _TRIAL_STATUSES:
            raise ValueError(f"未知的定位终态：{trial['status']}")
        for field in _NUMERIC_FIELDS:
            trial[field] = _number(trial.get(field))
        for field in ("position_error_m", "processing_seconds", "localization_seconds", "checked_seconds"):
            if trial[field] is not None and trial[field] < 0:
                raise ValueError(f"{field} 不能是负数：{key}")
        for field in ("forward_valid", "identifiable"):
            if trial.get(field) is not None and not isinstance(trial[field], (bool, np.bool_)):
                raise ValueError(f"{field} 必须是布尔值或空值：{key}")
            if trial.get(field) is not None:
                trial[field] = bool(trial[field])
    trials = list(planned.values())
    # 优先使用冻结点的生成侧标签。兼容标签仅随旧记录保存的情况，但要求同一 UE 一致，
    # 并把标签补给它的全部计划重复，不能让失败/待运行项掉出对应传播类别的分母。
    by_point = defaultdict(list)
    for trial in trials:
        by_point[(trial["cohort"], trial["ue_id"])].append(trial)
    for point_key, subset in by_point.items():
        for field in _CHANNEL_FIELDS:
            values = [trial[field] for trial in subset if trial[field] is not None]
            if field == "has_diffraction" and any(not isinstance(value, (bool, np.bool_)) for value in values):
                raise ValueError(f"has_diffraction 必须是生成侧布尔标签或空值：{point_key}")
            if field == "channel_category" and any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"channel_category 必须是非空类别名称或空值：{point_key}")
            if values and any(value != values[0] for value in values[1:]):
                raise ValueError(f"同一 UE 的 {field} 标签不一致：{point_key}")
            value = values[0] if values else None
            if field == "has_diffraction" and value is not None:
                value = bool(value)
            for trial in subset:
                trial[field] = value
    for trial in trials:
        trial["has_numeric_output"] = trial["position_error_m"] is not None
        for threshold in THRESHOLDS_M:
            trial[f"within_{threshold:g}m"] = bool(trial["has_numeric_output"] and trial["position_error_m"] <= threshold)
    return trials


def _group_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    output = [trial for trial in trials if trial["has_numeric_output"]]
    executed = [trial for trial in trials if trial["status"] != "pending"]
    diagnostics = [trial for trial in trials if trial.get("forward_valid") is not None]
    constraints = [trial for trial in trials if trial.get("identifiable") is not None]
    counts = Counter(trial["status"] for trial in trials)
    result = {
        "planned_count": len(trials), "executed_count": len(executed),
        "ue_count": len({trial["ue_id"] for trial in trials}),
        "numeric_output_count": len(output),
        "output_fraction_all_planned": _fraction(len(output), len(trials)),
        "status_counts": {status: counts[status] for status in sorted(_TRIAL_STATUSES)},
        "position_error_m": _stats((trial["position_error_m"] for trial in trials), rmse=True),
        "clock_bias_error_ns": _stats((trial["clock_bias_error_ns"] for trial in trials), rmse=True),
        "clock_bias_mae_ns": _stats(abs(trial["clock_bias_error_ns"]) for trial in trials
                                   if trial["clock_bias_error_ns"] is not None)["mean"],
        "thresholds_m": [{"threshold_m": threshold,
                          "count": sum(trial[f"within_{threshold:g}m"] for trial in trials),
                          "denominator": len(trials),
                          "fraction_all_planned": _fraction(sum(trial[f"within_{threshold:g}m"] for trial in trials), len(trials))}
                         for threshold in THRESHOLDS_M],
        "forward_checked_count": len(diagnostics),
        "forward_valid_count": sum(trial["forward_valid"] for trial in diagnostics),
        "forward_valid_fraction_checked": _fraction(sum(trial["forward_valid"] for trial in diagnostics), len(diagnostics)),
        "forward_valid_fraction_all_planned": _fraction(sum(trial["forward_valid"] for trial in diagnostics), len(trials)),
        "identifiability_checked_count": len(constraints),
        "unidentifiable_count": sum(not trial["identifiable"] for trial in constraints),
        "unidentifiable_fraction_checked": _fraction(sum(not trial["identifiable"] for trial in constraints), len(constraints)),
        "timeout_fraction_all_planned": _fraction(counts["timeout"], len(trials)),
        "latency_seconds": {},
        "counts": {},
        "execution_state_counts": dict(Counter((trial.get("execution_state") or "unknown") for trial in executed)),
        "latency_by_execution_state": {
            state: {field: _stats(trial.get(field) for trial in executed if (trial.get("execution_state") or "unknown") == state)
                    for field in ("processing_seconds", "localization_seconds", "checked_seconds")}
            for state in sorted({(trial.get("execution_state") or "unknown") for trial in executed})},
    }
    for field in ("processing_seconds", "localization_seconds", "checked_seconds"):
        result["latency_seconds"][field] = {
            "all_measured_requests": _stats(trial[field] for trial in trials),
            "numeric_output_requests": _stats(trial[field] for trial in output),
            "missing_executed_count": sum(trial[field] is None for trial in executed),
            "planned_count": len(trials),
        }
    for field in ("initial_count", "cluster_count", "representative_count", "diffraction_representative_count"):
        result["counts"][field] = _stats(trial[field] for trial in trials)
    return result


def _stage_tables(trials, metadata):
    declared = metadata.get("stage_names", list(STAGE_LABELS))
    if not isinstance(declared, (list, tuple)):
        raise ValueError("stage_names 必须是阶段名称列表")
    names = list(dict.fromkeys([*declared, *(str(stage["name"]) for trial in trials
                                         for stage in trial.get("stage_timings", []))]))
    rows = []
    for trial in trials:
        grouped = defaultdict(list)
        for index, stage in enumerate(trial.get("stage_timings", [])):
            name = str(stage["name"])
            elapsed = _number(stage.get("elapsed_s"))
            exclusive = _number(stage.get("exclusive_s"))
            if (elapsed is not None and elapsed < 0) or (exclusive is not None and exclusive < 0):
                raise ValueError(f"阶段耗时不能为负：{name}")
            if elapsed is not None and exclusive is not None and exclusive > elapsed + 1e-8:
                raise ValueError(f"阶段自身耗时超过包含子阶段的耗时：{name}")
            status = str(stage.get("status", "completed"))
            if status in _SUCCESS_STAGE_STATES and elapsed is None:
                raise ValueError(f"完整执行阶段缺少有效耗时：{name}")
            row = {key: trial[key] for key in _KEYS}
            row.update(name=name, label=STAGE_LABELS.get(name, SUBSTAGE_LABELS.get(name, name)), event_index=index,
                       status=status, elapsed_s=elapsed, exclusive_s=exclusive,
                       trial_status=trial["status"], parent=stage.get("parent", stage.get("parent_event_id")),
                       event_id=stage.get("event_id"), parent_event_id=stage.get("parent_event_id"),
                       depth=stage.get("depth"), start_s=_number(stage.get("start_s")),
                       end_s=_number(stage.get("end_s")), device=stage.get("device"),
                       quantity=stage.get("quantity"), metadata=stage.get("metadata"))
            rows.append(row)
            grouped[name].append(row)
        for name in names:
            if name not in grouped:
                row = {key: trial[key] for key in _KEYS}
                row.update(name=name, label=STAGE_LABELS.get(name, SUBSTAGE_LABELS.get(name, name)), event_index=None,
                           status="not_executed", elapsed_s=None, exclusive_s=None,
                           trial_status=trial["status"], parent=None, depth=None,
                           start_s=None, end_s=None, device=None, quantity=None,
                           event_id=None, parent_event_id=None, metadata=None)
                rows.append(row)
    by_request = defaultdict(list)
    for row in rows:
        by_request[(*_trial_key(row), row["name"])].append(row)
    grouped_requests = defaultdict(list)
    for key, events in by_request.items():
        active = [row for row in events if row["status"] != "not_executed"]
        complete = bool(active) and all(row["status"] in _SUCCESS_STAGE_STATES for row in active)
        value = {"executed": bool(active), "complete": complete,
                 "failed": any(row["status"] in {"failed", "error", "timeout"} for row in active),
                 "elapsed_ms": sum(row["elapsed_s"] for row in active) * 1000
                 if active and all(row["elapsed_s"] is not None for row in active) else None,
                 "exclusive_ms": sum(row["exclusive_s"] for row in active) * 1000
                 if active and all(row["exclusive_s"] is not None for row in active) else None,
                 "event_count": len(active)}
        grouped_requests[(key[0], key[3], key[4])].append(value)
    summaries = []
    for (cohort, strategy, name), requests in sorted(grouped_requests.items()):
        complete = [request for request in requests if request["complete"]]
        failed = [request for request in requests if request["failed"]]
        incomplete = [request for request in requests if request["executed"] and not request["complete"] and not request["failed"]]
        summary = {"cohort": cohort, "strategy": strategy, "name": name,
                   "label": STAGE_LABELS.get(name, SUBSTAGE_LABELS.get(name, name)), "planned_count": len(requests),
                   "executed_count": sum(request["executed"] for request in requests),
                   "complete_count": len(complete), "failed_count": len(failed),
                   "incomplete_count": len(incomplete),
                   "not_executed_count": sum(not request["executed"] for request in requests),
                   "event_count": sum(request["event_count"] for request in requests)}
        for label, values in (("complete_elapsed_ms", (request["elapsed_ms"] for request in complete)),
                              ("complete_exclusive_ms", (request["exclusive_ms"] for request in complete)),
                              ("failed_elapsed_ms", (request["elapsed_ms"] for request in failed)),
                              ("incomplete_elapsed_ms", (request["elapsed_ms"] for request in incomplete))):
            summary.update({f"{label}_{stat}": value for stat, value in _stats(values).items()})
        summaries.append(summary)
    return rows, summaries


def _paired_summary(trials):
    pairs = defaultdict(dict)
    for trial in trials:
        if trial["strategy"] in {"single", "coverage"}:
            pairs[(trial["cohort"], trial["ue_id"], trial["repeat_index"])][trial["strategy"]] = trial
    grouped = defaultdict(list)
    for key, pair in pairs.items():
        if set(pair) != {"single", "coverage"}:
            continue
        left, right = pair["single"], pair["coverage"]
        hashes = [(left.get(field), right.get(field)) for field in
                  ("input_sha256", "input_fingerprint", "csi_sha256")]
        comparable = [(a, b) for a, b in hashes if a is not None and b is not None]
        if any(a != b for a, b in comparable):
            raise ValueError(f"两种代表策略没有使用相同观测：{key}")
        item = {"both_output": left["has_numeric_output"] and right["has_numeric_output"],
                "single_only_output": left["has_numeric_output"] and not right["has_numeric_output"],
                "coverage_only_output": right["has_numeric_output"] and not left["has_numeric_output"],
                "neither_output": not left["has_numeric_output"] and not right["has_numeric_output"],
                "input_hash_verified": bool(comparable), "left": left, "right": right}
        grouped[key[0]].append(item)
    result = []
    for cohort, pairs in sorted(grouped.items()):
        both = [pair for pair in pairs if pair["both_output"]]
        summary = {"cohort": cohort, "planned_pair_count": len(pairs),
                   "delta_convention": "coverage_minus_single", "paired_deltas": {},
                   **{f"{key}_count": sum(pair[key] for pair in pairs) for key in
                      ("both_output", "single_only_output", "coverage_only_output", "neither_output", "input_hash_verified")}}
        timing_pairs = [pair for pair in pairs if pair["left"].get("pair_cache_comparable") is not False
                        and pair["right"].get("pair_cache_comparable") is not False]
        summary["cache_comparable_pair_count"] = len(timing_pairs)
        summary["cache_noncomparable_pair_count"] = len(pairs) - len(timing_pairs)
        summary["paired_latency_all_requests"] = {
            field: _stats(pair["right"][field] - pair["left"][field] for pair in timing_pairs
                          if pair["right"][field] is not None and pair["left"][field] is not None)
            for field in ("localization_seconds", "checked_seconds", "processing_seconds")}
        for field in ("position_error_m", "localization_seconds", "checked_seconds", "processing_seconds",
                      "representative_count", "diffraction_representative_count"):
            summary["paired_deltas"][field] = _stats(pair["right"][field] - pair["left"][field]
                                                   for pair in both if pair["right"][field] is not None
                                                   and pair["left"][field] is not None
                                                   and (not field.endswith("seconds") or
                                                        (pair["left"].get("pair_cache_comparable") is not False
                                                         and pair["right"].get("pair_cache_comparable") is not False)))
        summary["lower_error_coverage_count"] = sum(pair["right"]["position_error_m"] < pair["left"]["position_error_m"] for pair in both)
        result.append(summary)
    return result


def _channel_label(field, value):
    if field == "has_diffraction":
        return "含绕射路径" if value is True else "不含绕射路径" if value is False else "是否含绕射未知"
    return _CHANNEL_LABELS.get(value, value)


def _channel_summaries(trials, metadata):
    """两个生成侧分类维度分别观察同一总体，保持原始面积抽样权重。"""
    slices = defaultdict(list)
    for trial in trials:
        for field in _CHANNEL_FIELDS:
            value = trial[field] if trial[field] is not None else "unknown"
            slices[(field, value)].append(trial)
    precision_rows, timing_rows, paired_rows, summaries = [], [], [], []
    for (field, value), subset in sorted(slices.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        grouping = dict(group_field=field, group_value=value, group_label=_channel_label(field, value))
        per_group = defaultdict(list)
        for trial in subset:
            per_group[(trial["cohort"], trial["strategy"])].append(trial)
        for (cohort, strategy), members in sorted(per_group.items()):
            result = _group_summary(members)
            summaries.append({**grouping, "cohort": cohort, "strategy": strategy, **result})
            precision_rows.append({**grouping, **_flatten_precision(cohort, strategy, result)})
        _, timings = _stage_tables(subset, metadata)
        timing_rows.extend({**grouping, **timing} for timing in timings)
        for pair in _paired_summary(subset):
            # JSON 保留完整结构；CSV 同时展开常用差值，便于直接比较绕射子集的成本和收益。
            paired_rows.append({**grouping, **pair, **{
                f"{name}_delta_{stat}": number for name, stats in pair["paired_deltas"].items()
                for stat, number in stats.items()}})
    return summaries, precision_rows, timing_rows, paired_rows


def _flatten_precision(cohort, strategy, summary):
    row = {"cohort": cohort, "strategy": strategy,
           **{key: summary[key] for key in ("planned_count", "executed_count", "ue_count",
               "numeric_output_count", "output_fraction_all_planned", "forward_checked_count",
               "forward_valid_count", "forward_valid_fraction_checked", "forward_valid_fraction_all_planned",
               "identifiability_checked_count", "unidentifiable_count", "unidentifiable_fraction_checked",
               "timeout_fraction_all_planned")}}
    row.update({f"status_{key}_count": value for key, value in summary["status_counts"].items()})
    row.update({f"position_error_{key}_m": value for key, value in summary["position_error_m"].items() if key != "count"})
    row["clock_bias_mae_ns"] = summary["clock_bias_mae_ns"]
    for item in summary["thresholds_m"]:
        suffix = f"{item['threshold_m']:g}m"
        row[f"within_{suffix}_count"] = item["count"]
        row[f"within_{suffix}_denominator"] = item["denominator"]
        row[f"within_{suffix}_fraction_all_planned"] = item["fraction_all_planned"]
    for field, timings in summary["latency_seconds"].items():
        for population in ("all_measured_requests", "numeric_output_requests"):
            row.update({f"{field}_{population}_{stat}": value for stat, value in timings[population].items()})
        row[f"{field}_missing_executed_count"] = timings["missing_executed_count"]
    return row


def _write_csv(path: Path, rows, *, empty_fields=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row)) or list(empty_fields)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                     dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list, tuple)) else value for field, value in row.items()})
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def _plot_report(output_dir, trials, stage_summary, per_ue, metadata, channel_precision=(), channel_paired=()):
    plt = _plotting()
    folder = output_dir / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    artifacts = []
    scene = metadata.get("scene", {})
    if not scene and metadata.get("scene_json"):
        scene = json.loads(Path(metadata["scene_json"]).read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for trial in trials:
        grouped[trial["cohort"]].append(trial)
    strategy_labels = {"single": "绕射簇单代表", "coverage": "绕射簇多代表"}
    for cohort, cohort_trials in sorted(grouped.items()):
        strategies = list(dict.fromkeys(trial["strategy"] for trial in cohort_trials))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for strategy in strategies:
            subset = [trial for trial in cohort_trials if trial["strategy"] == strategy]
            errors = np.sort([trial["position_error_m"] for trial in subset if trial["has_numeric_output"]])
            label = f"{strategy_labels.get(strategy, strategy)}（输出 {len(errors)}/{len(subset)}）"
            if len(errors):
                x = np.r_[0., errors]
                axes[0].step(x, np.r_[0., np.arange(1, len(errors) + 1) / len(errors)], where="post", label=label)
                axes[1].step(x, np.r_[0., np.arange(1, len(errors) + 1) / len(subset)], where="post", label=label)
            else:
                axes[0].plot([], [], label=label)
                axes[1].plot([0., 5.], [0., 0.], label=label)
        for ax, title in zip(axes, ("有数值输出条件下的误差分布", "以全部计划请求为分母的达标率")):
            ax.set(xlabel="欧式位置误差 / 米", ylabel="比例", title=title, ylim=(0., 1.02))
            ax.grid(alpha=.2)
            ax.legend(fontsize=6)
        name = f"{cohort}_error_cdf"
        _save(plt, fig, folder, name)
        artifacts.append(f"plots/{name}.png")

        stage_names = [name for name in metadata.get("stage_names", list(STAGE_LABELS)) if name in STAGE_LABELS]
        stage_names = [name for name in stage_names if any(row["cohort"] == cohort and row["name"] == name
                       and row["complete_elapsed_ms_count"] for row in stage_summary)]
        fig, ax = plt.subplots(figsize=(10, max(3., .35 * len(stage_names))), layout="constrained")
        y = np.arange(len(stage_names))
        width = .8 / max(len(strategies), 1)
        for index, strategy in enumerate(strategies):
            lookup = {row["name"]: row for row in stage_summary if row["cohort"] == cohort and row["strategy"] == strategy}
            values = [lookup[name]["complete_elapsed_ms_mean"] for name in stage_names]
            # 主表仅画互不嵌套的 T01–T13，各项包含其内部子步骤；子步骤不再另画相加。
            heights = [float("nan") if value is None else value for value in values]
            ax.barh(y + (index - (len(strategies) - 1) / 2) * width, heights, height=width,
                    label=strategy_labels.get(strategy, strategy))
        ax.set(yticks=y, yticklabels=[STAGE_LABELS.get(name, name) for name in stage_names],
               xlabel="完整执行请求的阶段平均耗时 / 毫秒", title="主阶段包含内部子步骤；实际统计次数见 timing.csv")
        ax.invert_yaxis()
        ax.legend(fontsize=7)
        ax.grid(axis="x", alpha=.2)
        name = f"{cohort}_stage_times"
        _save(plt, fig, folder, name)
        artifacts.append(f"plots/{name}.png")

        ue_rows = [row for row in per_ue if row["cohort"] == cohort]
        for value_key, title, unit, suffix in (
            ("position_error_mean_m", "各 UE 的平均欧式位置误差", "米", "spatial_error"),
            ("localization_seconds_all_measured_requests_mean", "各 UE 的平均定位用时", "秒", "spatial_time"),
        ):
            fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 4.5),
                                     squeeze=False, layout="constrained")
            finite = [row[value_key] for row in ue_rows if row.get(value_key) is not None]
            maximum = max(finite, default=1.) or 1.
            for ax, strategy in zip(axes[0], strategies):
                subset = [row for row in ue_rows if row["strategy"] == strategy]
                valid = [row for row in subset if row.get(value_key) is not None]
                missing = [row for row in subset if row.get(value_key) is None]
                for wall in scene.get("walls", []):
                    start, end = wall["start_m"], wall["end_m"]
                    ax.plot([start[0], end[0]], [start[1], end[1]], color=".75", linewidth=.6, zorder=0)
                if valid:
                    collection = ax.scatter([row["true_x_m"] for row in valid], [row["true_y_m"] for row in valid],
                                            c=[row[value_key] for row in valid], vmin=0., vmax=maximum, cmap="viridis", s=25)
                    fig.colorbar(collection, ax=ax, label=unit)
                if missing:
                    ax.scatter([row["true_x_m"] for row in missing], [row["true_y_m"] for row in missing],
                               marker="x", c=".4", s=30, label="无可用数值（失败或待运行）")
                    ax.legend(fontsize=6)
                ax.set(xlabel="x / 米", ylabel="y / 米", title=strategy_labels.get(strategy, strategy))
                ax.set_aspect("equal", adjustable="datalim")
            fig.suptitle(f"{title}；每次噪声独立评分后再汇总")
            name = f"{cohort}_{suffix}"
            _save(plt, fig, folder, name)
            artifacts.append(f"plots/{name}.png")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for strategy in strategies:
            subset = [trial for trial in cohort_trials if trial["strategy"] == strategy]
            times = sorted(trial["localization_seconds"] for trial in subset if trial["localization_seconds"] is not None)
            if times:
                axes[0].step(np.r_[0., times], np.r_[0., np.arange(1, len(times) + 1) / len(times)],
                             where="post", label=f"{strategy_labels.get(strategy, strategy)}（n={len(times)}）")
            valid = [trial for trial in subset if trial["representative_count"] is not None
                     and trial["localization_seconds"] is not None]
            if valid:
                axes[1].scatter([trial["representative_count"] for trial in valid],
                                [trial["localization_seconds"] for trial in valid], s=12, alpha=.5,
                                label=strategy_labels.get(strategy, strategy))
        axes[0].set(xlabel="定位用时 / 秒", ylabel="已测请求比例", title="定位延迟分布，包含有计时的失败")
        axes[1].set(xlabel="代表点数量", ylabel="定位用时 / 秒", title="代表数量与定位用时")
        for ax in axes:
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=6)
            ax.grid(alpha=.2)
        name = f"{cohort}_latency"
        _save(plt, fig, folder, name)
        artifacts.append(f"plots/{name}.png")
        for field in _CHANNEL_FIELDS:
            category_rows = [row for row in channel_precision if row["cohort"] == cohort and row["group_field"] == field]
            labels = list(dict.fromkeys(row["group_label"] for row in category_rows))
            if not labels:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
            x = np.arange(len(labels))
            width = .8 / max(len(strategies), 1)
            for index, strategy in enumerate(strategies):
                lookup = {row["group_label"]: row for row in category_rows if row["strategy"] == strategy}
                for axis, metric in zip(axes[:2], ("position_error_mean_m", "localization_seconds_all_measured_requests_mean")):
                    values = [lookup[label].get(metric) for label in labels]
                    axis.bar(x + (index - (len(strategies) - 1) / 2) * width,
                             [float("nan") if value is None else value for value in values],
                             width, label=strategy_labels.get(strategy, strategy))
                for position, label in zip(x, labels):
                    row = lookup[label]
                    value = row.get("position_error_mean_m")
                    axes[0].annotate(f"{row['numeric_output_count']}/{row['planned_count']}",
                                     (position + (index - (len(strategies) - 1) / 2) * width, value or 0.),
                                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6)
            comparison = {row["group_label"]: row for row in channel_paired
                          if row["cohort"] == cohort and row["group_field"] == field}
            deltas = [comparison.get(label, {}).get("position_error_m_delta_mean") for label in labels]
            axes[2].bar(x, [float("nan") if value is None else value for value in deltas], .65, color="tab:purple")
            axes[2].axhline(0., color=".5", linewidth=.7)
            for position, label, delta in zip(x, labels, deltas):
                pair = comparison.get(label, {})
                axes[2].annotate(f"配对 n={pair.get('both_output_count', 0)}", (position, delta or 0.),
                                 xytext=(0, 4 if delta is None or delta >= 0 else -10),
                                 textcoords="offset points", ha="center", fontsize=6)
            for axis in axes:
                axis.set_xticks(x, labels, rotation=15, ha="right")
                axis.grid(axis="y", alpha=.2)
            axes[0].set(title="各类有输出条件下的平均误差", ylabel="欧式位置误差 / 米")
            axes[1].set(title="各类实际测得的平均定位用时", ylabel="定位用时 / 秒")
            axes[2].set(title="两组都有输出的配对误差差值", ylabel="多代表减单代表 / 米")
            axes[0].legend(fontsize=6)
            fig.suptitle("生成侧传播类别的独立分析；数字为输出次数/计划次数，总体均匀权重不变")
            name = f"{cohort}_by_{field}"
            _save(plt, fig, folder, name)
            artifacts.append(f"plots/{name}.png")
    return artifacts


def create_boundary_report(output_dir, records, points, *, metadata=None) -> dict[str, Any]:
    """在调用者指定的报告目录原子发布派生表；不改写输入、运行日志或原始结果。

    ``repeat_index`` 从 0 开始。误差必须已由独立评估阶段计算；本函数不调用定位器。
    阶段列表允许同一阶段多次出现，统计时先按请求求和，避免细谱区域数改变均值分母。
    """
    output_dir = Path(output_dir)
    metadata = dict(metadata or {})
    trials = _materialize_trials(list(records), list(points), metadata)
    stage_rows, stage_summary = _stage_tables(trials, metadata)
    grouped = defaultdict(list)
    per_point = defaultdict(list)
    for trial in trials:
        grouped[(trial["cohort"], trial["strategy"])].append(trial)
        per_point[(trial["cohort"], trial["strategy"], trial["ue_id"])].append(trial)
    groups = []
    precision_rows = []
    for (cohort, strategy), subset in sorted(grouped.items()):
        result = _group_summary(subset)
        groups.append({"cohort": cohort, "strategy": strategy, **result})
        precision_rows.append(_flatten_precision(cohort, strategy, result))
    per_ue = []
    for (cohort, strategy, ue_id), subset in sorted(per_point.items()):
        per_ue.append({"ue_id": ue_id, "true_x_m": subset[0]["true_x_m"], "true_y_m": subset[0]["true_y_m"],
                       **{field: subset[0][field] for field in _CHANNEL_FIELDS},
                       **_flatten_precision(cohort, strategy, _group_summary(subset))})
    channel_groups, channel_precision, channel_timing, channel_paired = _channel_summaries(trials, metadata)
    summary = {
        "schema_version": 1, "report_kind": "diffraction_boundary_experiment",
        "planned_count": len(trials), "groups": groups,
        "paired_comparison": _paired_summary(trials), "stage_summary": stage_summary,
        "channel_groups": channel_groups, "channel_paired_comparison": channel_paired,
        "channel_stage_summary": channel_timing,
        "definitions": {
            "position_error": "二维欧式距离；所有有限数值输出均保留，不按正向检查或可辨识性筛除",
            "output_fraction": "有限位置误差的次数 / 全部冻结计划次数；待运行也保留在分母中",
            "latency": "对实际测得值统计并给出次数；待运行或未测量时间不填零",
            "stage_mean": "先累加同一请求的同名阶段，再对该阶段完整执行的请求统计；失败阶段另列",
            "stage_exclusive": "阶段自身耗时；不含已单列的子阶段，避免重复相加",
            "execution_state": "cold 为重建后的首次请求；prewarmed 为成功预热后的首次请求；reused 为同策略进程内复用，分层计时见 latency_by_execution_state",
            "paired_cache_policy": "恢复运行的半对或状态不可比的请求不计算成对时间差；精度保留，主时间表保留所有已测请求",
            "paired_delta": "多代表减去单代表；仅对两组都有数值位置输出的相同请求计算",
            "channel_groups": "按生成侧 channel_category 和 has_diffraction 分别观察原均匀样本；两种分类视图互有重叠，不跨视图求和或重加权总体",
            "channel_denominator": "类别来自冻结 UE 或一致的生成侧记录，并传播到同一 UE 的失败和待运行项；缺失类别单列未知组",
            "std": "实际已测样本的总体标准差，ddof=0；不作为独立空间采样的标准误",
            "ofdm_receiver_frontend": "同步、FFT、导频 CSI 估计未测量，不计为零",
        },
        "metadata": metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "trials.csv", [{key: value for key, value in trial.items() if key != "stage_timings"} for trial in trials], empty_fields=_KEYS)
    _write_csv(output_dir / "stages.csv", stage_rows, empty_fields=(*_KEYS, "name", "status", "elapsed_s", "exclusive_s"))
    _write_csv(output_dir / "per_ue.csv", per_ue, empty_fields=("cohort", "strategy", "ue_id"))
    _write_csv(output_dir / "summary" / "precision.csv", precision_rows, empty_fields=("cohort", "strategy", "planned_count"))
    _write_csv(output_dir / "summary" / "timing.csv", stage_summary, empty_fields=("cohort", "strategy", "name"))
    _write_csv(output_dir / "summary" / "paired.csv", summary["paired_comparison"], empty_fields=("cohort", "planned_pair_count"))
    _write_csv(output_dir / "summary" / "precision_by_channel.csv", channel_precision,
               empty_fields=("group_field", "group_value", "cohort", "strategy", "planned_count"))
    _write_csv(output_dir / "summary" / "timing_by_channel.csv", channel_timing,
               empty_fields=("group_field", "group_value", "cohort", "strategy", "name"))
    _write_csv(output_dir / "summary" / "paired_by_channel.csv", channel_paired,
               empty_fields=("group_field", "group_value", "cohort", "planned_pair_count"))
    if metadata.get("plots", True):
        try:
            summary["plot_artifacts"] = _plot_report(output_dir, trials, stage_summary, per_ue, metadata,
                                                     channel_precision, channel_paired)
            summary["plot_status"] = "complete"
        except (ImportError, RuntimeError) as error:
            summary["plot_artifacts"] = []
            summary["plot_status"] = "unavailable"
            summary["plot_error"] = str(error)
    else:
        summary["plot_artifacts"] = []
        summary["plot_status"] = "disabled"
    write_json(output_dir / "summary.json", summary)
    return summary
