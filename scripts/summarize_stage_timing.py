"""只读统计：汇总一次蒙特卡罗实验里每个定时步骤的真实耗时。

数据来源是每个定位尝试目录下的 ``timing_live.json``（``collect_timings`` 的原子快照）。
同一份时间有两种口径，不能混用：

* ``marks``：顶层相位边界，是求解流程的一级分割（输入、传播函数库、连续优化、发布）。
* ``stages``：嵌套计时步骤。只累加 ``exclusive_s``（已扣除子步骤），
  否则父步骤和子步骤会重复计一次。

脚本不重跑定位、不修改任何实验产物，只在指定目录写出汇总 JSON 和 Markdown。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

NUMERIC_KEYS = ("mean", "median", "p90", "p95", "max", "min")


def numeric_summary(values):
    array = np.asarray(list(values), float)
    if not array.size:
        return {"count": 0, **{key: None for key in NUMERIC_KEYS}}
    return {"count": int(array.size),
            "mean": float(array.mean()), "median": float(np.median(array)),
            "min": float(array.min()), "p90": float(np.quantile(array, 0.9)),
            "p95": float(np.quantile(array, 0.95)), "max": float(array.max())}


def load_samples(run_root: Path):
    """返回每个样本一条记录；同一样本只取最新一次尝试。"""
    samples = []
    for directory in sorted((run_root / "samples").glob("*")):
        if not directory.is_dir():
            continue
        attempts = sorted((directory / "localization_attempts").glob("*/timing_live.json"))
        if not attempts:
            continue
        # 目录名带时间戳前缀，排序后最后一个是最近一次尝试。
        record = {"sample_id": directory.name, "timing": json.loads(attempts[-1].read_text()),
                  "timing_path": str(attempts[-1])}
        result_path = directory / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            record["status"] = result.get("status")
            record["position_error_m"] = result.get("position_error_m")
            record["processing_seconds"] = result.get("processing_seconds")
        else:
            record["status"] = "no_result"
            record["position_error_m"] = None
            record["processing_seconds"] = None
        samples.append(record)
    return samples


def collect_stage_rows(samples):
    """按步骤名汇总 exclusive_s。失败步骤保留其已消耗时间。"""
    per_stage = defaultdict(list)
    per_stage_status = defaultdict(lambda: defaultdict(int))
    totals = []
    for sample in samples:
        timing = sample["timing"]
        totals.append(timing["elapsed_s"])
        for stage in timing["stages"]:
            if stage["exclusive_s"] is None:
                continue
            per_stage[stage["name"]].append(stage["exclusive_s"])
            per_stage_status[stage["name"]][stage["status"]] += 1
    rows = []
    grand_total = float(np.sum(totals)) if totals else 0.0
    for name, values in per_stage.items():
        summary = numeric_summary(values)
        summary["name"] = name
        summary["total_s"] = float(np.sum(values))
        summary["share_of_total"] = (summary["total_s"] / grand_total) if grand_total else None
        summary["status_counts"] = dict(per_stage_status[name])
        rows.append(summary)
    rows.sort(key=lambda row: row["total_s"], reverse=True)
    return rows, totals


def mark_order(samples):
    """marks 的显示顺序；不同样本的插入顺序可能不同（失败路径更短）。"""
    ordered = []
    for sample in samples:
        for name in sample["timing"]["marks"]:
            if name not in ordered:
                ordered.append(name)
    return ordered


def collect_mark_rows(samples, grand_total):
    """相邻 mark 的差分即该相位耗时。前一个 mark 必须来自同一样本自己的顺序，
    否则成功路径与失败路径的相位会被错误相减。"""
    rows = []
    for name in mark_order(samples):
        values = []
        for sample in samples:
            marks = sample["timing"]["marks"]
            if name not in marks:
                continue
            keys = list(marks)
            position = keys.index(name)
            start = marks[keys[position - 1]] if position else 0.0
            values.append(max(0.0, marks[name] - start))
        if not values:
            continue
        summary = numeric_summary(values)
        summary["name"] = name
        summary["total_s"] = float(np.sum(values))
        summary["share_of_total"] = (summary["total_s"] / grand_total) if grand_total else None
        rows.append(summary)
    return rows


def format_table(rows, title, key="total_s", value_header="总耗时(s)"):
    lines = [f"## {title}", "",
             f"| 步骤 | 样本数 | {value_header} | 占比 | 均值 | 中位 | P90 | 最大 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        share = row.get("share_of_total")
        share_text = f"{share*100:.2f}%" if share is not None else "-"
        lines.append("| {name} | {count} | {total:.1f} | {share} | {mean:.3f} | "
                     "{median:.3f} | {p90:.3f} | {max:.3f} |".format(
                         name=row["name"], count=row["count"], total=row[key], share=share_text,
                         **{item: (row[item] or 0.0) for item in ("mean", "median", "p90", "max")}))
    lines.append("")
    return lines


SOLVER_COUNTER_KEYS = ("observation_count", "hypothesis_count", "starts_attempted",
                       "path_checks", "invalid_path_checks", "backtracked_steps",
                       "acceptable_distinct_solutions", "competitive_alternative_count")


def collect_solver_rows(samples):
    """连续求解器把内部工作量记在 continuous_search.json 的诊断里。

    C02 只有一个计时步骤，无法再从阶段名细分；但求解器的几何求值次数解释了
    它的耗时，因此把计数器一并汇总，用于判断 NN 替代的收益上限。
    """
    per_key = defaultdict(list)
    for sample in samples:
        search = Path(sample["timing_path"]).parent / "localization" / "continuous_search.json"
        if not search.exists():
            continue
        diagnostics = json.loads(search.read_text()).get("diagnostics", {})
        for key in SOLVER_COUNTER_KEYS:
            if diagnostics.get(key) is not None:
                per_key[key].append(diagnostics[key])
    rows = []
    for key in SOLVER_COUNTER_KEYS:
        if key not in per_key:
            continue
        summary = numeric_summary(per_key[key])
        summary["name"] = key
        summary["total_s"] = float(np.sum(per_key[key]))
        rows.append(summary)
    return rows


def attach_geometry_cost(solver_rows, stage_rows):
    """用 path_checks 与 C02 时间估算单次几何求值成本，并给出无效路径占比。"""
    counters = {row["name"]: row for row in solver_rows}
    checks = counters.get("path_checks", {}).get("total_s")
    c02 = next((row for row in stage_rows if row["name"] == "C02_continuous_optimization"), None)
    if not checks or c02 is None:
        return None
    invalid = counters.get("invalid_path_checks", {}).get("total_s", 0.0)
    return {
        "geometry_evaluations_total": checks,
        "geometry_evaluations_per_sample_median": counters["path_checks"]["median"],
        "invalid_fraction": invalid / checks if checks else None,
        "c02_total_s": c02["total_s"],
        "per_evaluation_ms": c02["total_s"] / checks * 1000.0,
        "scope": "path_checks 覆盖 evaluate_hypothesis 的几何检查；不含纯代数残差求值",
    }


def summarize(run_root: Path, out_dir: Path, workers=None):
    samples = load_samples(run_root)
    if not samples:
        raise SystemExit(f"没有找到定时快照：{run_root}")
    stage_rows, totals = collect_stage_rows(samples)
    grand_total = float(np.sum(totals))
    wall_estimate = grand_total / workers if workers else None
    solver_rows = collect_solver_rows(samples)
    statuses = defaultdict(int)
    for sample in samples:
        statuses[sample["status"]] += 1
    report = {
        "schema_version": 1, "run_root": str(run_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clock": "perf_counter_wall_time",
        "sample_count": len(samples), "status_counts": dict(statuses),
        "total_stage_seconds": grand_total,
        "total_stage_hours": grand_total / 3600,
        "workers_assumed": workers, "estimated_wall_hours": (wall_estimate / 3600) if wall_estimate else None,
        "per_sample_elapsed_s": numeric_summary(totals),
        "per_sample_processing_s": numeric_summary(
            [s["processing_seconds"] for s in samples if s["processing_seconds"] is not None]),
        "stages": stage_rows,
        "marks": collect_mark_rows(samples, grand_total),
        "solver_counters": solver_rows,
        "geometry_cost": attach_geometry_cost(solver_rows, stage_rows),
        "summation_rule": "只累加 exclusive_s；不把父步骤与其子步骤相加",
        "source": "samples/*/localization_attempts/*/timing_live.json（每样本取最新一次尝试）",
    }
    report_dir = out_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "stage_timing_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n")

    lines = ["# 定位流程分步耗时统计", "",
             f"- 实验目录：{run_root}", f"- 样本数：{len(samples)}", f"- 状态计数：{dict(statuses)}",
             f"- 各步骤 exclusive 时间合计：{grand_total/3600:.1f} h"]
    if wall_estimate:
        lines.append(f"- 按 {workers} 个定位进程估算墙钟：{wall_estimate/3600:.1f} h")
    lines.append("")
    lines += format_table(stage_rows, "嵌套步骤（exclusive 口径）")
    lines += format_table(report["marks"], "顶层相位（marks 差分口径；同为秒）")
    lines += format_table(report["solver_counters"], "连续求解器内部计数（不是时间，单位见名）",
                          value_header="合计")
    lines += ["## 读数注意", "",
              "`position_available` 只在求解真正输出位置时打点；求解失败的样本没有这个 mark，"
              "其连续优化耗时会计入 `checked_complete` 的差分。跨样本比较看中位数，不要只看均值。", ""]
    cost = report["geometry_cost"]
    if cost:
        lines += ["## 几何求值成本", "",
                  f"- 几何求值总次数：{cost['geometry_evaluations_total']:.0f}，"
                  f"每样本中位 {cost['geometry_evaluations_per_sample_median']:.0f} 次",
                  f"- 其中被判定不合法的比例：{cost['invalid_fraction']*100:.1f}%",
                  f"- 单次几何求值折算墙钟：{cost['per_evaluation_ms']:.3f} ms", "",
                  cost["scope"], ""]
    lines += ["## 口径说明", "", report["summation_rule"], "", report["source"], ""]
    (report_dir / "stage_timing_summary.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="蒙特卡罗实验输出目录")
    parser.add_argument("--out", type=Path, help="汇总输出目录；默认写到 run-root/timing_summary")
    parser.add_argument("--workers", type=int, help="定位进程数，仅用于估算墙钟")
    parser.add_argument("--top", type=int, default=20, help="控制台打印的步骤条数")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    out_dir = args.out.resolve() if args.out else run_root / "timing_summary"
    report = summarize(run_root, out_dir, workers=args.workers)
    print(f"样本 {report['sample_count']}，状态 {report['status_counts']}")
    print(f"合计 {report['total_stage_hours']:.1f} h"
          + (f"，估算墙钟 {report['estimated_wall_hours']:.1f} h" if report["estimated_wall_hours"] else ""))
    print(f"{'步骤':34s} {'样本':>5s} {'总耗时(s)':>11s} {'占比':>8s} {'中位(s)':>10s} {'P90(s)':>10s}")
    for row in report["stages"][:args.top]:
        share = row["share_of_total"] or 0.0
        print(f"{row['name']:34s} {row['count']:5d} {row['total_s']:11.1f} "
              f"{share*100:7.2f}% {row['median']:10.3f} {row['p90']:10.3f}")
    print(f"报告：{out_dir / 'stage_timing_summary.md'}")
    cost = report["geometry_cost"]
    if cost:
        print(f"几何求值 {cost['geometry_evaluations_total']:.0f} 次，"
              f"不合法 {cost['invalid_fraction']*100:.1f}%，"
              f"单次 {cost['per_evaluation_ms']:.3f} ms")


if __name__ == "__main__":
    main()
