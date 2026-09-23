"""分析对照实验：伪峰剔除的效果、幅值加权的效果、以及安慰剂组。

三次运行只差一个开关，基准组相同：
  剔除效果 = 第 2 组 − 第 1 组
  加权效果 = 第 3 组 − 第 1 组
安慰剂组（不含边界伪峰）用来检验剔除实现是否引入了无关副作用。
真值只用于离线误差统计。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def describe(values):
    array = np.asarray([v for v in values if v is not None], float)
    if not array.size:
        return {"count": 0}
    return {"count": int(array.size), "median": float(np.median(array)),
            "mean": float(array.mean()), "p90": float(np.percentile(array, 90)),
            "max": float(array.max())}


def load(report):
    result_path = report / "results.json"
    results = json.loads(result_path.read_text())["results"] if result_path.exists() else []
    plan = json.loads((report / "plan.json").read_text())
    truth = {sid: {**row, "true_position_m": row["true_position_m"]}
             for sid, row in plan["records"].items()}
    index = {}
    for row in results:
        index[(row["group"], row["arm"], row["sample_id"])] = row
    return index, truth, plan


def error_of(row, truth):
    if row.get("mu_m") is None or row["sample_id"] not in truth:
        return None, None
    position = np.asarray(row["mu_m"], float)
    true_position = np.asarray(truth[row["sample_id"]]["true_position_m"], float)
    position_error = float(np.linalg.norm(position - true_position))
    bias_error = None
    if row.get("clock_bias_s") is not None:
        bias_error = abs(row["clock_bias_s"] * 1e9
                         - truth[row["sample_id"]]["true_clock_bias_ns"])
    return position_error, bias_error


def summarize(index, truth, group, arm, sample_ids):
    rows = [index[(group, arm, sid)] for sid in sample_ids if (group, arm, sid) in index]
    successes = [row for row in rows if row["status"] == "success"]
    errors = [error_of(row, truth) for row in successes]
    pending = len(sample_ids) - len(rows)
    counts = Counter(row["status"] for row in rows)
    if pending:
        counts["pending"] = pending
    return {
        "tasks": len(sample_ids), "completed": len(rows), "pending": pending,
        "success": len(successes),
        "success_rate": len(successes) / max(len(sample_ids), 1),
        "status_counts": dict(counts),
        "position_error_m": describe([e[0] for e in errors]),
        "bias_error_ns": describe([e[1] for e in errors]),
        "median_seconds": (float(np.median([row["processing_seconds"] for row in rows]))
                           if rows else None),
    }


def paired_compare(index, truth, group, arm_a, arm_b):
    """同名样本在两组下的配对比较。"""
    paired, only_a, only_b = [], 0, 0
    for (g, a, sample_id), row_a in index.items():
        if g != group or a != arm_a:
            continue
        row_b = index.get((g, arm_b, sample_id))
        if row_b is None:
            continue
        ok_a, ok_b = row_a["status"] == "success", row_b["status"] == "success"
        err_a, _ = error_of(row_a, truth)
        err_b, _ = error_of(row_b, truth)
        identical = (row_a["mu_m"] == row_b["mu_m"]
                     and row_a["clock_bias_s"] == row_b["clock_bias_s"]
                     and row_a["status"] == row_b["status"])
        if ok_a and not ok_b:
            only_a += 1
        elif ok_b and not ok_a:
            only_b += 1
        if ok_a and ok_b:
            paired.append({"sample_id": sample_id, "error_a_m": err_a, "error_b_m": err_b,
                           "delta_m": err_b - err_a, "identical": identical})
    deltas = [entry["delta_m"] for entry in paired]
    return {"group": group, "arm_a": arm_a, "arm_b": arm_b,
            "paired_success_count": len(paired),
            "only_a_success": only_a, "only_b_success": only_b,
            "identical_outputs": sum(entry["identical"] for entry in paired),
            "delta_position_error_m": describe(deltas),
            "improved": sum(1 for d in deltas if d < -1e-9),
            "worsened": sum(1 for d in deltas if d > 1e-9),
            "unchanged": sum(1 for d in deltas if abs(d) <= 1e-9),
            "pairs": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, output = args.report.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    index, truth, plan = load(report)
    groups = list(plan["selection"])
    arms = plan["arms"]

    table = {group: {arm: summarize(index, truth, group, arm, plan["selection"][group]) for arm in arms}
             for group in groups}
    comparisons = [paired_compare(index, truth, group, "B0", arm)
                   for arm in ("M", "W") if "B0" in arms and arm in arms for group in groups]
    threads = Counter(row["process_threads"] for row in index.values())
    summary = {
        "report": str(report), "sample_groups": {name: len(ids)
                                                 for name, ids in plan["selection"].items()},
        "group_meanings": {
            "excluded": "原本无位置输出的样本（筛选阶段整点排除）",
            "success_with_boundary": "含角度边界伪峰的成功样本",
            "success_without_boundary": "不含边界伪峰的成功样本（安慰剂对照）",
        },
        "arms": {arm: plan["arm_definitions"][arm] for arm in arms},
        "per_group_arm": table, "paired_comparisons": comparisons,
        "process_threads_histogram": dict(threads),
        "all_single_threaded": set(threads) == {1},
        "uses_truth_only_for_offline_error": True,
    }
    (output / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False,
                                                     indent=2, default=str) + "\n")
    write_markdown(output / "analysis.md", summary)
    print(json.dumps({k: summary[k] for k in ("all_single_threaded",)}, ensure_ascii=False))
    for group in groups:
        print(f"\n[{group}]")
        for arm in arms:
            row = table[group][arm]
            seconds = "未完成" if row["median_seconds"] is None else f"{row['median_seconds']:.0f} s"
            print(f"  {arm}: 成功 {row['success']}/{row['tasks']} "
                  f"({row['success_rate']*100:.1f}%)，未完成 {row['pending']}；位置误差中位 "
                  f"{row['position_error_m'].get('median', float('nan')):.4f} m  "
                  f"耗时中位 {seconds}")
    print(f"\n[完成] {output / 'analysis.md'}")


def write_markdown(path, summary):
    table, comparisons = summary["per_group_arm"], summary["paired_comparisons"]
    lines = ["# 对照实验分析", "",
             "同一份代码与配置，每次只开一个开关。第 1 组是两次比较共用的基准。", "",
             f"- 工作进程线程数分布 {summary['process_threads_histogram']}"
             f"（全部单线程：{summary['all_single_threaded']}）", "",
             "## 样本分组", ""]
    for name, count in summary["sample_groups"].items():
        lines.append(f"- **{name}**（{count} 个）：{summary['group_meanings'][name]}")
    lines.extend(["", "## 各组结果", "",
                  "成功率分母是计划任务数；未完成任务单独计数，耗时仅统计已返回结果。", "",
                  "| 分组 | 配置 | 成功数/计划数 | 未完成 | 成功率 | 位置误差中位/m | 位置误差 p90/m | 偏置误差中位/ns | 耗时中位/s |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for group, group_table in table.items():
        for arm, row in group_table.items():
            pos, bias = row["position_error_m"], row["bias_error_ns"]
            seconds = "未完成" if row["median_seconds"] is None else f"{row['median_seconds']:.0f}"
            lines.append(f"| {group} | {arm} | {row['success']}/{row['tasks']} | "
                         f"{row['pending']} | {row['success_rate']*100:.1f}% | "
                         f"{pos.get('median', float('nan')):.4f} | "
                         f"{pos.get('p90', float('nan')):.4f} | "
                         f"{bias.get('median', float('nan')):.4f} | "
                         f"{seconds} |")
    lines.extend(["", "## 逐样本配对比较", "",
                  "同名样本在两组下的配对；`仅在基准成功`/`仅在改动组成功` 计新增或丢失的定位。",
                  "", "| 分组 | 比较 | 配对成功数 | 仅基准成功 | 仅改动组成功 | 输出完全相同 | 误差改善 | 误差变差 | 误差不变 | Δ误差中位/m |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in comparisons:
        delta = row["delta_position_error_m"]
        lines.append(f"| {row['group']} | {row['arm_a']}→{row['arm_b']} | "
                     f"{row['paired_success_count']} | {row['only_a_success']} | "
                     f"{row['only_b_success']} | {row['identical_outputs']} | "
                     f"{row['improved']} | {row['worsened']} | {row['unchanged']} | "
                     f"{delta.get('median', float('nan')):+.5f} |")
    lines.extend(["", "## 怎么读", "",
                  "**剔除伪峰**看 `B0→M`：主战场是 `excluded` 组；`success_without_boundary` "
                  "是不含边界伪峰的安慰剂组，它若出现明显变化，说明剔除实现引入了无关副作用。", "",
                  "**幅值加权**看 `B0→W`：三组都应关注，尤其是位置误差的配对变化。", "",
                  "## 边界", "",
                  "真值仅用于离线误差统计，不参与任何求解或选择。",
                  "两个改动各自只开一个开关，但幅值加权同时改变了 5σ 接受门限的口径，"
                  "因此其效果不能完全归因于权重本身。", ""])
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
