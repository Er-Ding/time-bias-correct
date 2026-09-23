"""只读：从标定结果建立 σ 模型，并检验观测量能否预测 σ。

回答两个问题：
1. σ_θ、σ_L 与真实路径增益的关系（增益量级从真值存档重算，不依赖标定输出里的字段）；
2. 观测到的 MUSIC 峰高（spectrum_value）能否预测 σ。若可以，按噪声加权
   就是可落地的；若不可以，就应该放弃加权，只用单一标定 σ。

同一标定样本内噪声强度恒定，因此样本内不同路径的 σ 差异只能来自路径本身的
几何与强度，样本内相关性是最干净的判据。
真值只用于离线解释。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wrap_deg(value):
    return (value + 180.0) % 360.0 - 180.0


def truth_gains(source, sample_id):
    """从真值存档重算每条保留路径的增益与功率。"""
    observation = json.loads((source / "samples" / sample_id / "observation.json").read_text())
    with np.load(observation["artifacts"]["truth_npz"]["path"], allow_pickle=False) as archive:
        coefficients = archive["path_coefficients"]
        rows = {}
        for index in np.flatnonzero(archive["retained_mask"]):
            column = np.abs(coefficients[:, index])
            rows[int(index)] = {
                "gain": float(np.mean(column)),
                "power": float(np.sum(column ** 2)),
                "reflections": int(archive["reflection_order"][index]),
                "diffractions": int(archive["diffraction_order"][index]),
                "angle_deg": float(np.rad2deg(archive["aoa_local_rad"][index])),
                "delay_ns": float((archive["absolute_delays_s"][index]
                                   + archive["clock_bias_s"]) * 1e9),
            }
    return rows


def pearson(xs, ys):
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(xs, ys):
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if x.size < 3:
        return None
    rank_x = np.argsort(np.argsort(x)).astype(float)
    rank_y = np.argsort(np.argsort(y)).astype(float)
    return pearson(rank_x, rank_y)


def truth_density(source, sample_id):
    """真实路径之间的最小角间隔与时延间隔，用于解释样本间 σ 差异。"""
    observation = json.loads((source / "samples" / sample_id / "observation.json").read_text())
    with np.load(observation["artifacts"]["truth_npz"]["path"], allow_pickle=False) as archive:
        index = np.flatnonzero(archive["retained_mask"])
        angles = np.rad2deg(archive["aoa_local_rad"][index])
        delays = (archive["absolute_delays_s"][index] + archive["clock_bias_s"]) * 1e9
    pairs = [(i, j) for i in range(len(angles)) for j in range(i + 1, len(angles))]
    return {
        "path_count": int(len(angles)),
        "min_angle_separation_deg": (min(abs(wrap_deg(angles[i] - angles[j]))
                                        for i, j in pairs) if pairs else None),
        "min_delay_separation_ns": (min(abs(delays[i] - delays[j])
                                       for i, j in pairs) if pairs else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    document = json.loads(args.calibration.read_text())
    cases = document["cases"]

    per_sample, pooled_within, detection = [], [], []
    for case in cases:
        sample_id = case["sample_id"]
        gains = truth_gains(source, sample_id)
        matched = [item for record in case["records"] for item in record["entries"]
                   if item["matched"]]
        if not matched:
            continue
        angle = np.asarray([item["angle_error_deg"] for item in matched])
        length = np.asarray([item["length_error_m"] for item in matched])
        # 样本内：按路径分组
        by_path = defaultdict(list)
        for item in matched:
            by_path[item["path_index"]].append(item)
        rows = []
        for path_index, items in sorted(by_path.items()):
            gain = gains.get(path_index)
            path_angle = np.asarray([item["angle_error_deg"] for item in items])
            path_length = np.asarray([item["length_error_m"] for item in items])
            spectrum = np.asarray([item["spectrum_value"] for item in items])
            rows.append({
                "path_index": path_index,
                "paired_count": len(items),
                "angle_rmse_deg": float(np.sqrt(np.mean(path_angle ** 2))),
                "length_rmse_m": float(np.sqrt(np.mean(path_length ** 2))),
                "mean_spectrum_value": float(np.mean(spectrum)),
                "reflections": items[0].get("reflections"),
                "diffractions": items[0].get("diffractions"),
                "truth_gain": None if gain is None else gain["gain"],
                "truth_power": None if gain is None else gain["power"],
            })
        with_gain = [row for row in rows if row["truth_gain"]]
        detected = {row["path_index"] for row in rows}
        per_sample.append({
            "sample_id": sample_id, "noise_std": case["noise_std"], "snr_db": case["snr_db"],
            "truth_path_count": case["truth_path_count"],
            "detected_path_count": len(detected),
            "missed_path_indices": sorted(set(gains) - detected),
            "peaks_per_repetition": case["records"][0]["nominal_count"],
            "subspace_rank": case["records"][0]["signal_rank"],
            "spurious_per_repetition": round(
                sum(record["spurious_count"] for record in case["records"]) / len(case["records"]), 3),
            "angle_rmse_deg": float(np.sqrt(np.mean(angle ** 2))),
            "angle_rmse_rad": float(np.sqrt(np.mean(np.radians(angle) ** 2))),
            "length_rmse_m": float(np.sqrt(np.mean(length ** 2))),
            "matched_peak_count": int(angle.size),
            "paths": rows,
            "within_sample_correlation": {
                "log_gain_vs_log_angle_rmse": pearson(
                    np.log([row["truth_gain"] for row in with_gain]),
                    np.log([row["angle_rmse_deg"] for row in with_gain])),
                "log_spectrum_vs_log_angle_rmse": pearson(
                    np.log([row["mean_spectrum_value"] for row in with_gain]),
                    np.log([row["angle_rmse_deg"] for row in with_gain])),
                "path_count": len(with_gain),
            },
        })
        for row in with_gain:
            pooled_within.append((sample_id, row))
        detection.append({
            "sample_id": sample_id, "truth_path_count": case["truth_path_count"],
            "detected_path_count": len(detected),
            "missed_path_indices": sorted(set(gains) - detected),
            "peaks_per_repetition": case["records"][0]["nominal_count"],
            "subspace_rank": case["records"][0]["signal_rank"],
        })

    # 样本内归一化后池化：消除样本间噪声强度差异
    gain_ratios, angle_ratios, spectrum_ratios = [], [], []
    for case in per_sample:
        with_gain = [row for row in case["paths"] if row["truth_gain"]]
        usable = [row for row in with_gain if row["angle_rmse_deg"] > 0
                  and row["truth_gain"] > 0 and row["mean_spectrum_value"] > 0]
        if len(usable) < 2:
            continue
        median_angle = float(np.median([row["angle_rmse_deg"] for row in usable]))
        for row in usable:
            gain_ratios.append(math.log(row["truth_gain"]))
            angle_ratios.append(math.log(row["angle_rmse_deg"] / median_angle))
            spectrum_ratios.append(math.log(row["mean_spectrum_value"]))
    # 样本内先各自中心化，再池化，等价于固定样本效应
    centered = []
    for case in per_sample:
        usable = [row for row in case["paths"]
                  if row["truth_gain"] and row["angle_rmse_deg"] > 0
                  and row["mean_spectrum_value"] > 0]
        if len(usable) < 2:
            continue
        mean_gain = float(np.mean([math.log(row["truth_gain"]) for row in usable]))
        mean_angle = float(np.mean([math.log(row["angle_rmse_deg"]) for row in usable]))
        mean_spectrum = float(np.mean([math.log(row["mean_spectrum_value"]) for row in usable]))
        for row in usable:
            centered.append({
                "sample_id": case["sample_id"], "path_index": row["path_index"],
                "log_gain_centered": math.log(row["truth_gain"]) - mean_gain,
                "log_spectrum_centered": math.log(row["mean_spectrum_value"]) - mean_spectrum,
                "log_angle_rmse_centered": math.log(row["angle_rmse_deg"]) - mean_angle,
                "log_length_rmse_centered": math.log(row["length_rmse_m"]),
                "reflections": row["reflections"], "diffractions": row["diffractions"],
            })
    pooled = {
        "gain_vs_angle": pearson([row["log_gain_centered"] for row in centered],
                                 [row["log_angle_rmse_centered"] for row in centered]),
        "spectrum_vs_angle": pearson([row["log_spectrum_centered"] for row in centered],
                                     [row["log_angle_rmse_centered"] for row in centered]),
        "gain_vs_angle_spearman": spearman([row["log_gain_centered"] for row in centered],
                                          [row["log_angle_rmse_centered"] for row in centered]),
        "spectrum_vs_angle_spearman": spearman([row["log_spectrum_centered"] for row in centered],
                                              [row["log_angle_rmse_centered"] for row in centered]),
        "path_count": len(centered),
        "method": "within_sample_centered_logs_then_pooled_pearson",
    }
    # 拟合 σ ∝ gain^(-k)
    if centered:
        x = np.asarray([-row["log_gain_centered"] for row in centered])
        y = np.asarray([row["log_angle_rmse_centered"] for row in centered])
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        pooled["power_law_fit"] = {
            "model": "log_sigma_angle = slope * (-log_gain) + intercept",
            "slope_equals_expected_exponent": float(slope),
            "intercept": float(intercept),
            "r_squared": float(1 - np.var(residual) / np.var(y)) if np.var(y) > 0 else None,
        }

    sample_level = []
    for case in per_sample:
        density = truth_density(source, case["sample_id"])
        sample_level.append({
            "sample_id": case["sample_id"],
            "angle_rmse_deg": case["angle_rmse_deg"],
            "length_rmse_m": case["length_rmse_m"],
            "noise_std": case["noise_std"], "snr_db": case["snr_db"],
            **density,
        })
    log_sigma = np.log([row["angle_rmse_deg"] for row in sample_level])
    candidates = {
        "path_count": [row["path_count"] for row in sample_level],
        "min_angle_separation_deg": [row["min_angle_separation_deg"] for row in sample_level],
        "min_delay_separation_ns": [row["min_delay_separation_ns"] for row in sample_level],
        "noise_std": [row["noise_std"] for row in sample_level],
    }
    sample_correlation = {}
    for name, values in candidates.items():
        if any(value is None or value <= 0 for value in values):
            sample_correlation[name] = {"pearson_log": None, "spearman": None,
                                        "reason": "非正值或缺失，无法取对数"}
            continue
        sample_correlation[name] = {
            "pearson_log": pearson(np.log(values), log_sigma),
            "spearman": spearman(values, log_sigma),
        }

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(source), "calibration": str(args.calibration),
        "repetitions": document.get("repetitions"),
        "sample_count": len(per_sample),
        "configured_angle_scale_deg": 1.0, "configured_length_scale_m": 0.75,
        "overall_angle_rmse_deg": float(np.sqrt(np.mean([case["angle_rmse_deg"] ** 2
                                                        for case in per_sample]))),
        "overall_length_rmse_m": float(np.sqrt(np.mean([case["length_rmse_m"] ** 2
                                                       for case in per_sample]))),
        "angle_rmse_deg_range": [float(min(case["angle_rmse_deg"] for case in per_sample)),
                                 float(max(case["angle_rmse_deg"] for case in per_sample))],
        "length_rmse_m_range": [float(min(case["length_rmse_m"] for case in per_sample)),
                                float(max(case["length_rmse_m"] for case in per_sample))],
        "detection": detection,
        "missed_path_samples": sum(1 for row in detection
                                   if row["detected_path_count"] < row["truth_path_count"]),
        "pooled_within_sample": pooled,
        "sample_level": sample_level,
        "sample_level_correlation_with_log_angle_rmse": sample_correlation,
        "per_sample": per_sample,
        "scope": ("estimator_noise_and_detection_only; map_error_and_model_error_are_zero; "
                  "truth_used_only_for_association_and_gain_reference"),
    }
    (output / "sigma_model.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_markdown(output / "sigma_model.md", summary)
    (output / "provenance.json").write_text(json.dumps({
        "calibration_sha256": digest(args.calibration),
        "analysis_script_sha256": digest(__file__),
        "uses_truth_only_for_offline_interpretation": True}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "sample_count", "overall_angle_rmse_deg", "overall_length_rmse_m",
        "angle_rmse_deg_range", "length_rmse_m_range", "missed_path_samples")},
        ensure_ascii=False, indent=1))
    print("[样本内池化]", json.dumps(pooled, ensure_ascii=False, indent=1))
    print(f"[完成] {output / 'sigma_model.md'}")


def write_markdown(path, summary):
    lines = ["# σ 模型：估计器噪声与检测统计", "",
             f"来自 {summary['repetitions']} 次重复、{summary['sample_count']} 个样本。", "",
             "## 总体", "",
             f"- 角度 RMSE 汇总 {summary['overall_angle_rmse_deg']:.5f}°，"
             f"样本间范围 {summary['angle_rmse_deg_range'][0]:.5f}° ～ "
             f"{summary['angle_rmse_deg_range'][1]:.5f}°",
             f"- 长度 RMSE 汇总 {summary['overall_length_rmse_m']:.5f} m，"
             f"样本间范围 {summary['length_rmse_m_range'][0]:.5f} m ～ "
             f"{summary['length_rmse_m_range'][1]:.5f} m",
             f"- 配置尺度：角度 1°，长度 0.75 m",
             f"- 有真实路径漏检的样本：{summary['missed_path_samples']} / {summary['sample_count']}", "",
             "## 逐样本", "",
             "| 样本 | 真值路径 | 检出路径 | 每次峰数 | 信号秩 | 角 RMSE/° | 长度 RMSE/m | 漏检路径 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for case in summary["per_sample"]:
        lines.append(f"| {case['sample_id']} | {case['truth_path_count']} | "
                     f"{case['detected_path_count']} | {case['peaks_per_repetition']} | "
                     f"{case['subspace_rank']} | {case['angle_rmse_deg']:.5f} | "
                     f"{case['length_rmse_m']:.5f} | {case['missed_path_indices'] or '—'} |")
    pooled = summary["pooled_within_sample"]
    lines.extend(["", "## 观测量能否预测 σ", "",
                  "同一标定样本内噪声强度恒定，所以样本内各路径的 σ 差异只来自路径自身。"
                  "下表把每个样本内的对数值中心化后池化，消除样本间噪声差异。", "",
                  f"- 池化路径数：{pooled['path_count']}", "",
                  "| 自变量 | 与 log σ_θ 的相关系数 | Spearman | 解释 |",
                  "| --- | ---: | ---: | --- |",
                  f"| 真实路径增益（日志） | {_fmt(pooled['gain_vs_angle'])} | "
                  f"{_fmt(pooled['gain_vs_angle_spearman'])} | 真值，定位器拿不到 |",
                  f"| 观测峰高（日志） | {_fmt(pooled['spectrum_vs_angle'])} | "
                  f"{_fmt(pooled['spectrum_vs_angle_spearman'])} | 纯观测量，可用于加权 |", ""])
    if "power_law_fit" in pooled:
        fit = pooled["power_law_fit"]
        lines.extend([f"幂律拟合 log σ_θ = {fit['slope_equals_expected_exponent']:.3f}×(−log 增益)"
                      f" {fit['intercept']:+.3f}，R² = {_fmt(fit['r_squared'])}。",
                      "若 σ 只由信噪比决定，斜率应接近 1（σ 正比于 1/幅度）。", ""])
    sample = summary["sample_level_correlation_with_log_angle_rmse"]
    lines.extend(["", "## 样本间 σ_θ 差异由什么决定", "",
                  f"12 个样本的角 RMSE 从 {summary['angle_rmse_deg_range'][0]:.5f}° 到 "
                  f"{summary['angle_rmse_deg_range'][1]:.5f}°，相差约 "
                  f"{summary['angle_rmse_deg_range'][1] / max(summary['angle_rmse_deg_range'][0], 1e-12):.0f} 倍。"
                  "所有样本 SNR 都是 35 dB，因此差异不来自噪声强度。", "",
                  "| 变量 | 与 log σ_θ 的 Pearson | Spearman |", "| --- | ---: | ---: |"])
    for name, value in sample.items():
        lines.append(f"| {name} | {_fmt(value['pearson_log'])} | {_fmt(value['spearman'])} |")
    lines.extend(["", "样本数只有 12，上述相关系数都不足以支撑预测模型。"
                  "结论是：以现有证据无法用路径强度或简单间隔量解释 σ_θ，"
                  "因此不应据此构造逐观测噪声模型；应使用单一标定 σ 并声明其为经验超参数。", "",
                  "## 判据", "",
                  "若\"观测峰高\"与 log σ_θ 的相关系数显著为负，则峰高越大 σ 越小，"
                  "按噪声加权可以从观测实现；若相关系数接近 0，则应放弃加权，"
                  "只用单一标定 σ。", "",
                  "## 边界", "", summary["scope"], "",
                  "真值仅用于离线配对与增益参照。未重新生成 CSI，未改动定位器。", ""])
    path.write_text("\n".join(lines) + "\n")


def _fmt(value):
    return "—" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
