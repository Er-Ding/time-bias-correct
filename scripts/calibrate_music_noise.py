"""只读：用同一几何信道加独立新噪声，测量 MUSIC 峰估计误差（σ_θ、σ_L）。

真值存档里同时保存了无噪的 ``csi_geometric`` 与 ``injected_noise_std``，
因此可以原样重放噪声而不必重跑 Sionna。每一步都走定位主流程同一条
MUSIC 代码路径（prepare → spectrum → extract_local_music_peaks →
refine_music_peaks），保证测量的就是真实估计器的误差。

真值只用于把峰配对到真实路径，不参与谱计算。
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from time_bias_localization.config import DEFAULT_CONFIG, localization_config_view
from time_bias_localization.music_stage import get_music_computer
from time_bias_localization.music_subspace import SubspaceSelectionError, subspace_settings
from time_bias_localization.path_detection import detection_settings
from time_bias_localization.pipeline import _make_grids, _separation_bins
from time_bias_localization.signal import apply_common_delay_bias, extract_local_music_peaks
from time_bias_localization.spectrum_sampling import refine_music_peaks


ANGLE_MATCH_DEG = 3.0
DELAY_MATCH_NS = 5.0


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                     default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"无法序列化 {type(value).__name__}")


def wrap_deg(value):
    return (value + 180.0) % 360.0 - 180.0


def load_truth_rows(truth):
    """路径增益来自 path_coefficients：形状 (阵元, 路径)，每列是增益×导向矢量。

    各阵元幅度相同，因此取幅度即为该路径的增益，无需按交互数切片。
    """
    rows = []
    coefficients = truth["path_coefficients"]
    for index in np.flatnonzero(truth["retained_mask"]):
        column = np.abs(coefficients[:, index])
        gain = float(np.mean(column))
        rows.append({
            "path_index": int(index),
            "angle_deg": float(np.rad2deg(truth["aoa_local_rad"][index])),
            "delay_ns": float((truth["absolute_delays_s"][index] + truth["clock_bias_s"]) * 1e9),
            "reflections": int(truth["reflection_order"][index]),
            "diffractions": int(truth["diffraction_order"][index]),
            "path_gain": gain,
            "path_power": float(np.sum(column ** 2)),
        })
    for row in rows:
        others = [other for other in rows if other["path_index"] != row["path_index"]]
        row["nearest_other_angle_deg"] = (min(abs(wrap_deg(row["angle_deg"] - other["angle_deg"]))
                                              for other in others) if others else None)
        row["nearest_other_delay_ns"] = (min(abs(row["delay_ns"] - other["delay_ns"])
                                            for other in others) if others else None)
    return rows


def match_peak(angle_deg, delay_ns, rows, period_ns):
    """把峰配对到最近的真实路径；时延按子载波周期折回后比较。"""
    best, best_distance = None, None
    for row in rows:
        angle_error = wrap_deg(angle_deg - row["angle_deg"])
        if abs(angle_error) > ANGLE_MATCH_DEG:
            continue
        period_index = int(round((row["delay_ns"] - delay_ns) / period_ns))
        folded = row["delay_ns"] - period_index * period_ns
        delay_error = delay_ns - folded
        if abs(delay_error) > DELAY_MATCH_NS:
            continue
        distance = math.hypot(angle_error / ANGLE_MATCH_DEG, delay_error / DELAY_MATCH_NS)
        if best_distance is None or distance < best_distance:
            best, best_distance = (row, angle_error, delay_error, period_index), distance
    return best


def run_music(csi_observed, online, music_config, computer):
    """定位主流程同一条 MUSIC 路径；返回主要诊断。"""
    detector = detection_settings(music_config.get("path_detection", {}))
    selection = subspace_settings(music_config.get("subspace_selection", {}))
    automatic = selection["mode"] == "eigenvalue_threshold"
    aoa_grid, delay_grid = _make_grids(music_config)
    requested = int(music_config.get("signal_subspace_rank", int(music_config["num_paths"])))
    prepared = computer.prepare(
        csi_observed, subcarrier_frequencies_hz=online["subcarrier_frequencies_hz"],
        carrier_frequency_hz=float(online["carrier_frequency_hz"]),
        antenna_spacing_m=float(online["antenna_spacing_m"]),
        num_sources=requested, subspace_selection=selection,
        spatial_subarray_size=int(music_config["spatial_subarray_size"]),
        frequency_subarray_size=int(music_config["frequency_subarray_size"]),
        diagonal_loading=float(music_config["diagonal_loading"]))
    signal_rank = int(prepared.subspace_diagnostics["signal_rank"])
    spectrum = prepared.spectrum(aoa_grid_rad=aoa_grid, delay_grid_s=delay_grid)
    peak_limit = signal_rank if (automatic and not detector["enabled"]) else requested
    coarse = extract_local_music_peaks(
        spectrum, aoa_grid_rad=aoa_grid, delay_grid_s=delay_grid, max_peaks=peak_limit,
        minimum_relative_height=0.0, minimum_separation_bins=_separation_bins(music_config))
    sampled = refine_music_peaks(
        prepared, coarse, aoa_grid_rad=aoa_grid, delay_grid_s=delay_grid,
        bs_boresight_rad=0.0, settings=music_config["spectrum_sampling"], seed=0,
        minimum_angle_separation_rad=math.radians(float(music_config["min_angle_separation_deg"])),
        minimum_delay_separation_s=float(music_config["min_delay_separation_s"]),
        accepted_peak_centers=False)
    return {"signal_rank": signal_rank, "coarse_count": len(coarse),
            "nominal_peaks": list(sampled.refined_peaks),
            "nominal_source_indices": list(sampled.refined_peak_source_indices),
            "suppressed_count": len(sampled.diagnostics.get("suppressed_refined_peaks", [])),
            "maximum_response_correlation": None}


def calibrate_sample(source, sample_id, config, computer, repetitions, base_seed, output):
    sample_dir = source / "samples" / sample_id
    observation = json.loads((sample_dir / "observation.json").read_text())
    music_config = config["music"]
    with np.load(observation["artifacts"]["online_npz"]["path"], allow_pickle=False) as archive:
        online = {name: archive[name] for name in archive.files}
    with np.load(observation["artifacts"]["truth_npz"]["path"], allow_pickle=False) as archive:
        truth = {name: archive[name] for name in archive.files}
    period_ns = 1e9 / float(np.median(np.diff(online["subcarrier_frequencies_hz"])))
    rows = load_truth_rows(truth)
    noise_std = float(truth["injected_noise_std"])
    bias_s = float(truth["clock_bias_s"])
    frequencies = online["subcarrier_frequencies_hz"]
    geometric = truth["csi_geometric"]
    signal_rms = float(np.sqrt(np.mean(np.abs(geometric) ** 2)))
    noise_rms = float(np.sqrt(np.mean(np.abs(online["csi_observed"]) ** 2)))
    snr_db = 10.0 * math.log10(signal_rms ** 2 / noise_std ** 2) if noise_std else None

    # 对照：无噪输入无法建立噪声基准（特征值中位数参考失效），因此以
    # 低噪声对照替代，并把无噪失败本身记录下来。
    def control(level, label):
        try:
            csi = apply_common_delay_bias(geometric, frequencies, bias_s, noise_std=level, seed=0)
            stage = run_music(csi, online, music_config, computer)
        except SubspaceSelectionError as error:
            return {"label": label, "noise_std": level, "error": type(error).__name__,
                    "message": str(error), "nominal_count": 0, "records": [], "all_matched": False}
        records = []
        for peak in stage["nominal_peaks"]:
            match = match_peak(math.degrees(peak.aoa_rad), peak.delay_s * 1e9, rows, period_ns)
            records.append(None if match is None else {
                "angle_error_deg": match[1], "delay_error_ns": match[2],
                "path_index": match[0]["path_index"]})
        return {"label": label, "noise_std": level, "error": None, "message": None,
                "nominal_count": len(stage["nominal_peaks"]), "signal_rank": stage["signal_rank"],
                "records": records, "all_matched": bool(records) and all(
                    item is not None for item in records),
                "angle_rmse_deg": (float(np.sqrt(np.mean([item["angle_error_deg"] ** 2
                                                          for item in records if item is not None])))
                                   if any(item is not None for item in records) else None)}

    controls = [control(0.0, "noiseless"), control(0.1 * noise_std, "one_tenth_noise"),
                control(noise_std, "nominal_noise")]

    records, per_peak = [], {}
    for repetition in range(repetitions):
        seed = base_seed + repetition
        csi = apply_common_delay_bias(geometric, frequencies, bias_s, noise_std=noise_std, seed=seed)
        stage = run_music(csi, online, music_config, computer)
        matched, spurious, entries = 0, 0, []
        for peak in stage["nominal_peaks"]:
            angle_deg, delay_ns = math.degrees(peak.aoa_rad), peak.delay_s * 1e9
            match = match_peak(angle_deg, delay_ns, rows, period_ns)
            if match is None:
                spurious += 1
                entries.append({"matched": False, "angle_deg": angle_deg,
                                "delay_ns": delay_ns, "spectrum_value": float(peak.spectrum_value)})
                continue
            row, angle_error, delay_error, period_index = match
            matched += 1
            length_error_m = delay_error * 1e-9 * 299792458.0
            entries.append({"matched": True, "path_index": row["path_index"],
                            "angle_error_deg": angle_error, "delay_error_ns": delay_error,
                            "length_error_m": length_error_m,
                            "angle_error_rad": math.radians(angle_error),
                            "path_gain": row["path_gain"], "path_power": row["path_power"],
                            "nearest_other_angle_deg": row["nearest_other_angle_deg"],
                            "nearest_other_delay_ns": row["nearest_other_delay_ns"],
                            "reflections": row["reflections"], "diffractions": row["diffractions"],
                            "spectrum_value": float(peak.spectrum_value),
                            "matched_path_angle_deg": row["angle_deg"],
                            "matched_path_delay_ns": row["delay_ns"]})
            per_peak.setdefault(row["path_index"], []).append(entries[-1])
        records.append({"repetition": repetition, "seed": seed,
                        "signal_rank": stage["signal_rank"],
                        "coarse_count": stage["coarse_count"],
                        "nominal_count": len(stage["nominal_peaks"]),
                        "matched_count": matched, "spurious_count": spurious,
                        "suppressed_count": stage["suppressed_count"],
                        "entries": entries})
    return {
        "sample_id": sample_id, "repetitions": repetitions,
        "noise_std": noise_std, "snr_db": snr_db,
        "signal_rms": signal_rms, "observed_rms": noise_rms,
        "delay_period_ns": period_ns,
        "truth_path_count": len(rows), "truth_paths": rows,
        "controls": controls,
        "records": records,
        "per_path": {str(key): {**_summarize([item["angle_error_deg"] for item in value],
                                             [item["length_error_m"] for item in value]),
                                "reflections": value[0]["reflections"],
                                "diffractions": value[0]["diffractions"],
                                "path_angle_deg": value[0]["matched_path_angle_deg"],
                                "path_delay_ns": value[0]["matched_path_delay_ns"],
                                "path_gain": value[0]["path_gain"],
                                "path_power": value[0]["path_power"],
                                "nearest_other_angle_deg": value[0]["nearest_other_angle_deg"]}
                     for key, value in sorted(per_peak.items())},
    }


def _summarize(angle_errors_deg, length_errors_m):
    angle = np.asarray(angle_errors_deg, float)
    length = np.asarray(length_errors_m, float)
    if not angle.size:
        return {"count": 0}
    return {"count": int(angle.size),
            "angle_rmse_deg": float(np.sqrt(np.mean(angle ** 2))),
            "angle_std_deg": float(np.std(angle, ddof=1)) if angle.size > 1 else None,
            "angle_rmse_rad": float(np.sqrt(np.mean(np.radians(angle) ** 2))),
            "length_rmse_m": float(np.sqrt(np.mean(length ** 2))),
            "length_std_m": float(np.std(length, ddof=1)) if length.size > 1 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", default="000015", help="逗号分隔，SAMPLE_ 后缀")
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("输出必须与只读输入分开")
    output.mkdir(parents=True, exist_ok=False)

    experiment = json.loads((source / "experiment.json").read_text())
    merged = deepcopy(DEFAULT_CONFIG)
    for name, value in experiment["generation_config"].items():
        merged[name] = value
    config = localization_config_view(merged)
    compute = config.get("compute", {})
    computer = get_music_computer("numpy", 0, int(compute.get("batch_size", 4)),
                                 int(compute.get("angle_chunk_size", 32)))

    cases = []
    for suffix in [value.strip() for value in args.samples.split(",") if value.strip()]:
        name = f"SAMPLE_{int(suffix):06d}"
        case = calibrate_sample(source, name, config, computer, args.repetitions, args.seed, output)
        cases.append(case)
        overall = _summarize(
            [item["angle_error_deg"] for record in case["records"] for item in record["entries"]
             if item["matched"]],
            [item["length_error_m"] for record in case["records"] for item in record["entries"]
             if item["matched"]])
        print(f"[标定] {name}：真值路径 {case['truth_path_count']}，"
              f"角 RMSE {overall['angle_rmse_deg']:.5f}°（{overall['angle_rmse_rad']:.3e} rad），"
              f"长度 RMSE {overall['length_rmse_m']:.5f} m，峰值 {overall['count']}", flush=True)
        write_json(output / "calibration.json", {"cases": cases})
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(source), "repetitions": args.repetitions, "seed": args.seed,
        "configured_angle_scale_deg": 1.0, "configured_length_scale_m": 0.75,
        "samples": [case["sample_id"] for case in cases],
        "controls": {case["sample_id"]: [{key: row[key] for key in
                                          ("label", "noise_std", "nominal_count", "all_matched", "error")}
                                         for row in case["controls"]] for case in cases},
        "spurious_counts": {case["sample_id"]: sum(r["spurious_count"] for r in case["records"])
                            for case in cases},
        "matched_counts": {case["sample_id"]: sum(r["matched_count"] for r in case["records"])
                           for case in cases},
        "uses_truth_only_for_peak_association": True,
        "csi_regenerated_from_saved_geometric_channel": True,
        "scope": "estimator_noise_only; map_error_and_model_error_are_zero_in_this_simulation",
    }
    write_json(output / "summary.json", summary)
    write_markdown(output / "summary.md", summary, cases)
    write_json(output / "provenance.json", {
        "input_files_sha256": {str(source / "samples" / case["sample_id"] / "observation.json"):
                               digest(source / "samples" / case["sample_id"] / "observation.json")
                               for case in cases},
        "code_changed": False, "localization_rerun": False})
    print(f"[完成] {output / 'summary.md'}", flush=True)


def write_markdown(path, summary, cases):
    lines = ["# MUSIC 峰估计误差标定", "",
             f"输入：`{summary['input_root']}`。每条重复用同一几何信道加独立新噪声，"
             f"共 {summary['repetitions']} 次。", "",
             "走定位主流程同一条 MUSIC 路径，所以测的是真实估计器的误差，不是解析近似。", "",
             f"当前配置的工程尺度：角度 {summary['configured_angle_scale_deg']:g}°，"
             f"长度 {summary['configured_length_scale_m']:g} m。", ""]
    for case in cases:
        matched = [item for record in case["records"] for item in record["entries"] if item["matched"]]
        overall = _summarize([item["angle_error_deg"] for item in matched],
                             [item["length_error_m"] for item in matched])
        lines.extend([f"## {case['sample_id']}", "",
                      f"- 真值路径 {case['truth_path_count']} 条；每次重复提取的峰数 "
                      f"{_counts([r['nominal_count'] for r in case['records']])}",
                      f"- subspace 信号秩 {_counts([r['signal_rank'] for r in case['records']])}",
                      f"- 配到真实路径的峰 {sum(r['matched_count'] for r in case['records'])}，"
                      f"未配上的峰 {sum(r['spurious_count'] for r in case['records'])}",
                      f"- 去重抑制的峰 {sum(r['suppressed_count'] for r in case['records'])}", "",
                      "| 量 | 实测 RMSE | 当前配置尺度 | 偏大倍数 |", "| --- | ---: | ---: | ---: |",
                      f"| 角度 | {overall['angle_rmse_deg']:.5f}° = {overall['angle_rmse_rad']:.3e} rad | "
                      f"1° = 1.745e-02 rad | {1.745e-02 / overall['angle_rmse_rad']:.2f}× |",
                      f"| 长度 | {overall['length_rmse_m']:.5f} m | 0.75 m | "
                      f"{0.75 / overall['length_rmse_m']:.2f}× |", "",
                      "噪声对照（同一几何信道，只改注入噪声强度）：", "",
                      "| 对照 | 注入噪声 | 提取峰数 | 全部配对 | 角度 RMSE/° | 异常 |",
                      "| --- | ---: | ---: | --- | ---: | --- |"])
        for row in case["controls"]:
            lines.append(f"| {row['label']} | {row['noise_std']:.3e} | {row['nominal_count']} | "
                         f"{row['all_matched']} | "
                         f"{'' if row.get('angle_rmse_deg') is None else format(row['angle_rmse_deg'], '.5f')} | "
                         f"{row['error'] or ''} |")
        lines.extend(["", "按真实路径逐条：", "",
                      "| 真实路径 | 反射 | 绕射 | 配对次数 | 角度 RMSE/° | 长度 RMSE/m |",
                      "| ---: | ---: | ---: | ---: | ---: | ---: |"])
        for key, value in case["per_path"].items():
            count = value["count"]
            if not count:
                continue
            lines.append(f"| {key} | {value.get('reflections', '')} | {value.get('diffractions', '')} | "
                         f"{count} | {value['angle_rmse_deg']:.5f} | {value['length_rmse_m']:.5f} |")
        lines.append("")
    lines.extend(["## 边界", "", summary["scope"], "",
                  "真值只用于把峰配对到真实路径；谱计算不使用真值。",
                  "未重新生成 CSI，未修改原实验，未改动定位器。", ""])
    path.write_text("\n".join(lines) + "\n")


def _counts(values):
    return dict(sorted(Counter(values).items()))


if __name__ == "__main__":
    main()
