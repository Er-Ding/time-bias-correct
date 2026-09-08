#!/usr/bin/env python3
"""相同在线 CSI、种子和精度下的 CPU/GPU 对照，所有结果写入新目录。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback


def _write_json(path: Path, value) -> None:
    def encode(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.write_text(
        json.dumps(value, default=encode, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n", encoding="utf-8"
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_signature(candidate: dict) -> tuple:
    metadata = candidate.get("metadata", candidate)
    return (
        candidate["observation_id"],
        candidate.get("candidate_id", candidate.get("sample_id")),
        metadata.get("topology_id"),
        tuple(metadata.get("reflection_wall_ids", [])),
        metadata.get("raw_count"),
        tuple(metadata.get("source_sample_ids", [])),
    )


def _candidate_signatures(candidates: list[dict]) -> list[tuple]:
    return sorted((_candidate_signature(item) for item in candidates), key=repr)


def _discrete_signature(folder: Path) -> dict:
    peaks = _read_json(folder / "music_peaks.json")
    bootstrap = _read_json(folder / "bootstrap_diagnostics.json")
    result = _read_json(folder / "localization_result.json")
    return {
        "nominal_peak_indices": [
            [item["aoa_index"], item["delay_index"]] for item in peaks["nominal"]
        ],
        "peak_associations": [
            {key: item[key] for key in (
                "nominal_to_sample_index", "unmatched_sample_indices",
                "missed_nominal_indices", "valid_sample_indices",
            )}
            for item in peaks["associations"]
        ],
        "perturbed_observation_samples": peaks["perturbed_observation_samples"],
        "raw_candidates": _candidate_signatures(
            _read_json(folder / "raw_reverse_candidates.json")
        ),
        "first_clusters": _candidate_signatures(
            _read_json(folder / "clustered_candidates.json")
        ),
        "central_selected": _candidate_signatures(
            list(result["central_selected_candidates"].values())
        ),
        "perturbations": [
            {
                "repetition": item["repetition"],
                "solved": item["solved"],
                "raw_candidates": _candidate_signatures(item.get("raw_reverse_candidates", [])),
                "clusters": _candidate_signatures(item.get("clustered_candidates", [])),
                "selected_topologies": item.get("selected_topologies", {}),
            }
            for item in bootstrap
        ],
    }


def compare_localizations(cpu_root: Path, gpu_root: Path, *, position_atol_m: float,
                          bias_atol_ns: float) -> dict:
    """离散峰、候选和每次扰动都必须一致；不能仅凭最终位置接近判定通过。"""
    import numpy as np

    cpu_folder, gpu_folder = cpu_root / "localization", gpu_root / "localization"
    cpu = _read_json(cpu_folder / "localization_result.json")
    gpu = _read_json(gpu_folder / "localization_result.json")
    cpu_signature, gpu_signature = _discrete_signature(cpu_folder), _discrete_signature(gpu_folder)
    checks = {key: cpu_signature[key] == gpu_signature[key] for key in cpu_signature}
    position_difference = float(np.linalg.norm(np.asarray(cpu["mu_m"]) - gpu["mu_m"]))
    bias_difference = float((gpu["clock_bias_s"] - cpu["clock_bias_s"]) * 1e9)
    central_position_difference = float(np.linalg.norm(
        np.asarray(cpu["central_solution"]["mu_m"]) - gpu["central_solution"]["mu_m"]
    ))
    central_bias_difference = float((
        gpu["central_solution"]["clock_bias_s"] - cpu["central_solution"]["clock_bias_s"]
    ) * 1e9)
    checks.update({
        "position_within_tolerance": position_difference <= position_atol_m,
        "bias_within_tolerance": abs(bias_difference) <= bias_atol_ns,
        "central_position_within_tolerance": central_position_difference <= position_atol_m,
        "central_bias_within_tolerance": abs(central_bias_difference) <= bias_atol_ns,
    })
    with np.load(cpu_folder / "music_spectrum.npz") as archive:
        cpu_spectrum = archive["spectrum"]
    with np.load(gpu_folder / "music_spectrum.npz") as archive:
        gpu_spectrum = archive["spectrum"]
    difference = np.abs(gpu_spectrum - cpu_spectrum)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "position_difference_m": position_difference,
        "gpu_minus_cpu_bias_ns": bias_difference,
        "central_position_difference_m": central_position_difference,
        "central_gpu_minus_cpu_bias_ns": central_bias_difference,
        "spectrum_max_absolute_difference": float(difference.max()),
        "spectrum_relative_l2_difference": float(np.linalg.norm(difference) / np.linalg.norm(cpu_spectrum)),
        "tolerances": {"position_atol_m": position_atol_m, "bias_atol_ns": bias_atol_ns},
        "cpu_nominal_peaks": _read_json(cpu_folder / "music_peaks.json")["nominal"],
        "gpu_nominal_peaks": _read_json(gpu_folder / "music_peaks.json")["nominal"],
        "mismatched_discrete_stages": {
            key: {"cpu": cpu_signature[key], "cuda": gpu_signature[key]}
            for key in cpu_signature if not checks[key]
        },
    }


def _spectrum_arguments(config: dict, measurement) -> dict:
    from time_bias_localization.pipeline import _make_grids

    music = config["music"]
    angles, delays = _make_grids(music)
    return {
        "subcarrier_frequencies_hz": measurement.subcarrier_frequencies_hz,
        "carrier_frequency_hz": measurement.carrier_frequency_hz,
        "antenna_spacing_m": measurement.antenna_spacing_m,
        "aoa_grid_rad": angles,
        "delay_grid_s": delays,
        "num_sources": music.get("signal_subspace_rank", music["num_paths"]),
        "spatial_subarray_size": music["spatial_subarray_size"],
        "frequency_subarray_size": music["frequency_subarray_size"],
        "diagonal_loading": music["diagonal_loading"],
    }


def _synchronize(backend: str, device_id: int) -> None:
    if backend == "cuda":
        import cupy as cp
        cp.cuda.Device(device_id).synchronize()


def _load_inputs(input_root: Path, config_path: Path | None):
    from time_bias_localization.config import load_localization_config
    from time_bias_localization.provenance import capture_file, load_generation_manifest, verify_generation_artifact
    from time_bias_localization.data import load_online_measurement_bytes

    config_path = config_path or input_root / "localization.yaml"
    config = load_localization_config(config_path)
    manifest_path = input_root / "generation_manifest.json"
    manifest, _, _ = load_generation_manifest(manifest_path)
    scene_path = Path(manifest["artifact_hashes"]["scene_json"]["path"])
    online_path = Path(manifest["artifact_hashes"]["online_measurement"]["path"])
    online_capture = capture_file(online_path)
    verify_generation_artifact(manifest, "scene_json", capture_file(scene_path))
    verify_generation_artifact(manifest, "online_measurement", online_capture)
    measurement = load_online_measurement_bytes(online_capture.data, source_path=online_path)
    return config, manifest_path, manifest, scene_path, online_path, measurement


def _run_full(args, root: Path, config: dict, manifest_path: Path,
              scene_path: Path, online_path: Path, measurement) -> dict:
    from time_bias_localization.compute import ComputeSettings, MusicComputer
    from time_bias_localization.pipeline import localize
    from time_bias_localization.signal import music_2d_spectrum

    records = {}
    spectrum_arguments = _spectrum_arguments(config, measurement)
    for backend in ("numpy", "cuda"):
        label = "cpu" if backend == "numpy" else "cuda"
        run_root = root / label
        run_config = deepcopy(config)
        run_config["compute"] = {
            "backend": backend, "device_id": args.device_id,
            "batch_size": args.batch_size, "angle_chunk_size": args.angle_chunk_size,
        }
        run_config["output"]["root"] = str(run_root)
        record = {"status": "started", "output_root": str(run_root)}
        records[label] = record
        print(f"{label}: 相同输入的完整定位对照", flush=True)
        try:
            # CPU 与 GPU 都执行一次谱预热。新建定位计算器，避免跨后端缓存污染。
            start = time.perf_counter()
            if backend == "numpy":
                music_2d_spectrum(measurement.csi_observed, **spectrum_arguments)
                record["warmup_compute"] = {"backend": "numpy", "entrypoint": "signal.music_2d_spectrum"}
            else:
                computer = MusicComputer(ComputeSettings(**run_config["compute"]))
                computer.spectrum(measurement.csi_observed, **spectrum_arguments)
                record["warmup_compute"] = computer.metadata()
                del computer
            _synchronize(backend, args.device_id)
            record["warmup_s"] = time.perf_counter() - start
            _synchronize(backend, args.device_id)
            start = time.perf_counter()
            try:
                result = localize(
                    run_config, scene_json=scene_path, online_input=online_path,
                    generation_manifest=manifest_path, output_root=run_root,
                    run_receipt=run_root / "receipt.json",
                )
            finally:
                _synchronize(backend, args.device_id)
                record["localize_wall_s"] = time.perf_counter() - start
            record.update({
                "status": "success", "run_id": result["localization_run_id"],
                "mu_m": result["mu_m"], "clock_bias_s": result["clock_bias_s"],
                "compute": result["diagnostics"].get("compute", {}),
                "stage_timings_s": result["diagnostics"].get("stage_timings_s", {}),
            })
        except Exception as error:
            record.update({"status": "failed", "error_type": type(error).__name__,
                           "error": str(error), "traceback": traceback.format_exc()})
            print(f"{label}: 失败：{error}", file=sys.stderr, flush=True)
        _write_json(root / f"{label}_benchmark.json", record)
    report = {"runs": records, "comparison": {"passed": False, "reason": "至少一个后端未完成定位"}}
    if all(item["status"] == "success" for item in records.values()):
        report["comparison"] = compare_localizations(
            root / "cpu", root / "cuda", position_atol_m=args.position_atol_m,
            bias_atol_ns=args.bias_atol_ns,
        )
        report["cpu_over_gpu_time_ratio"] = records["cpu"]["localize_wall_s"] / records["cuda"]["localize_wall_s"]
    return report


def _run_kernel(args, root: Path, config: dict, measurement) -> dict:
    import numpy as np
    from time_bias_localization.compute import ComputeSettings, MusicComputer
    from time_bias_localization.pipeline import _separation_bins, estimate_noise_std_from_observed_csi
    from time_bias_localization.signal import _complex_gaussian_noise, extract_local_music_peaks, music_2d_spectrum

    kwargs = _spectrum_arguments(config, measurement)
    observed = measurement.csi_observed
    rng = np.random.default_rng(int(config["project"]["random_seed"]) + 2)
    noise_std = estimate_noise_std_from_observed_csi(observed, kwargs["num_sources"])
    noise_std *= config["music"]["uncertainty_noise_scale"]
    batch = np.stack([observed] + [
        observed + _complex_gaussian_noise(observed.shape, noise_std, rng)
        for _ in range(config["music"]["uncertainty_repeats"])
    ])
    if batch.ndim == 3:
        batch = batch[:, None, :, :]
    spectra, records = {}, {}
    for backend in ("numpy", "cuda"):
        label = "cpu" if backend == "numpy" else "cuda"
        record = {"status": "started"}
        records[label] = record
        print(f"{label}: 原始 CSI 及同种子的 {len(batch) - 1} 次扰动谱对照", flush=True)
        try:
            start = time.perf_counter()
            computer = MusicComputer(ComputeSettings(
                backend=backend, device_id=args.device_id, batch_size=args.batch_size,
                angle_chunk_size=args.angle_chunk_size,
            ))
            if backend == "numpy":
                music_2d_spectrum(batch[0], **kwargs)
            else:
                computer.spectrum(batch[0], **kwargs)
            _synchronize(backend, args.device_id)
            record["warmup_s"] = time.perf_counter() - start
            start = time.perf_counter()
            values = (np.stack([music_2d_spectrum(item, **kwargs) for item in batch])
                      if backend == "numpy" else computer.spectra(batch, **kwargs))
            _synchronize(backend, args.device_id)
            record["spectrum_wall_s"] = time.perf_counter() - start
            peak_sets = []
            start = time.perf_counter()
            nominal_count = None
            for index, spectrum in enumerate(values):
                limit = (config["music"]["num_paths"] if index == 0
                         else nominal_count + config["music"]["uncertainty_extra_peaks"])
                peaks = extract_local_music_peaks(
                    spectrum, aoa_grid_rad=kwargs["aoa_grid_rad"], delay_grid_s=kwargs["delay_grid_s"],
                    max_peaks=limit, minimum_separation_bins=_separation_bins(config["music"]),
                    minimum_relative_height=(0.0 if index == 0 else config["music"]["uncertainty_min_relative_height"]),
                )
                if index == 0:
                    nominal_count = len(peaks)
                peak_sets.append([asdict(peak) for peak in peaks])
            metadata = (computer.metadata() if backend == "cuda" else {
                "backend": "numpy", "array_library": "numpy", "array_library_version": np.__version__,
                "device_name": "CPU", "entrypoint": "signal.music_2d_spectrum", "batch_size": 1,
                "complex_dtype": "complex128", "real_dtype": "float64", "completed_csi": len(batch),
            })
            record.update({"peak_extraction_s": time.perf_counter() - start,
                           "status": "success", "compute": metadata, "peaks": peak_sets})
            spectra[label] = values
            np.savez_compressed(root / f"{label}_spectra.npz", spectra=values,
                                aoa_grid_rad=kwargs["aoa_grid_rad"], delay_grid_s=kwargs["delay_grid_s"])
        except Exception as error:
            record.update({"status": "failed", "error_type": type(error).__name__,
                           "error": str(error), "traceback": traceback.format_exc()})
        _write_json(root / f"{label}_benchmark.json", record)
    comparison = {"passed": False}
    report = {"runs": records, "comparison": comparison, "spectrum_count": len(batch)}
    if len(spectra) == 2:
        indices = {key: [[[p["aoa_index"], p["delay_index"]] for p in items]
                         for items in records[key]["peaks"]] for key in records}
        difference = np.abs(spectra["cuda"] - spectra["cpu"])
        spectrum_close = bool(np.allclose(spectra["cpu"], spectra["cuda"], rtol=1e-4, atol=1e-8))
        comparison.update({
            "passed": indices["cpu"] == indices["cuda"] and spectrum_close,
            "all_peak_indices_equal": indices["cpu"] == indices["cuda"],
            "spectra_within_tolerance": spectrum_close,
            "spectrum_rtol": 1e-4, "spectrum_atol": 1e-8,
            "spectrum_max_absolute_difference": float(difference.max()),
            "spectrum_relative_l2_difference": float(np.linalg.norm(difference) / np.linalg.norm(spectra["cpu"])),
        })
        report["cpu_over_gpu_time_ratio"] = records["cpu"]["spectrum_wall_s"] / records["cuda"]["spectrum_wall_s"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("full", "kernel"), default="full")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--angle-chunk-size", type=int, default=32)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--position-atol-m", type=float, default=1e-5)
    parser.add_argument("--bias-atol-ns", type=float, default=1e-4)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.angle_chunk_size, args.blas_threads) < 1 or args.device_id < 0:
        parser.error("批大小、角度分块和 BLAS 线程数必须为正整数，设备编号不能为负")
    if any(not math.isfinite(value) or value < 0 for value in (args.position_atol_m, args.bias_atol_ns)):
        parser.error("误差容差必须为有限非负数")
    if args.evaluate and args.mode != "full":
        parser.error("只有完整定位对照支持独立真值评估")
    # 本脚本在这里之前不导入任何数值库，线程设置对直接 Python 入口同样生效。
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(args.blas_threads)
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    config, manifest_path, manifest, scene_path, online_path, measurement = _load_inputs(
        args.input_root.expanduser().resolve(), args.config
    )
    report = {
        "schema_version": 1, "mode": args.mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable, "platform": platform.platform(),
        "arguments": vars(args), "random_seed": config["project"]["random_seed"],
        "input_provenance": {
            "generation_manifest": str(manifest_path), "bundle_id": manifest["bundle_id"],
            "online_measurement": str(online_path), "scene": str(scene_path),
            "online_sha256": manifest["artifact_hashes"]["online_measurement"]["sha256"],
            "scene_sha256": manifest["artifact_hashes"]["scene_json"]["sha256"],
            "localization_config_sha256": sha256(Path(config["_config_path"]).read_bytes()).hexdigest(),
        },
        "timing_policy": {
            "backend_order": ["cpu", "cuda"], "warmup_spectra_per_backend": 1,
            "warmup_excluded_from_localization_time": True,
            "cuda_synchronized_before_and_after_timing": True,
            "blas_threads": args.blas_threads,
            "comparison_scope": "重构后的 CPU 与 GPU；历史批次耗时不参与加速比",
            "limitations": "单输入单次计时，只说明此输入此设备的表现；不代表所有 sample 都加速。完整耗时不含射线生成、绘图和独立评估。",
        },
        "truth_file_content_loaded": False,
    }
    _write_json(root / "benchmark.json", report)
    start = time.perf_counter()
    if args.mode == "full":
        report.update(_run_full(args, root, config, manifest_path, scene_path, online_path, measurement))
    else:
        report.update(_run_kernel(args, root, config, measurement))
    report["benchmark_wall_s"] = time.perf_counter() - start
    _write_json(root / "benchmark.json", report)
    evaluation_ok = True
    if args.evaluate and all(item["status"] == "success" for item in report["runs"].values()):
        from time_bias_localization.pipeline import evaluate
        report["evaluation"] = {}
        for label, record in report["runs"].items():
            report["truth_file_content_loaded"] = True
            try:
                metrics = evaluate(
                    result_json=root / label / "localization" / "localization_result.json",
                    truth_npz=manifest["artifact_hashes"]["ground_truth"]["path"],
                    output_json=root / label / "evaluation" / "metrics.json",
                    expected_run_id=record["run_id"],
                )
                report["evaluation"][label] = {"status": "success", "metrics": metrics}
            except Exception as error:
                evaluation_ok = False
                report["evaluation"][label] = {"status": "failed", "error": str(error),
                                               "error_type": type(error).__name__}
    _write_json(root / "benchmark.json", report)
    with (root / "timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backend", "status", "stage", "seconds"])
        for label, record in report["runs"].items():
            for key in ("warmup_s", "localize_wall_s", "spectrum_wall_s", "peak_extraction_s"):
                if key in record:
                    writer.writerow([label, record["status"], key, record[key]])
            for key, value in record.get("stage_timings_s", {}).items():
                writer.writerow([label, record["status"], key, value])
    print(f"对照报告：{root / 'benchmark.json'}", flush=True)
    print(f"数值及步骤一致性：{'通过' if report['comparison']['passed'] else '未通过'}", flush=True)
    if "cpu_over_gpu_time_ratio" in report:
        print(f"CPU 耗时 / GPU 耗时 = {report['cpu_over_gpu_time_ratio']:.3f}", flush=True)
    return 0 if report["comparison"]["passed"] and evaluation_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
