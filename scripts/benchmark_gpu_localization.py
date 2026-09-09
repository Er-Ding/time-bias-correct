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


FINE_SPECTRUM_WORKFLOW = "music_fine_spectrum_dbscan_v3"
POINT_WORKFLOWS = {"music_point_clustering_v2", FINE_SPECTRUM_WORKFLOW}
SPECTRUM_WORKFLOWS = {"music_spectrum_sampling_v1", *POINT_WORKFLOWS}


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
    result = _read_json(folder / "localization_result.json")
    if peaks.get("workflow") in SPECTRUM_WORKFLOWS:
        sampling = _read_json(folder / "spectrum_samples.json")
        signature = {
            "workflow": peaks["workflow"],
            "nominal_peak_indices": [
                [item["aoa_index"], item["delay_index"]] for item in peaks["nominal"]
            ],
            "spectrum_sample_sources_and_cells": [
                {key: item[key] for key in (
                    "observation_id", "sample_id", "sampling_kind",
                    "cell_aoa_index", "cell_delay_index",
                )} for item in sampling["samples"]
            ],
            "central_selected": _candidate_signatures(
                list(result["central_selected_candidates"].values())
            ),
        }
        if peaks["workflow"] in POINT_WORKFLOWS:
            initial = _read_json(folder / "initial_candidates.json")
            representatives = _read_json(folder / "representative_points.json")
            signature.update(
                reference_bias_s=initial["reference_bias_s"],
                initial_candidates=_candidate_signatures(initial["points"]),
                initial_rejections=initial["rejected_samples"],
                point_clusters=[
                    (item["candidate_id"], item["point"]["sample_id"],
                     tuple(member["sample_id"] for member in item["members"]))
                    for item in representatives["representatives"]
                ],
                representative_trajectories=_candidate_signatures(
                    _read_json(folder / "representative_trajectories.json")
                ),
            )
            if peaks["workflow"] == FINE_SPECTRUM_WORKFLOW:
                signature.update(
                    coarse_peak_indices=[(item["aoa_index"], item["delay_index"]) for item in peaks["coarse"]],
                    nominal_source_indices=peaks["nominal_source_indices"],
                    fine_region_sources_and_indices=[
                        (item["observation_id"], item["source_peak_index"],
                         item["refined_aoa_grid_index"], item["refined_delay_grid_index"])
                        for item in sampling["regions"]
                    ],
                    suppressed_refined_peak_sources=[
                        (item["source_peak_index"], item["kept_source_peak_index"], item["reason"])
                        for item in sampling["diagnostics"]["suppressed_refined_peaks"]
                    ],
                    dbscan_noise_points=_candidate_signatures(representatives["noise_points"]),
                    dbscan_membership_roles=sorted([
                        (item["observation_id"], item["sample_id"], tuple(item["reflection_wall_ids"]),
                         item["candidate_id"], item["role"], item["neighbor_count"], item["is_representative"])
                        for item in representatives["memberships"]
                    ], key=repr),
                    dbscan_core_border_members=sorted([
                        (item["candidate_id"], tuple(sorted(item["metadata"]["core_sample_ids"])),
                         tuple(sorted(item["metadata"]["border_sample_ids"])))
                        for item in representatives["representatives"]
                    ], key=repr),
                )
        else:
            signature.update(
                raw_candidates=_candidate_signatures(_read_json(folder / "raw_reverse_candidates.json")),
                first_clusters=_candidate_signatures(_read_json(folder / "clustered_candidates.json")),
            )
        return signature
    # 只读兼容历史产物；新实验不再生成扰动支路。
    bootstrap = _read_json(folder / "bootstrap_diagnostics.json")
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


def _fine_music_checks(cpu_peaks: dict, gpu_peaks: dict, cpu_sampling: dict, gpu_sampling: dict) -> dict:
    """正式峰坐标及用于画图和抽样的同一细谱，完整核对两个后端。"""
    import numpy as np

    def close(cpu, gpu, absolute, relative=0):
        cpu_array, gpu_array = np.asarray(cpu), np.asarray(gpu)
        return cpu_array.shape == gpu_array.shape and bool(np.allclose(
            cpu_array, gpu_array, atol=absolute, rtol=relative,
        ))

    checks = {"fine_nominal_source_indices_equal":
              cpu_peaks["nominal_source_indices"] == gpu_peaks["nominal_source_indices"]}
    for field, absolute, relative in (("aoa_rad", 1e-10, 0), ("delay_s", 1e-17, 0),
                                      ("spectrum_value", 1e-8, 1e-4)):
        checks[f"fine_peak_{field}_within_tolerance"] = close(
            [peak[field] for peak in cpu_peaks["nominal"]],
            [peak[field] for peak in gpu_peaks["nominal"]], absolute, relative,
        )
    cpu_regions, gpu_regions = cpu_sampling["regions"], gpu_sampling["regions"]
    same_regions = len(cpu_regions) == len(gpu_regions)
    checks["fine_region_sources_equal"] = same_regions and all(
        cpu["observation_id"] == gpu["observation_id"]
        and cpu["source_peak_index"] == gpu["source_peak_index"]
        and cpu["refined_aoa_grid_index"] == gpu["refined_aoa_grid_index"]
        and cpu["refined_delay_grid_index"] == gpu["refined_delay_grid_index"]
        for cpu, gpu in zip(cpu_regions, gpu_regions)
    )
    for field, absolute, relative in (
        ("aoa_grid_rad", 1e-10, 0), ("delay_grid_s", 1e-17, 0),
        ("spectrum", 1e-8, 1e-4), ("cell_spectrum", 1e-8, 1e-4),
        ("cell_probabilities", 1e-12, 1e-4),
    ):
        checks[f"fine_region_{field}_within_tolerance"] = same_regions and all(
            close(cpu[field], gpu[field], absolute, relative)
            for cpu, gpu in zip(cpu_regions, gpu_regions)
        )
    checks["fine_region_boundary_status_equal"] = same_regions and all(
        all(cpu[field] == gpu[field] for field in (
            "peak_on_window_edge", "refined_peak_search_boundary_axes", "unresolved_window_peak",
        )) for cpu, gpu in zip(cpu_regions, gpu_regions)
    )
    checks["fine_peak_and_proposal_share_spectrum"] = all(
        sampling["diagnostics"]["peak_and_proposal_share_fine_spectrum"] is True
        and sampling["diagnostics"]["exact_sample_spectrum_used_for_proposal"] is False
        for sampling in (cpu_sampling, gpu_sampling)
    )
    return checks


def compare_localizations(cpu_root: Path, gpu_root: Path, *, position_atol_m: float,
                          bias_atol_ns: float) -> dict:
    """逐步核对峰、采样来源、候选和求解；兼容历史扰动产物的只读比较。"""
    import numpy as np

    cpu_folder, gpu_folder = cpu_root / "localization", gpu_root / "localization"
    cpu = _read_json(cpu_folder / "localization_result.json")
    gpu = _read_json(gpu_folder / "localization_result.json")
    cpu_signature, gpu_signature = _discrete_signature(cpu_folder), _discrete_signature(gpu_folder)
    checks = {key: cpu_signature.get(key) == gpu_signature.get(key)
              for key in cpu_signature.keys() | gpu_signature.keys()}
    workflow = cpu_signature.get("workflow")
    same_workflow = workflow == gpu_signature.get("workflow")
    if same_workflow and workflow in SPECTRUM_WORKFLOWS:
        cpu_sampling = _read_json(cpu_folder / "spectrum_samples.json")
        gpu_sampling = _read_json(gpu_folder / "spectrum_samples.json")
        for field, absolute, relative in (
            ("aoa_local_rad", 1e-10, 0), ("delay_s", 1e-17, 0),
            ("spectrum_value", 1e-8, 1e-4),
        ):
            cpu_values = np.asarray([item[field] for item in cpu_sampling["samples"]])
            gpu_values = np.asarray([item[field] for item in gpu_sampling["samples"]])
            checks[f"sample_{field}_within_tolerance"] = (
                cpu_values.shape == gpu_values.shape
                and bool(np.allclose(cpu_values, gpu_values, atol=absolute, rtol=relative))
            )
        checks["single_observation_subspace_per_backend"] = all(
            item["diagnostics"]["prepared_music"]["eigendecomposition_count"] == 1
            and item["diagnostics"]["added_csi_noise"] is False
            for item in (cpu_sampling, gpu_sampling)
        )
        if workflow == FINE_SPECTRUM_WORKFLOW:
            checks.update(_fine_music_checks(
                _read_json(cpu_folder / "music_peaks.json"), _read_json(gpu_folder / "music_peaks.json"),
                cpu_sampling, gpu_sampling,
            ))
    if same_workflow and workflow in POINT_WORKFLOWS:
        point_artifacts = [
            ("initial_points", "initial_candidates.json", "points", None),
            ("representative_points", "representative_points.json", "representatives", "point"),
        ]
        if workflow == FINE_SPECTRUM_WORKFLOW:
            point_artifacts.append(("dbscan_noise_points", "representative_points.json", "noise_points", None))
        for name, filename, list_key, point_key in point_artifacts:
            values = []
            for folder in (cpu_folder, gpu_folder):
                rows = _read_json(folder / filename)[list_key]
                values.append(np.asarray([
                    (item[point_key] if point_key else item)["position_m"] for item in rows
                ]))
            checks[f"{name}_coordinates_within_tolerance"] = (
                values[0].shape == values[1].shape
                and bool(np.allclose(*values, atol=1e-7, rtol=0))
            )
        trajectories = [_read_json(folder / "representative_trajectories.json")
                        for folder in (cpu_folder, gpu_folder)]
        for field in ("anchor_m", "direction", "beta_interval_m"):
            values = [np.asarray([item[field] for item in rows]) for rows in trajectories]
            checks[f"representative_trajectory_{field}_within_tolerance"] = (
                values[0].shape == values[1].shape
                and bool(np.allclose(*values, atol=1e-7, rtol=0))
            )
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
    checks["spectrum_within_tolerance"] = bool(np.allclose(
        cpu_spectrum, gpu_spectrum, atol=1e-8, rtol=1e-4,
    ))
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
            key: {"cpu": cpu_signature.get(key), "cuda": gpu_signature.get(key)}
            for key in cpu_signature.keys() | gpu_signature.keys() if not checks[key]
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
    """同一份带噪 CSI：单次子空间分解、全局谱和连续局部谱面采样。"""
    import numpy as np
    from time_bias_localization.compute import ComputeSettings, MusicComputer
    from time_bias_localization.pipeline import _separation_bins
    from time_bias_localization.signal import extract_local_music_peaks
    from time_bias_localization.spectrum_sampling import sample_music_spectrum

    kwargs = _spectrum_arguments(config, measurement)
    grids = {key: kwargs.pop(key) for key in ("aoa_grid_rad", "delay_grid_s")}
    seed = int(config["project"]["random_seed"]) + 2
    spectra, samplings, records = {}, {}, {}
    for backend in ("numpy", "cuda"):
        label = "cpu" if backend == "numpy" else "cuda"
        record = {"status": "started"}
        records[label] = record
        print(f"{label}: 同一份带噪 CSI 的一次分解、全局谱和局部谱采样对照", flush=True)
        try:
            settings = ComputeSettings(
                backend=backend, device_id=args.device_id, batch_size=args.batch_size,
                angle_chunk_size=args.angle_chunk_size,
            )
            start = time.perf_counter()
            warmup = MusicComputer(settings)
            warmup.spectrum(measurement.csi_observed, **kwargs, **grids)
            _synchronize(backend, args.device_id)
            record["warmup_s"] = time.perf_counter() - start
            del warmup
            computer = MusicComputer(settings)
            _synchronize(backend, args.device_id)
            total_start = time.perf_counter()
            start = time.perf_counter()
            prepared = computer.prepare(measurement.csi_observed, **kwargs)
            values = prepared.spectrum(**grids)
            _synchronize(backend, args.device_id)
            record["spectrum_wall_s"] = time.perf_counter() - start
            start = time.perf_counter()
            peaks = extract_local_music_peaks(
                values, **grids, max_peaks=config["music"]["num_paths"],
                minimum_separation_bins=_separation_bins(config["music"]),
                minimum_relative_height=0.0,
            )
            record["peak_extraction_s"] = time.perf_counter() - start
            start = time.perf_counter()
            sampled = sample_music_spectrum(
                prepared, peaks, **grids, bs_boresight_rad=measurement.bs_boresight_rad,
                settings=config["music"]["spectrum_sampling"], seed=seed,
                minimum_angle_separation_rad=np.deg2rad(float(config["music"]["min_angle_separation_deg"])),
                minimum_delay_separation_s=float(config["music"]["min_delay_separation_s"]),
            )
            _synchronize(backend, args.device_id)
            record["sampling_wall_s"] = time.perf_counter() - start
            record["music_and_sampling_wall_s"] = time.perf_counter() - total_start
            record.update({
                "status": "success", "compute": computer.metadata(),
                "prepared_music": prepared.metadata(), "coarse_peaks": [asdict(peak) for peak in peaks],
                "peaks": [asdict(peak) for peak in sampled.refined_peaks],
                "nominal_source_indices": sampled.refined_peak_source_indices,
                "sampling_diagnostics": sampled.diagnostics,
            })
            spectra[label] = values
            samplings[label] = sampled
            np.savez_compressed(root / f"{label}_spectra.npz", spectra=values, **grids)
            _write_json(root / f"{label}_spectrum_samples.json", {
                "samples": sampled.records, "regions": sampled.regions,
                "diagnostics": sampled.diagnostics,
            })
        except Exception as error:
            record.update({"status": "failed", "error_type": type(error).__name__,
                           "error": str(error), "traceback": traceback.format_exc()})
        _write_json(root / f"{label}_benchmark.json", record)
    comparison = {"passed": False}
    from time_bias_localization.pipeline import WORKFLOW
    report = {"runs": records, "comparison": comparison, "spectrum_count": 1,
              "workflow": WORKFLOW, "scope": "music_spectrum_and_sampling_only",
              "added_csi_noise": False}
    if len(spectra) == 2:
        indices = {key: [[p["aoa_index"], p["delay_index"]] for p in records[key]["peaks"]]
                   for key in records}
        difference = np.abs(spectra["cuda"] - spectra["cpu"])
        spectrum_close = bool(np.allclose(spectra["cpu"], spectra["cuda"], rtol=1e-4, atol=1e-8))
        cpu_records, gpu_records = samplings["cpu"].records, samplings["cuda"].records
        same_count = len(cpu_records) == len(gpu_records)
        same_sources = same_count and all(
            all(cpu[key] == gpu[key] for key in (
                "sample_id", "observation_id", "sampling_kind", "cell_aoa_index", "cell_delay_index",
            )) for cpu, gpu in zip(cpu_records, gpu_records)
        )
        continuous_checks = {}
        for field, absolute, relative in (
            ("aoa_local_rad", 1e-10, 0), ("delay_s", 1e-17, 0),
            ("spectrum_value", 1e-8, 1e-4),
        ):
            continuous_checks[field] = same_count and bool(np.allclose(
                [item[field] for item in cpu_records], [item[field] for item in gpu_records],
                atol=absolute, rtol=relative,
            ))
        single_eigh = all(records[key]["compute"]["eigendecomposition_count"] == 1 for key in records)
        fine_checks = _fine_music_checks(
            {"nominal": records["cpu"]["peaks"], "nominal_source_indices": records["cpu"]["nominal_source_indices"]},
            {"nominal": records["cuda"]["peaks"], "nominal_source_indices": records["cuda"]["nominal_source_indices"]},
            {"regions": samplings["cpu"].regions, "diagnostics": samplings["cpu"].diagnostics},
            {"regions": samplings["cuda"].regions, "diagnostics": samplings["cuda"].diagnostics},
        )
        coarse_indices_equal = [
            (peak["aoa_index"], peak["delay_index"]) for peak in records["cpu"]["coarse_peaks"]
        ] == [(peak["aoa_index"], peak["delay_index"]) for peak in records["cuda"]["coarse_peaks"]]
        comparison.update({
            "passed": (indices["cpu"] == indices["cuda"] and spectrum_close and same_sources
                       and all(continuous_checks.values()) and single_eigh
                       and coarse_indices_equal and all(fine_checks.values())),
            "all_peak_indices_equal": indices["cpu"] == indices["cuda"],
            "coarse_peak_indices_equal": coarse_indices_equal,
            "fine_music_checks": fine_checks,
            "spectra_within_tolerance": spectrum_close,
            "sample_sources_and_cells_equal": same_sources,
            "continuous_sample_checks": continuous_checks,
            "single_eigendecomposition_per_backend": single_eigh,
            "spectrum_rtol": 1e-4, "spectrum_atol": 1e-8,
            "spectrum_max_absolute_difference": float(difference.max()),
            "spectrum_relative_l2_difference": float(np.linalg.norm(difference) / np.linalg.norm(spectra["cpu"])),
        })
        report["cpu_over_gpu_time_ratio"] = (records["cpu"]["music_and_sampling_wall_s"]
                                                  / records["cuda"]["music_and_sampling_wall_s"])
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
            for key in ("warmup_s", "localize_wall_s", "spectrum_wall_s", "peak_extraction_s",
                        "sampling_wall_s", "music_and_sampling_wall_s"):
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
