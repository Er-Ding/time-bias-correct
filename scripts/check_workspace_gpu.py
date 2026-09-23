"""用已保存观测检查 MUSIC GPU 及连续求解耗时；不读取真值、不覆盖实验。"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import pstats
import sys
import time

import numpy as np

from benchmark_gpu_localization import _spectrum_arguments
from time_bias_localization import continuous_solver
from time_bias_localization.compute import ComputeSettings, MusicComputer
from time_bias_localization.data import load_online_measurement
from time_bias_localization.propagation_hypotheses import HypothesisBank
from time_bias_localization.propagation_model import ContinuousObservation, PropagationHypothesis
from time_bias_localization.scene import Scene2D


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=2000)
    parser.add_argument("--solver-repeats", type=int, default=3)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if min(args.starts, args.iterations, args.seeds, args.solver_repeats) < 1:
        parser.error("检查预算必须为正整数")
    if source == output or source in output.parents or output in source.parents:
        parser.error("检查结果必须保存到独立目录")
    output.mkdir(parents=True, exist_ok=False)
    manifest = read(source / "localization_manifest.json")
    files = [source / name for name in ("localization_config.json", "continuous_search.json",
             "propagation_hypotheses.json", "continuous_observations.json", "localization_manifest.json")]
    files += [Path(manifest["scene_input"]), Path(manifest["online_measurement_input"])]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    code_paths = [Path(__file__), Path(__file__).with_name("benchmark_gpu_localization.py")]
    code_paths += [Path(continuous_solver.__file__).with_name(name + ".py") for name in
                  ("continuous_solver", "propagation_model", "diffraction", "raytrace2d", "scene", "compute")]
    if args.baseline_model:
        code_paths.append(args.baseline_model)
    config = read(source / "localization_config.json")["resolved_config"]
    saved = read(source / "propagation_hypotheses.json")
    observations = read(source / "continuous_observations.json")["observations"]
    search = read(source / "continuous_search.json")
    scene = Scene2D.from_dict(read(manifest["scene_input"]))
    hypotheses = tuple(PropagationHypothesis(**{**row, "interactions": tuple(map(tuple, row["interactions"]))})
                       for row in saved["hypotheses"])
    bank = HypothesisBank(scene, hypotheses[0].receiver_m,
        tuple(ContinuousObservation(**row) for row in observations), hypotheses,
        tuple(map(tuple, saved["observation_hypothesis_indices"])), saved["search_report"])
    solver_config = replace(continuous_solver.ContinuousSolverConfig(**search["diagnostics"]["config"]),
                           max_starts=args.starts, max_iterations=args.iterations,
                           max_seed_combinations=args.seeds)
    report = {"input_files_sha256": hashes, "truth_loaded": False,
              "code_files_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in code_paths},
              "scope": "single_saved_observation_limited_solver_budget_not_scientific_validation",
              "solver_budget": {"starts": args.starts, "iterations": args.iterations, "seeds": args.seeds},
              "observations": len(observations), "hypotheses": len(hypotheses), "music": {}, "solver": {}}
    measurement = load_online_measurement(manifest["online_measurement_input"])
    kwargs = _spectrum_arguments(config, measurement)
    spectra = {}
    for backend in ("numpy", "cuda"):
        print(f"[MUSIC] {backend}：预热后重复 3 次", flush=True)
        computer = MusicComputer(ComputeSettings(backend=backend))
        computer.spectrum(measurement.csi_observed, **kwargs)
        computer.synchronize()
        elapsed = []
        for _ in range(3):
            start = time.perf_counter()
            spectra[backend] = computer.spectrum(measurement.csi_observed, **kwargs)
            computer.synchronize()
            elapsed.append(time.perf_counter() - start)
        report["music"][backend] = {"seconds": elapsed, "median_s": float(np.median(elapsed)),
                                     "device": computer.metadata(),
                                     "subspace": computer._last_subspace_diagnostics}
    np.testing.assert_allclose(spectra["cuda"], spectra["numpy"], rtol=1e-4, atol=1e-8)
    report["music"]["comparison"] = {
        "rtol": 1e-4, "atol": 1e-8,
        "relative_l2_difference": float(np.linalg.norm(spectra["cuda"] - spectra["numpy"])
                                        / np.linalg.norm(spectra["numpy"])),
        "cpu_over_gpu": report["music"]["numpy"]["median_s"] / report["music"]["cuda"]["median_s"]}
    write(output / "review.json", report)
    evaluators = {"current": continuous_solver.evaluate_hypothesis}
    if args.baseline_model:
        spec = importlib.util.spec_from_file_location("time_bias_localization._review_before", args.baseline_model)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        evaluators = {"before": module.evaluate_hypothesis, **evaluators}
    for name, evaluator in evaluators.items():
        continuous_solver.evaluate_hypothesis = evaluator
        print(f"[连续求解] {name}：{args.starts} 个初值", flush=True)
        profiler = cProfile.Profile()
        began = time.perf_counter()
        result = profiler.runcall(continuous_solver.solve_continuous_position_and_bias, bank, solver_config).to_dict()
        elapsed = time.perf_counter() - began
        with (output / f"{name}_profile.txt").open("w") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(60)
        write(output / f"{name}_result.json", result)
        report["solver"][name] = {"profiled_seconds": elapsed, "status": result["status"],
                                   "position_m": result["position_m"], "beta_m": result["beta_m"]}
        elapsed = []
        for repetition in range(args.solver_repeats):
            print(f"[连续求解计时] {name} {repetition + 1}/{args.solver_repeats}", flush=True)
            began = time.perf_counter()
            repeated = continuous_solver.solve_continuous_position_and_bias(bank, solver_config).to_dict()
            elapsed.append(time.perf_counter() - began)
            assert repeated == result, "相同输入的重复结果应逐项一致"
        report["solver"][name].update(seconds=elapsed, median_s=float(np.median(elapsed)))
        write(output / "review.json", report)
    for path, expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    continuous_solver.evaluate_hypothesis = evaluators["current"]
    if "before" in evaluators:
        before, after = (read(output / f"{name}_result.json") for name in ("before", "current"))
        assert before["status"] == after["status"]
        if before["position_m"] is not None:
            np.testing.assert_allclose(before["position_m"], after["position_m"], atol=1e-5, rtol=0)
            assert abs(before["beta_m"] - after["beta_m"]) <= 1e-5
        signature = lambda row: [(p["observation_id"], p["hypothesis_id"]) for p in row["selected_paths"]]
        assert signature(before) == signature(after)
        assert before == after, "缓存复用应保持完整求解记录逐项一致"
        report["solver"]["comparison_passed"] = True
        report["solver"]["before_over_current"] = (report["solver"]["before"]["median_s"]
                                                    / report["solver"]["current"]["median_s"])
    report["input_hashes_unchanged"] = True
    write(output / "review.json", report)
    print(f"检查完成：{output / 'review.json'}", flush=True)


if __name__ == "__main__":
    main()
