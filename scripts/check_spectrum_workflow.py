"""在独立目录重放已有接收 CSI，检查谱面采样流程和逐步报告。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time
import traceback

import yaml

from time_bias_localization.config import load_localization_config
from time_bias_localization.pipeline import WORKFLOW, evaluate, localize
from time_bias_localization.provenance import artifact_record, load_generation_manifest
from time_bias_localization.visualization import create_report, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ue-ids", default="UE001,UE002,UE006")
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--samples-per-peak", type=int, default=128)
    parser.add_argument("--backend", choices=("numpy", "cuda"), default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args(argv)
    source = args.source_experiment.resolve()
    output = args.output.resolve()
    config = load_localization_config(args.config)
    source_plan = source / "experiment_plan.json"
    plan = read_json(source_plan)
    ue_ids = args.ue_ids.split(",")
    if len(set(ue_ids)) != len(ue_ids) or any(not name for name in ue_ids):
        parser.error("UE 编号必须非空且不重复")
    by_id = {point["ue_id"]: point for point in plan["points"]}
    if set(ue_ids) - set(by_id):
        parser.error("指定的 UE 不在原实验计划中")
    if not 1 <= args.noise_repeats <= plan["noise_repeats"] or args.samples_per_peak < 1:
        parser.error("重复数必须在原实验范围内，每峰采样数必须为正")
    config["compute"].update(backend=args.backend, device_id=args.device_id)
    config["music"]["spectrum_sampling"]["samples_per_peak"] = args.samples_per_peak
    # 重放使用原接收数据。计划中的真值只供后续独立评估/绘图，绝不传入 localize。
    points = [deepcopy(by_id[name]) for name in ue_ids]
    for point in points:
        point["noise_seeds"] = point["noise_seeds"][:args.noise_repeats]
    plan = dict(plan, workflow=WORKFLOW, execution_mode="saved_observation_replay",
                replay_source=artifact_record(source_plan),
                replay_localization_config=artifact_record(args.config),
                noise_repeats=args.noise_repeats, points=points)
    # 先核对全部输入存在，防止把拼错的来源当作算法失败。
    for point in points:
        for repeat in range(args.noise_repeats):
            path = source / point["ue_id"] / f"repeat_{repeat:03d}" / "generation_manifest.json"
            if not path.is_file():
                raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "experiment_plan.json", plan)
    results = []
    for point in points:
        for repeat, seed in enumerate(point["noise_seeds"]):
            root = output / point["ue_id"] / f"repeat_{repeat:03d}"
            root.mkdir(parents=True)
            manifest_path = source / point["ue_id"] / f"repeat_{repeat:03d}" / "generation_manifest.json"
            manifest, _, _ = load_generation_manifest(manifest_path)
            current = deepcopy(config)
            current["output"]["root"] = str(root)
            current["project"]["random_seed"] = seed + 100000000
            current.pop("_config_path", None)
            config_path = root / "localization.yaml"
            config_path.write_text(yaml.safe_dump(current, allow_unicode=True, sort_keys=False))
            status = dict(workflow=WORKFLOW, noise_seed=seed, source_generation_manifest=artifact_record(manifest_path))
            stage = "localization"
            started = time.perf_counter()
            print(f"{point['ue_id']} / {repeat}: 重放原接收 CSI，运行谱面采样", flush=True)
            try:
                result = localize(
                    load_localization_config(config_path),
                    scene_json=manifest["artifact_hashes"]["scene_json"]["path"],
                    online_input=manifest["artifact_hashes"]["online_measurement"]["path"],
                    generation_manifest=manifest_path, run_receipt=root / "receipt.json",
                )
                status["localization_seconds"] = time.perf_counter() - started
                stage = "evaluation"
                metrics = evaluate(
                    result_json=root / "localization/localization_result.json",
                    truth_npz=manifest["artifact_hashes"]["ground_truth"]["path"],
                    output_json=root / "evaluation/metrics.json",
                    expected_run_id=result["localization_run_id"],
                )
                status.update(status="success", run_id=result["localization_run_id"])
                results.append(dict(ue_id=point["ue_id"], repeat=repeat, status="success",
                    position_error_m=metrics["localization_error_m"],
                    bias_error_ns=metrics["clock_bias_error_ns"],
                    raw_count=result["diagnostics"]["raw_candidate_count"],
                    clustered_count=result["diagnostics"]["clustered_candidate_count"],
                    compute=result["diagnostics"]["compute"]))
            except Exception as error:
                status.update(status=f"{stage}_failed", error=f"{type(error).__name__}: {error}")
                if getattr(error, "failure_progress", None):
                    status["failure_progress"] = error.failure_progress
                (root / "failure.txt").write_text(traceback.format_exc())
                results.append(dict(ue_id=point["ue_id"], repeat=repeat, **status))
            write_json(root / "attempt.json", status)
            print(f"  {status['status']}", flush=True)
    write_json(output / "replay_results.json", results)
    report = create_report(output / "step_report", experiment=output)
    print(f"重放结束：{sum(row['status'] == 'success' for row in results)}/{len(results)} 份观测求解成功")
    print(f"逐步报告：{report}")


if __name__ == "__main__":
    main()
