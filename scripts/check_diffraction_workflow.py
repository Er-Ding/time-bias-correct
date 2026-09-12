"""在独立运行目录检查绕射闭环、完整测试和旧 CSI 对照。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from time_bias_localization.config import load_config, load_localization_config, localization_config_view
from time_bias_localization.pipeline import generate_data, localize, evaluate
from time_bias_localization.scene import WallSegment, make_synthetic_room
from time_bias_localization.visualization import create_report
from time_bias_localization.provenance import artifact_record


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def demo(config_path, output):
    config = load_config(config_path)
    output.mkdir(parents=True)
    config["output"]["root"] = str(output)
    config.pop("_config_path", None)
    snapshot = output / "generation.yaml"
    snapshot.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    scene = make_synthetic_room(bounds_m=config["scene"]["bounds_m"],
                                fixed_height_m=config["scene"]["fixed_height_m"],
                                bev_resolution_m=config["scene"]["bev_resolution_m"])
    scene = replace(scene, name=config["scene"]["name"],
                    walls=(*scene.walls, WallSegment("screen", (10., 0.), (10., 8.))))
    scene_artifacts = scene.save(output / "scene")
    print("绕射示例 1/4：按公开墙线生成含一次绕射的带噪 CSI", flush=True)
    bundle = generate_data(config, scene_json=scene_artifacts["scene_json"], output_root=output)
    online_config = localization_config_view(config)
    online_path = output / "localization.yaml"
    online_path.write_text(yaml.safe_dump(online_config, allow_unicode=True, sort_keys=False))
    print("绕射示例 2/4：MUSIC、反向追踪、DBSCAN、多代表与共享偏差求解", flush=True)
    result = localize(load_localization_config(online_path), scene_json=scene_artifacts["scene_json"],
                      online_input=bundle["online_npz"], generation_manifest=bundle["generation_manifest"],
                      output_root=output)
    print("绕射示例 3/4：独立读取真值评估，不回传定位器", flush=True)
    metrics = evaluate(result_json=output / "localization/localization_result.json",
                       truth_npz=bundle["truth_npz"], output_json=output / "evaluation/metrics.json",
                       expected_run_id=result["localization_run_id"])
    print("绕射示例 4/4：导出逐步报告并核对绕射簇数量", flush=True)
    report = create_report(output / "step_report", run_roots=[output])
    diagnostics = result["diagnostics"]
    reps = json.loads((output / "localization/representative_points.json").read_text())["representatives"]
    d_reps = [item for item in reps if any(kind == "diffraction" for kind, _ in item["point"]["propagation_interactions"])]
    d_clusters = {item["metadata"]["point_cluster_id"] for item in d_reps}
    if not d_clusters or len(d_reps) <= len(d_clusters):
        raise RuntimeError("示例没有实际产生绕射簇内多代表")
    if not all(item["metadata"]["all_member_valid_bias_intervals_covered"] for item in d_reps):
        raise RuntimeError("绕射代表未覆盖原成员合法偏差区间")
    selected_d = diagnostics["diffraction_physical_constraints"]["selected_diffraction_observation_count"]
    if selected_d < 1:
        raise RuntimeError("示例最终未选中绕射解释，尚未验证绕射求解闭环")
    if not result["forward_check"]["all_selected_paths_valid"]:
        raise RuntimeError("示例已选路径未通过正向几何检查")
    summary = {"workflow": result["workflow"], "initial_points": diagnostics["initial_candidate_count"],
               "clusters": diagnostics["point_clustering"]["cluster_count"], "representatives": len(reps),
               "diffraction_clusters": len(d_clusters), "diffraction_representatives": len(d_reps),
               "selected_diffraction_observations": selected_d,
               "position_error_m": metrics["localization_error_m"],
               "bias_error_ns": metrics["clock_bias_error_ns"],
               "all_selected_paths_valid": result["forward_check"]["all_selected_paths_valid"],
               "report": str(report), "source_config": artifact_record(config_path),
               "validation_scope": "single_configured_amplitude_synthetic_case_not_calibrated_utd_or_batch_accuracy"}
    write(output / "check_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def replay(source, output):
    output.mkdir(parents=True)
    manifest = json.loads((source / "localization/localization_manifest.json").read_text())
    config_snapshot = json.loads(Path(manifest["config_snapshot"]["path"]).read_text())
    config = config_snapshot["resolved_config"]
    config["output"]["root"] = str(output)
    # 对照复用原来的公开配置和随机种子，唯一修改是输出目录。
    config_path = output / "localization.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    old = json.loads((source / "localization/localization_result.json").read_text())
    print(f"旧数据对照：{source}", flush=True)
    current = localize(load_localization_config(config_path), scene_json=manifest["inputs"]["scene"]["path"],
                       online_input=manifest["inputs"]["online_measurement"]["path"],
                       generation_manifest=manifest["generation_bundle"]["manifest"]["path"], output_root=output)
    np.testing.assert_allclose(current["mu_m"], old["mu_m"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(current["clock_bias_s"], old["clock_bias_s"], atol=1e-15, rtol=0)
    summary = {"passed": True, "source_result": artifact_record(source / "localization/localization_result.json"),
               "position_change_m": float(np.linalg.norm(np.asarray(current["mu_m"]) - old["mu_m"])),
               "bias_change_ns": float((current["clock_bias_s"] - old["clock_bias_s"]) * 1e9)}
    write(output / "check_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tests", "demo", "replay", "all"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    if args.mode in {"all", "tests"}:
        print("运行完整测试集", flush=True)
        subprocess.run([sys.executable, "-u", "-m", "pytest", "-q", "--junitxml", str(args.output / "tests.xml")], check=True)
        results["tests"] = "passed"
    if args.mode in {"all", "demo"}:
        results["demo"] = demo(args.config.resolve(), args.output / "demo")
    if args.mode in {"all", "replay"}:
        results["replay"] = replay(args.source_run.resolve(), args.output / "replay")
    write(args.output / "check_summary.json", results)


if __name__ == "__main__":
    main()
