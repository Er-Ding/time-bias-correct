"""核查固定 BS 端不可能的走法，复算候选库，并独立检查固定位置的有效路径。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

from time_bias_localization.propagation_hypotheses import (
    _fully_hidden_walls_from_receiver, _receiver_geometry_failure, build_hypothesis_bank,
)
from time_bias_localization.propagation_model import ContinuousObservation, PropagationHypothesis, evaluate_hypothesis
from time_bias_localization.provenance import file_sha256
from time_bias_localization.scene import Scene2D


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_hypothesis(value):
    item = dict(value)
    item["interactions"] = tuple(tuple(x) for x in item["interactions_ue_to_bs"])
    return PropagationHypothesis(**{f.name: item[f.name] for f in fields(PropagationHypothesis)})


def check_fixed_position(scene, hypotheses, retained_ids, position):
    valid, invalid = [], Counter()
    for h in hypotheses:
        result = evaluate_hypothesis(scene, h, position)
        if result.valid:
            assert h.hypothesis_id in retained_ids, ("合法路径被提前筛掉", h.hypothesis_id)
            valid.append({"hypothesis_id": h.hypothesis_id, "interactions": h.interactions,
                          "nodes_m": result.path.nodes.tolist(), "length_m": result.length_m})
        else:
            invalid[result.invalid_reason] += 1
    unique = {tuple(np.round(np.asarray(path["nodes_m"]).ravel(), 6)) for path in valid}
    return {"position_m": position, "valid_hypotheses": len(valid), "distinct_physical_paths_round6": len(unique),
        "duplicate_valid_paths": len(valid) - len(unique), "invalid_reason_counts": dict(invalid),
        "all_valid_paths_retained_by_new_bank": True, "valid_paths": valid,
        "scope": "valid_geometry_within_observation_filtered_bank_not_unrestricted_full_scene_rt"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="墙段修复后的 validation 目录")
    parser.add_argument("--experiment", type=Path, required=True, help="原实验，只在独立评估步骤读取真实位置")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", type=int, default=[103, 414, 685, 748, 820])
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source, experiment, output = args.input.resolve(), args.experiment.resolve(), args.output.resolve()
    for root in (source, experiment):
        if root == output or root in output.parents or output in root.parents:
            raise ValueError("输出目录必须与已有结果分开")
    output.mkdir(parents=True, exist_ok=False)
    code_files = list((project / "src/time_bias_localization").glob("*.py"))
    tests = sorted(set((project / "tests").glob("test_continuous*.py"))
                   | set((project / "tests").glob("test_diffraction*.py")))
    tests += [project / "tests/test_receiver_geometry_pruning.py", project / "tests/test_wall_segment_union.py"]
    code_files += tests + [Path(__file__).resolve(), project / "run_rt_candidate_audit.sh",
                           project / "scripts/detached_task.py"]
    code_hashes = {str(path): file_sha256(path) for path in code_files}
    inputs = {}

    def read(path, expected=None):
        path = Path(path).resolve()
        digest = file_sha256(path)
        if expected is not None and expected != digest:
            raise ValueError(f"输入来源校验失败：{path}")
        inputs[str(path)] = digest
        return json.loads(path.read_text())

    write_json(output / "source.json", code_hashes)
    command = [sys.executable, "-u", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--junitxml=" + str(output / "pytest.xml"), *(str(path) for path in tests)]
    write_json(output / "test_command.json", {"cwd": str(project), "command": command})
    print("[1/3] 检查提前排除是否保留全部参考路径。", flush=True)
    completed = subprocess.run(command, cwd=project, stdin=subprocess.DEVNULL)
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in ET.parse(output / "pytest.xml").getroot().iter("testsuite"):
        for key in counts:
            counts[key] += int(suite.get(key, 0))
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    write_json(output / "tests.json", {**counts, "exit_code": completed.returncode})
    if completed.returncode:
        raise SystemExit(completed.returncode)

    scene = Scene2D.from_dict(read(source / "corrected_scene/scene_2d.json"))
    walls = {wall.wall_id: wall for wall in scene.walls}
    prior_provenance = read(source / "provenance.json")
    for path, digest in prior_provenance["inputs_sha256"].items():
        if file_sha256(path) != digest:
            raise ValueError(f"原始实验已变化：{path}")
        inputs[path] = digest
    keys = ("max_reflections", "max_diffractions", "diffraction_position", "max_hypotheses",
            "max_enumerated_sequences", "aoa_gate_rad", "beta_interval_m", "length_gate_sigma")
    rows = []
    for number in args.samples:
        sid = f"SAMPLE_{number:06d}"
        saved = read(source / sid / "propagation_hypotheses.json")
        assert saved["search_report"]["complete_within_configured_orders"]
        bs = np.asarray(saved["receiver_m"])
        hypotheses = [load_hypothesis(h) for h in saved["hypotheses"]]
        hidden_walls = _fully_hidden_walls_from_receiver(walls, bs)
        blocked, wrong_side = set(), set()
        witnesses = []
        for h in hypotheses:
            first = _receiver_geometry_failure(h.interactions, walls, bs, hidden_walls)
            side_only = _receiver_geometry_failure(h.interactions, walls, bs, {})
            if first == "last_wall_fully_hidden_from_bs":
                blocked.add(h.hypothesis_id)
                witnesses.append({"hypothesis_id": h.hypothesis_id, "last_wall_id": h.interactions[-1][1],
                                  "blocker_wall_id": hidden_walls[h.interactions[-1][1]]})
            if side_only == "previous_wall_on_opposite_side_of_last_wall":
                wrong_side.add(h.hypothesis_id)
        started = time.perf_counter()
        bank = build_hypothesis_bank(scene, bs,
            tuple(ContinuousObservation(**o) for o in saved["observations"]),
            **{key: saved["search_report"][key] for key in keys})
        elapsed = time.perf_counter() - started
        retained = {h.hypothesis_id for h in bank.hypotheses}
        assert retained == {h.hypothesis_id for h in hypotheses} - blocked - wrong_side
        assert bank.search_report["complete_within_configured_orders"]
        destination = output / sid
        destination.mkdir()
        write_json(destination / "propagation_hypotheses.json", bank.to_dict())
        write_json(destination / "pruning_evidence.json", {
            "uses_ue_position": False, "blocking_walls": hidden_walls, "blocked_hypotheses": witnesses,
            "opposite_side_hypothesis_ids": sorted(wrong_side)})
        print(f"[2/3 候选] {sid}：{len(hypotheses)} → {len(retained)}；"
              f"完全遮挡 {len(blocked)}，反射侧别错误 {len(wrong_side)}，两者重叠 {len(blocked & wrong_side)}。", flush=True)

        # 在候选库生成并保存后，才从原实验读取评估位置；不反向修改候选库。
        original = read(experiment / "samples" / sid / "result.json")
        evaluation = check_fixed_position(scene, hypotheses, retained, original["true_position_m"])
        evaluation["truth_used_only_for_offline_geometry_check"] = True
        evaluation["sionna_paths_retained_by_original_generation"] = original["rt_path_count"]
        write_json(destination / "offline_fixed_position_geometry.json", evaluation)
        row = {"sample_id": sid, "before_candidates": len(hypotheses),
            "fully_blocked_candidates": len(blocked), "opposite_side_candidates": len(wrong_side),
            "overlap": len(blocked & wrong_side), "after_blocking_check_only": len(hypotheses) - len(blocked),
            "after_candidates": len(retained), "new_bank_seconds": elapsed,
            "valid_at_evaluation_position": evaluation["valid_hypotheses"],
            "distinct_paths_at_evaluation_position": evaluation["distinct_physical_paths_round6"],
            "valid_paths_lost": 0, "complete_enumeration": True,
            "original_generation_retained_path_count": original["rt_path_count"]}
        rows.append(row)
        write_json(output / "samples.json", rows)
        print(f"[3/3 固定位置] {sid}：候选中有 {row['valid_at_evaluation_position']} 条走通，"
              f"{row['distinct_paths_at_evaluation_position']} 条不同物理路径；提前筛选未丢失它们。", flush=True)

    assert all(file_sha256(path) == digest for path, digest in inputs.items())
    assert all(file_sha256(path) == digest for path, digest in code_hashes.items())
    write_json(output / "provenance.json", {"source_sha256": code_hashes, "inputs_sha256": inputs,
        "inputs_and_source_unchanged_during_check": True, "truth_used_in_candidate_generation": False,
        "truth_used_only_in_separate_offline_geometry_check": True, "full_localization_rerun": False,
        "timeout_limit_changed": False, "candidate_capacity_changed": False})
    lines = ["# RT 候选几何检查", "",
        "修复墙段重复后，原候选库仍保留大量对任意 UE 位置都不成立的走法。两项检查现在前移到候选入库之前：", "",
        "1. 如果 BS 到某墙两个端点的连线均严格穿过同一挡墙，则该整段墙处于挡墙的凸阴影区域，无法作为 BS 一侧末次反射墙。部分遮挡和擦边情况保守保留。",
        "2. 两次反射时，如果前一面墙整体位于最后反射墙的另一侧，且与 BS 分处两侧，则无法形成所声明的镜面反射顺序。", "",
        "两项检查不使用 UE 位置、真实路径、角度采样或数量小于 100 的限制。原 4096 容量与 1800 秒时限未修改。", "",
        f"{counts['passed']} 项检查通过，{counts['skipped']} 项跳过。", "",
        "| 样本 | 原候选 | 完全被挡的候选 | 仅查遮挡后剩余 | 加入反射侧别检查后 | 固定真实位置走通/不同路径 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['sample_id']} | {row['before_candidates']} | {row['fully_blocked_candidates']} | "
                     f"{row['after_blocking_check_only']} | {row['after_candidates']} | "
                     f"{row['valid_at_evaluation_position']}/{row['distinct_paths_at_evaluation_position']} |")
    lines += ["", "固定真实位置的检查仅用于事后验证，没有反馈给候选生成；它统计的是观测筛选后候选库中的有效几何路径，不等于对全场景、不限观测方向进行的完整正向 RT 结果。", "",
        "五个样本的原候选库和新候选库使用同一份修复后的地图、同一观测和原筛选参数；新候选集合恰好等于旧集合删除上述两类不可能走法，检查位置上没有丢失原本合法的路径。", "",
        "Sionna 官方也区分内部候选与最终有效路径，并在镜像法回溯阶段检查实际传播段：[Path Solver 技术说明](https://nvlabs.github.io/sionna/rt/tech-report/S3.html)。", "",
        "本次未重跑位置优化，不能据此报告整批定位成功率或定位加速比。", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(f"[完成] {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
