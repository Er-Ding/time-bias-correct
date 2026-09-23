"""验证墙段合并，并在独立目录从原始网格重建地图、复算候选库。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

from time_bias_localization.diffraction import diffraction_edges
from time_bias_localization.propagation_hypotheses import build_hypothesis_bank
from time_bias_localization.propagation_model import ContinuousObservation
from time_bias_localization.provenance import file_sha256
from time_bias_localization.scene import (
    Scene2D, WallSegment, _triangle_horizontal_slice, merge_collinear_wall_segments,
    preprocess_sionna_triangle_mesh,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def covering_segments(wall, others):
    """独立检查整个有限墙段的覆盖区间，不以栅格图片相同代替几何检查。"""
    points = np.asarray([[w.start_m, w.end_m] for w in others])
    delta = points - wall.start
    tangent = wall.tangent
    perpendicular = np.abs(delta[:, :, 0] * tangent[1] - delta[:, :, 1] * tangent[0])
    aligned = np.max(perpendicular, axis=1) < 1e-7
    projected = delta @ tangent
    lower = np.maximum(0., np.min(projected, axis=1))
    upper = np.minimum(wall.length_m, np.max(projected, axis=1))
    indices = np.flatnonzero(aligned & (upper > lower))
    intervals = sorted((float(lower[i]), float(upper[i])) for i in indices)
    covered, end = 0., 0.
    for low, high in intervals:
        covered += max(0., high - max(end, low))
        end = max(end, high)
    return max(0., wall.length_m - covered), [others[i].wall_id for i in indices]


def raw_slices(vertices, faces, setup, height):
    # 沿用原来的切片判据；本轮被检验的是切片之后的合并与长度过滤。
    owners = np.full(len(vertices), -1, dtype=int)
    names = list(setup["object_vertex_ranges"])
    for i, (start, end) in enumerate(setup["object_vertex_ranges"].values()):
        owners[start:end] = i
    walls = []
    for i, face in enumerate(faces):
        triangle = vertices[face]
        if float(np.ptp(triangle[:, 2])) < .5:
            continue
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        size = np.linalg.norm(normal)
        if size <= 1e-9 or abs(normal[2]) / size > .15:
            continue
        segment = _triangle_horizontal_slice(triangle, fixed_height_m=height, plane_tolerance_m=1e-6)
        if segment is None or np.linalg.norm(segment[1] - segment[0]) <= 1e-9:
            continue
        owner = owners[face]
        number = int(owner[0]) if np.all(owner == owner[0]) else -1
        token = f"{number:04d}" if number >= 0 else "unknown"
        walls.append(WallSegment(f"sionna_{token}_{i:06d}", tuple(segment[0]), tuple(segment[1]),
                                 names[number] if number >= 0 else "sionna_mesh"))
    return tuple(walls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-scope", choices=("focused", "all"), default="focused")
    parser.add_argument("--samples", nargs="+", type=int, default=[103, 414, 685, 748, 820])
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source, output = args.input.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("验证输出必须与原始实验分开")
    output.mkdir(parents=True, exist_ok=False)
    tracked = list((project / "src/time_bias_localization").glob("*.py"))
    tracked += list((project / "tests").glob("test_*.py"))
    tracked += [Path(__file__).resolve(), project / "run_wall_geometry_check.sh"]
    code_hashes = {str(path): file_sha256(path) for path in tracked}
    write_json(output / "source.json", code_hashes)
    tests = ["test_wall_segment_union.py", "test_scene_and_candidates.py", "test_scene_persistence.py",
             "test_continuous_geometry.py", "test_continuous_hypothesis_search.py", "test_continuous_visibility.py",
             "test_diffraction_workflow.py", "test_boundary_channel.py", "test_sionna_generation.py",
             "test_contracts.py", "test_visualization_scene_equivalence.py"]
    command = [sys.executable, "-u", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--junitxml=" + str(output / "pytest.xml")]
    if args.test_scope == "focused":
        command.extend(str(project / "tests" / name) for name in tests)
    write_json(output / "test_command.json", {"cwd": str(project), "command": command})
    print("[1/4] 检查合并后门洞、拐角、反射及绕射路径是否正确。", flush=True)
    completed = subprocess.run(command, cwd=project, stdin=subprocess.DEVNULL)
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in ET.parse(output / "pytest.xml").getroot().iter("testsuite"):
        for key in counts:
            counts[key] += int(suite.get(key, 0))
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    write_json(output / "tests.json", {**counts, "exit_code": completed.returncode})
    if completed.returncode:
        raise SystemExit(completed.returncode)

    inputs = {}
    def capture(path, expected=None):
        path = Path(path).resolve()
        digest = file_sha256(path)
        if expected is not None and expected != digest:
            raise ValueError(f"输入指纹不符：{path}")
        inputs[str(path)] = digest
        return path

    def read(path, expected=None):
        return json.loads(capture(path, expected).read_text())

    print("[2/4] 从冻结的原始三角网格重新生成二维地图。", flush=True)
    setup_path, = (source / "channel_setups").glob("*/channel_setup.json")
    setup = read(setup_path)
    mesh_record = setup["public_mesh"]
    mesh_path = capture(mesh_record["path"], mesh_record["sha256"])
    original_scene_path = setup["scene_artifacts"]["scene_json"]
    original = Scene2D.from_dict(read(original_scene_path, setup["scene_fingerprint"]["sha256"]))
    with np.load(mesh_path) as mesh:
        vertices, faces = mesh["vertices_m"], mesh["faces"]
    started = time.perf_counter()
    scene = preprocess_sionna_triangle_mesh(vertices, faces, name=original.name,
        fixed_height_m=original.fixed_height_m, bev_resolution_m=original.bev_resolution_m,
        bounds_m=original.bounds_m, object_vertex_ranges=setup["object_vertex_ranges"])
    preparation_s = time.perf_counter() - started
    artifacts = scene.save(output / "corrected_scene")
    normalized_old = replace(original, walls=merge_collinear_wall_segments(original.walls))
    # 原场景读取不变；显式合并旧墙时，应双向保持原来的覆盖范围。
    old_missing = max(covering_segments(w, normalized_old.walls)[0] for w in original.walls)
    old_added = max(covering_segments(w, original.walls)[0] for w in normalized_old.walls)
    assert max(old_missing, old_added) < 1e-6
    assert merge_collinear_wall_segments(scene.walls) == scene.walls
    raw = raw_slices(vertices, faces, setup, original.fixed_height_m)
    mapping = {}
    for wall in scene.walls:
        missing, members = covering_segments(wall, raw)
        assert missing < 1e-6, (wall.wall_id, missing)
        mapping[wall.wall_id] = {"raw_face_ids": members, "uncovered_by_raw_mesh_m": missing}
    # 新图可以恢复旧流程提前删除的小片段，但不能丢掉旧图中已有的墙。
    missing_old = max(covering_segments(w, scene.walls)[0] for w in original.walls)
    assert missing_old < 1e-6
    restored = sum(covering_segments(w, normalized_old.walls)[0] for w in scene.walls)
    write_json(output / "wall_sources.json", mapping)
    example = []
    for face_index in (24960, 24961, 25357, 25358):
        if face_index >= len(faces):
            continue
        triangle = vertices[faces[face_index]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        example.append({"face_index": face_index, "vertices_m": triangle.tolist(),
            "normal": (normal / np.linalg.norm(normal)).tolist()})
    write_json(output / "opposite_facades_example.json", example)
    geometry = {"before_wall_count": len(original.walls), "after_wall_count": len(scene.walls),
        "before_edge_count": len(diffraction_edges(original)), "after_edge_count": len(diffraction_edges(scene)),
        "preparation_s": preparation_s, "old_wall_union_max_uncovered_m": old_missing,
        "old_wall_union_max_added_m": old_added, "old_walls_missing_in_new_scene_m": missing_old,
        "all_new_walls_covered_by_raw_mesh": True, "new_coverage_already_present_in_raw_mesh_m": restored,
        "idempotent": True, "scene_artifacts": artifacts}
    write_json(output / "geometry.json", geometry)
    print(f"[地图] 墙段 {len(original.walls)} → {len(scene.walls)}；旧墙覆盖与原网格覆盖检查通过。", flush=True)

    print("[3/4] 使用原来的观测及筛选参数，复算候选库；不使用真实位置。", flush=True)
    comparisons = []
    bank_keys = ("max_reflections", "max_diffractions", "diffraction_position", "max_hypotheses",
                 "max_enumerated_sequences", "aoa_gate_rad", "beta_interval_m", "length_gate_sigma")
    for number in args.samples:
        row = read(source / "samples" / f"SAMPLE_{number:06d}" / "result.json")
        folder, = Path(row["attempt_dir"]).glob("localization_unavailable/*")
        progress = read(folder / "progress.json")
        for name in ("propagation_hypotheses", "continuous_observations"):
            record = progress["artifacts"][name]
            capture(record["path"], record.get("sha256", record.get("file_sha256")))
        old_bank = read(folder / "propagation_hypotheses.json")
        observations = read(folder / "continuous_observations.json")["observations"]
        began = time.perf_counter()
        bank = build_hypothesis_bank(scene, setup["bs_position_m"],
            tuple(ContinuousObservation(**o) for o in observations),
            **{key: old_bank["search_report"][key] for key in bank_keys})
        elapsed = time.perf_counter() - began
        destination = output / row["sample_id"]
        destination.mkdir()
        write_json(destination / "propagation_hypotheses.json", bank.to_dict())
        write_json(destination / "continuous_observations.json", observations)
        comparison = {"sample_id": row["sample_id"], "before_count": len(old_bank["hypotheses"]),
            "after_count": len(bank.hypotheses), "before_search": old_bank["search_report"],
            "after_search": bank.search_report, "new_bank_seconds": elapsed,
            "truth_used": False, "full_position_solve_run": False,
            "after_family_counts": dict(Counter(f"R{h.reflection_order}D{h.diffraction_order}" for h in bank.hypotheses))}
        comparisons.append(comparison)
        write_json(output / "candidate_comparisons.json", comparisons)
        print(f"[候选库] {row['sample_id']}：{comparison['before_count']} → {comparison['after_count']}；"
              f"搜索停止原因={bank.search_report['stop_reason']}", flush=True)

    print("[4/4] 核对来源、保存报告。", flush=True)
    assert all(file_sha256(path) == digest for path, digest in inputs.items())
    assert all(file_sha256(path) == digest for path, digest in code_hashes.items())
    write_json(output / "provenance.json", {"inputs_sha256": inputs, "source_sha256": code_hashes,
        "inputs_and_source_unchanged_during_check": True, "original_experiment_modified": False,
        "timeout_limit_changed": False, "full_localization_rerun": False})
    lines = ["# 墙段重复修复验证", "", "原三维网格在同一墙位保存了方向相反、高度不同的立面。逐个三角形切片得到的分割点不同，旧版按端点去重无法识别重叠。", "",
        "现在按同一对象、同一条直线上的实际覆盖区间合并重叠或相接墙段，先合并后过滤短墙。只消除浮点舍入接缝，不跨真实间隙，不合并不同方向或不同对象的墙。", "",
        f"测试通过 {counts['passed']} 项，跳过 {counts['skipped']} 项。", "",
        f"地图墙段：{len(original.walls)} → {len(scene.walls)}；原始网格覆盖检查通过，没有删除旧图的墙或凭空增加墙。"
        f"补回原始网格中已有、旧图未覆盖的墙长约 {restored:.6f} m。", "",
        "| 样本 | 原候选数 | 新候选数 | 新搜索停止原因 |", "| --- | ---: | ---: | --- |"]
    for row in comparisons:
        lines.append(f"| {row['sample_id']} | {row['before_count']} | {row['after_count']} | {row['after_search']['stop_reason']} |")
    lines += ["", "候选数包括尚需在具体位置验证的走法，不等于实际有效多径数。若仍达到容量限制，应继续检查候选枚举与筛选；本报告不把墙段修复当成全部定位问题已解决。", "",
        f"新地图：[scene_2d.json]({artifacts['scene_json']})。新编号到原始三角面的对应关系：[wall_sources.json]({output / 'wall_sources.json'})。", "",
        "原实验和历史路径编号保持不变；修复作用于后续重新生成的二维地图。1800 秒时限及定位器的其他判断条件未修改。", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(f"[完成] {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
