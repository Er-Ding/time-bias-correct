"""独立评估侧绘图：只读已有结果，不把真值传回定位器。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S
from .provenance import artifact_record, file_sha256, load_generation_manifest, verify_generation_artifact


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 新报告目录内发布；不允许 NaN 混入结果。
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


def checked_record(record: dict[str, str]) -> Path:
    path = Path(record["path"]).resolve()
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"文件与记录的指纹不一致：{path}")
    return path


def load_run(root: Path) -> dict[str, Any]:
    """核对生成批次、主结果、诊断和评估；不修改原始产物。"""
    manifest_path = root / "localization/localization_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("evaluation_pending", True):
        raise ValueError(f"该次定位尚未完成独立评估：{root}")
    generation_path = checked_record(manifest["generation_bundle"]["manifest"])
    generation, _, bundle_id = load_generation_manifest(generation_path)
    if bundle_id != manifest["generation_bundle"]["bundle_id"]:
        raise ValueError("定位与生成批次不一致")
    sources = [artifact_record(manifest_path), artifact_record(generation_path)]
    for key in ("scene_json", "online_measurement", "ground_truth"):
        record = generation["artifact_hashes"][key]
        verify_generation_artifact(generation, key, record["path"])
        sources.append(record)
    artifacts = {key: checked_record(record) for key, record in manifest["artifacts"].items()}
    snapshot_record = manifest["config_snapshot"]
    snapshot_path = checked_record({"path": snapshot_record["path"], "sha256": snapshot_record["file_sha256"]})
    sources.append(artifact_record(snapshot_path))
    sources.extend(manifest["artifacts"].values())
    metrics_path = checked_record(manifest["evaluation"])
    metrics = read_json(metrics_path)
    result = read_json(artifacts["result"])
    if not (manifest["run_id"] == metrics["localization_run_id"] == result["localization_run_id"]):
        raise ValueError("指标与结果的运行编号不一致")
    if metrics["generation_bundle_id"] != bundle_id or metrics["source_result_sha256"] != file_sha256(artifacts["result"]):
        raise ValueError("指标与生成批次或主结果不一致")
    truth_path = Path(generation["artifact_hashes"]["ground_truth"]["path"])
    if metrics["source_truth_sha256"] != file_sha256(truth_path):
        raise ValueError("指标与真值不一致")
    with np.load(truth_path, allow_pickle=False) as data:
        truth = {key: data[key] for key in data.files}
    with np.load(generation["artifact_hashes"]["online_measurement"]["path"], allow_pickle=False) as data:
        bs = data["bs_position_m"]
        boresight = float(data["bs_boresight_rad"])
    true = np.asarray(truth["ue_position_m"], dtype=float)
    estimated = np.asarray(result["mu_m"], dtype=float)
    error = float(np.linalg.norm(estimated - true))
    bias_error = (float(result["clock_bias_s"]) - float(truth["clock_bias_s"])) * 1e9
    if not np.all(np.isfinite([*true, *estimated, error, bias_error])):
        raise ValueError("坐标或误差不是有限数")
    if not np.isclose(error, metrics["localization_error_m"], rtol=1e-10, atol=1e-10) or not np.isclose(bias_error, metrics["clock_bias_error_ns"], atol=1e-9):
        raise ValueError("重新计算的误差与评估指标不一致")
    scene = read_json(generation["artifact_hashes"]["scene_json"]["path"])
    paths = []
    if "retained_mask" in truth:
        for index in np.flatnonzero(truth["retained_mask"]):
            active = np.flatnonzero(truth["interactions"][:, index] != 0)
            points = truth["vertices_m"][active, index, :2]
            paths.append({"order": len(active), "points": np.vstack([true, points, bs]).tolist()})
    else:
        # 离线版本的反射点在独立 JSON 中；另外记录此文件指纹。
        metadata_path = truth_path.with_suffix(".json")
        metadata = read_json(metadata_path)
        sources.append(artifact_record(metadata_path))
        for index, path in enumerate(metadata["paths"]):
            if not np.isclose(path["delay_s"], truth["path_delays_s"][index], rtol=1e-10, atol=1e-15):
                raise ValueError("离线路径 JSON 与 NPZ 的时延不一致")
            points = np.array([true.tolist(), *path["interaction_points_m"], bs.tolist()])
            length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            if not np.isclose(length / SPEED_OF_LIGHT_M_S, path["delay_s"], rtol=1e-7):
                raise ValueError("离线路径坐标与传播时延不一致")
            paths.append({"order": path["reflection_order"], "points": points.tolist()})
    sources.append(manifest["evaluation"])
    return dict(root=str(root), scene=scene, bs=bs.tolist(), boresight=boresight,
                true=true.tolist(), result=result, metrics=metrics, paths=paths,
                artifacts={key: str(path) for key, path in artifacts.items()},
                input_paths={key: record["path"] for key, record in generation["artifact_hashes"].items()},
                config_path=str(snapshot_path),
                raw=read_json(artifacts["raw_reverse_candidates"]),
                clusters=read_json(artifacts["clustered_candidates"]), sources=sources)


def load_failed_run(root: Path, progress_path: str | Path) -> dict[str, Any]:
    """读取 attempt 明确绑定的失败记录；只保留已有阶段，不推断最终解。"""
    progress_path = Path(progress_path).resolve()
    failure_root = (root / "localization_failures").resolve()
    if not progress_path.is_relative_to(failure_root):
        raise ValueError("失败进度文件不属于本次样本的 localization_failures 目录")
    progress = read_json(progress_path)
    if progress.get("workflow") != "music_spectrum_sampling_v1":
        raise ValueError("失败进度记录的工作流不受支持")
    if progress_path.parent.name != progress["run_id"]:
        raise ValueError("失败进度目录与运行编号不一致")
    generation_path = checked_record(progress["generation_bundle"]["manifest"])
    generation, _, bundle_id = load_generation_manifest(generation_path)
    if bundle_id != progress["generation_bundle"]["bundle_id"]:
        raise ValueError("失败记录与生成批次不一致")
    sources = [artifact_record(progress_path), artifact_record(generation_path)]
    for key in ("scene_json", "online_measurement", "ground_truth"):
        record = generation["artifact_hashes"][key]
        verify_generation_artifact(generation, key, record["path"])
        sources.append(record)
    for key, record in progress.get("inputs", {}).items():
        path = checked_record(record)
        generation_key = "scene_json" if key == "scene" else key
        if generation_key in generation["artifact_hashes"]:
            if file_sha256(path) != generation["artifact_hashes"][generation_key]["sha256"]:
                raise ValueError("失败记录的输入与生成批次不一致")
        sources.append(record)
    artifacts = {}
    for key, record in progress.get("artifacts", {}).items():
        path = checked_record(record)
        if not path.is_relative_to(progress_path.parent):
            raise ValueError("失败阶段产物不属于所绑定的失败运行目录")
        artifacts[key] = str(path)
        sources.append(record)
    snapshot = progress["config_snapshot"]
    config_path = checked_record({"path": snapshot["path"], "sha256": snapshot["file_sha256"]})
    sources.append(artifact_record(config_path))
    inputs = {key: record["path"] for key, record in generation["artifact_hashes"].items()}
    with np.load(inputs["online_measurement"], allow_pickle=False) as data:
        bs = data["bs_position_m"].tolist()
        boresight = float(data["bs_boresight_rad"])
    with np.load(inputs["ground_truth"], allow_pickle=False) as data:
        truth = {key: data[key] for key in data.files}
    true = np.asarray(truth["ue_position_m"], dtype=float)
    paths = []
    if "retained_mask" in truth:
        for index in np.flatnonzero(truth["retained_mask"]):
            active = np.flatnonzero(truth["interactions"][:, index] != 0)
            vertices = truth["vertices_m"][active, index, :2]
            paths.append({"order": len(active), "points": np.vstack([true, vertices, bs]).tolist()})
    else:
        metadata_path = Path(inputs["ground_truth"]).with_suffix(".json")
        metadata = read_json(metadata_path)
        sources.append(artifact_record(metadata_path))
        for index, item in enumerate(metadata["paths"]):
            points = np.asarray([true.tolist(), *item["interaction_points_m"], bs])
            length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            if (not np.isclose(item["delay_s"], truth["path_delays_s"][index], rtol=1e-10, atol=1e-15)
                    or not np.isclose(length / SPEED_OF_LIGHT_M_S, item["delay_s"], rtol=1e-7)):
                raise ValueError("失败记录的真值路径 JSON 与 NPZ 不一致")
            paths.append({"order": item["reflection_order"], "points": points.tolist()})
    result = read_json(artifacts["result"]) if "result" in artifacts else {"workflow": progress["workflow"]}
    if result.get("localization_run_id", progress["run_id"]) != progress["run_id"]:
        raise ValueError("失败阶段结果与运行编号不一致")
    return dict(root=str(root), workflow=progress["workflow"], scene=read_json(inputs["scene_json"]),
                bs=bs, boresight=boresight, true=true.tolist(), result=result, metrics={}, paths=paths,
                artifacts=artifacts, input_paths=inputs, config_path=str(config_path),
                raw=read_json(artifacts["raw_reverse_candidates"]) if "raw_reverse_candidates" in artifacts else [],
                clusters=read_json(artifacts["clustered_candidates"]) if "clustered_candidates" in artifacts else [],
                progress=progress, sources=sources)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """误差统计只用成功项，门限达标率以全部计划尝试为分母。"""
    solved = [row for row in rows if row["status"] == "success"]
    failed = sum(row["status"].endswith("failed") for row in rows)
    error = np.array([row["position_error_m"] for row in solved], dtype=float)
    bias = np.array([row["bias_error_ns"] for row in solved], dtype=float)
    return {
        "planned_count": len(rows), "solved_count": len(solved), "failed_count": failed,
        "pending_count": len(rows) - len(solved) - failed,
        "ue_count": len({row["ue_id"] for row in rows}),
        "position_median_m": float(np.median(error)) if len(error) else None,
        "position_rmse_m": float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
        "position_p90_m": float(np.quantile(error, .9)) if len(error) else None,
        "bias_mean_signed_ns": float(np.mean(bias)) if len(bias) else None,
        "bias_mae_ns": float(np.mean(abs(bias))) if len(bias) else None,
        "bias_rmse_ns": float(np.sqrt(np.mean(bias ** 2))) if len(bias) else None,
        **{f"within_{threshold}m_fraction_all": float(np.sum(error <= threshold) / len(rows)) if rows else None for threshold in (1, 2, 5)},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Droid Sans Fallback", "SimHei") if name in available]
    if not fonts:
        raise RuntimeError("缺少中文绘图字体，请安装 Noto Sans CJK 后重试")
    plt.rcParams.update({"font.family": fonts[0], "font.size": 8, "axes.unicode_minus": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False})
    return plt


def _scene_axes(ax, scene, bs, boresight, *, show_bs=True):
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection([[wall["start_m"], wall["end_m"]] for wall in scene["walls"]], colors="0.65", linewidths=.65))
    x0, x1, y0, y1 = scene["bounds_m"]
    ax.set(xlim=(x0, x1), ylim=(y0, y1), xlabel="x / m", ylabel="y / m", aspect="equal")
    if not show_bs:
        return
    ax.scatter(*bs, marker="^", s=65, color="#202020", label="BS", zorder=8)
    length = .07 * min(x1 - x0, y1 - y0)
    direction = length * np.array([np.cos(boresight), np.sin(boresight)])
    ax.annotate("", xy=np.asarray(bs) + direction, xytext=bs, arrowprops={"arrowstyle": "->", "color": "#202020"})
    for angle in (boresight - np.pi / 2, boresight + np.pi / 2):
        end = np.asarray(bs) + length * np.array([np.cos(angle), np.sin(angle)])
        ax.plot([bs[0], end[0]], [bs[1], end[1]], ":", color="0.35", lw=.8)


def _save(plt, fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _label_points(ax, points):
    """只移动编号文字，保留采样坐标；优先避开已有文字和点标记。"""
    from matplotlib.transforms import Bbox
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    occupied = []
    for point in points.values():
        x, y = ax.transData.transform(point)
        occupied.append(Bbox.from_extents(x - 4, y - 4, x + 4, y + 4))
    offsets = [(3, 3, "left"), (3, -9, "left"), (-3, 3, "right"), (-3, -9, "right"),
               (0, 12, "center"), (0, -18, "center"), (5, 20, "left"), (-5, 20, "right")]
    for label, point in points.items():
        choices = []
        for dx, dy, alignment in offsets:
            annotation = ax.annotate(label, point, fontsize=6, xytext=(dx, dy), textcoords="offset points", ha=alignment)
            box = annotation.get_window_extent(renderer).expanded(1.08, 1.15)
            overlaps = sum(box.overlaps(other) for other in occupied)
            choices.append((overlaps, dx, dy, alignment))
            annotation.remove()
            if not overlaps:
                break
        _, dx, dy, alignment = min(choices, key=lambda choice: choice[0])
        annotation = ax.annotate(label, point, fontsize=6, xytext=(dx, dy), textcoords="offset points", ha=alignment)
        occupied.append(annotation.get_window_extent(renderer).expanded(1.08, 1.15))


def _same_scene_geometry(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """仅用于全局底图比较，不用于跨运行匹配候选或改写墙编号。

    Sionna 导出时对象名称、墙编号和排列可能变化。比较精确的无向墙段
    多重集合，保留重复段数量及其他物理字段；不引入坐标容差。
    每个 sample 的路径图仍读取其自己的场景及原始墙编号。
    """
    if any(first[key] != second[key] for key in ("bounds_m", "fixed_height_m", "source")):
        return False
    if first["source"] == "sionna_exported_triangle_mesh":
        def segments(scene):
            return Counter(
                (tuple(sorted((tuple(wall["start_m"]), tuple(wall["end_m"])))),
                 json.dumps({key: value for key, value in wall.items()
                             if key not in {"wall_id", "source_object", "start_m", "end_m"}},
                            sort_keys=True, allow_nan=False))
                for wall in scene["walls"]
            )
        return segments(first) == segments(second)
    first_walls = [{key: value for key, value in wall.items() if key != "source_object"}
                   for wall in first["walls"]]
    second_walls = [{key: value for key, value in wall.items() if key != "source_object"}
                    for wall in second["walls"]]
    return first_walls == second_walls


def create_report(output: Path, *, run_roots: list[Path] | None = None, experiment: Path | None = None, summary_only: bool = False) -> Path:
    """单次已有结果或批量实验均可；缺失项保留为 pending。"""
    from .step_visualization import export_steps, export_summary
    plt = _plotting()
    rows, runs, partial_runs, sources = [], [], [], []
    plan = read_json(experiment / "experiment_plan.json") if experiment else None
    if plan:
        sources.append(artifact_record(experiment / "experiment_plan.json"))
        entries = []
        for point in plan["points"]:
            for repeat in range(plan["noise_repeats"]):
                root = experiment / point["ue_id"] / f"repeat_{repeat:03d}"
                status_path = root / "attempt.json"
                record = read_json(status_path) if status_path.exists() else {"status": "pending"}
                if record["status"] not in {"pending", "success", "generation_failed", "localization_failed", "evaluation_failed", "interrupted_failed", "worker_failed"}:
                    raise ValueError(f"未知运行状态：{record['status']}")
                if status_path.exists():
                    sources.append(artifact_record(status_path))
                entries.append((point["ue_id"], repeat, point["position_m"], root, record))
        scene_path = checked_record(plan["scene"])
        scene, bs, boresight = read_json(scene_path), plan["bs_position_m"], plan["bs_boresight_rad"]
    else:
        entries = []
        for i, root in enumerate(run_roots or []):
            attempt_path = root / "attempt.json"
            record = read_json(attempt_path) if attempt_path.exists() else {"status": "success"}
            entries.append((f"UE{i + 1:03d}", 0, None, root, record))
    if not entries:
        raise ValueError("没有可绘图的输入")
    for ue_id, repeat, true, root, status in entries:
        run = None
        workflow = status.get("workflow", plan.get("workflow") if plan else None)
        if status["status"] == "success":
            run = load_run(root)
            workflow = run["result"].get("workflow", run.get("workflow"))
            if true is not None and not np.allclose(true, run["true"], rtol=0, atol=1e-9):
                raise ValueError("结果 UE 与采样计划不一致")
            if status.get("run_id", run["result"]["localization_run_id"]) != run["result"]["localization_run_id"]:
                raise ValueError("批量记录与定位运行编号不一致")
            true = run["true"]
            if plan and not np.isclose(run["metrics"]["true_clock_bias_s"], plan["clock_bias_s"], rtol=0, atol=1e-15):
                raise ValueError("结果真实偏差与实验计划不一致")
            if not runs and not plan:
                scene, bs, boresight = run["scene"], run["bs"], run["boresight"]
            same_geometry = _same_scene_geometry(scene, run["scene"])
            if not same_geometry or bs != run["bs"] or not np.isclose(boresight, run["boresight"]):
                raise ValueError(f"{ue_id}/repeat_{repeat:03d} 的场景几何、BS 位置或朝向与全局底图不一致：{root}")
            sources.extend(run["sources"])
            runs.append((ue_id, repeat, run))
        elif status.get("failure_progress") and (not summary_only or not plan):
            partial = load_failed_run(root, status["failure_progress"])
            workflow = partial.get("result", {}).get("workflow", partial.get("workflow"))
            if true is not None and not np.allclose(true, partial["true"], rtol=0, atol=1e-9):
                raise ValueError("失败记录的 UE 与采样计划不一致")
            true = partial["true"]
            if not runs and not partial_runs and not plan:
                scene, bs, boresight = partial["scene"], partial["bs"], partial["boresight"]
            if (not _same_scene_geometry(scene, partial["scene"]) or bs != partial["bs"]
                    or not np.isclose(boresight, partial["boresight"])):
                raise ValueError("失败记录的场景或 BS 与全局底图不一致")
            sources.extend(partial["sources"])
            partial_runs.append((ue_id, repeat, partial))
        if true is None:
            raise ValueError(f"缺少样本位置与已绑定的失败进度；请用完整实验计划生成报告：{root}")
        result = run["result"] if run else {}
        metrics = run["metrics"] if run else {}
        estimated = result.get("mu_m", [None, None])
        true_bias = metrics.get("true_clock_bias_s", plan.get("clock_bias_s") if plan else None)
        rows.append(dict(ue_id=ue_id, noise_repeat=repeat, status=status["status"], workflow=workflow,
                         true_x_m=true[0], true_y_m=true[1], estimated_x_m=estimated[0], estimated_y_m=estimated[1],
                         position_error_m=metrics.get("localization_error_m"),
                         true_bias_ns=true_bias * 1e9 if true_bias is not None else None,
                         estimated_bias_ns=result["clock_bias_s"] * 1e9 if result else None,
                         bias_error_ns=metrics.get("clock_bias_error_ns"),
                         absolute_bias_error_ns=abs(metrics["clock_bias_error_ns"]) if metrics else None,
                         localization_seconds=status.get("localization_seconds"),
                         run_id=result.get("localization_run_id", ""), error=status.get("error", ""), root=str(root)))
    output.mkdir(parents=True, exist_ok=False)
    summary = summarize(rows)
    summary_root = output / "summary"
    sample_summaries = export_summary(plt, rows, summary, summary_root)
    runs_by_id = {(ue_id, repeat): run for ue_id, repeat, run in [*runs, *partial_runs]}
    for sample in sample_summaries:
        sample_root = output / "samples" / sample["ue_id"]
        sample_root.mkdir(parents=True)
        write_json(sample_root / "sample_summary.json", sample)
        sample_rows = [row for row in rows if row["ue_id"] == sample["ue_id"]]
        write_csv(sample_root / "results.csv", sample_rows)
        (sample_root / "README.md").write_text(
            f"# {sample['ue_id']}\n\n真实坐标：({sample['true_x_m']}, {sample['true_y_m']}) m。\n\n"
            + ("本报告只汇总已有结果；逐次状态和误差见 results.csv。" if summary_only else
               "\n".join(f"- [噪声重复 {row['noise_repeat']}：{row['status']}](repeat_{row['noise_repeat']:03d}/README.md)" for row in sample_rows)), encoding="utf-8")
        if summary_only:
            continue
        for row in sample_rows:
            export_steps(plt, runs_by_id.get((row["ue_id"], row["noise_repeat"])),
                         sample_root / f"repeat_{row['noise_repeat']:03d}", row)
    fig, ax = plt.subplots(figsize=(7.2, 6), layout="constrained")
    _scene_axes(ax, scene, bs, boresight)
    points = {row["ue_id"]: [row["true_x_m"], row["true_y_m"]] for row in rows}
    for i, (label, point) in enumerate(points.items()):
        ax.scatter(*point, marker="*", s=35, color="#CC9239", label="UE 真值" if i == 0 else None, zorder=7)
        if len(points) <= 10:
            ax.annotate(label, point, fontsize=6, xytext=(3, 3), textcoords="offset points")
    if plan:
        from matplotlib.patches import Rectangle
        x0, x1, y0, y1 = plan["sampling_bounds_m"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", ec="#4477AA", label="固定采样区域"))
    ax.set_title(f"场景与采样位置：{len(points)} 个 UE；BS 朝向 {np.degrees(boresight):.1f}°")
    ax.legend()
    _save(plt, fig, summary_root, "01_scene_samples")
    if plan:
        fig, ax = plt.subplots(figsize=(7.2, 5.4), layout="constrained")
        x0, x1, y0, y1 = plan["sampling_bounds_m"]
        zoom_scene = {**scene, "bounds_m": [min(x0, bs[0]) - 2, max(x1, bs[0]) + 2, min(y0, bs[1]) - 2, max(y1, bs[1]) + 2]}
        _scene_axes(ax, zoom_scene, bs, boresight)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", ec="#4477AA", label="采样区域"))
        for label, point in points.items():
            ax.scatter(*point, marker="*", s=35, color="#CC9239", zorder=7)
        ax.set_title("采样区域放大图（米制坐标，与全局图一致）")
        ax.legend()
        _label_points(ax, points)
        _save(plt, fig, summary_root, "01_scene_samples_zoom")
    fig, ax = plt.subplots(figsize=(7.2, 6), layout="constrained")
    _scene_axes(ax, scene, bs, boresight)
    for i, point in enumerate(points.values()):
        ax.scatter(*point, s=30, marker="*", color="#CC9239", label="UE 真值" if i == 0 else None, zorder=7)
    solved = [row for row in rows if row["status"] == "success"]
    for row in solved:
        ax.plot([row["true_x_m"], row["estimated_x_m"]], [row["true_y_m"], row["estimated_y_m"]], lw=.6, color="0.5")
    if solved:
        scatter = ax.scatter([r["estimated_x_m"] for r in solved], [r["estimated_y_m"] for r in solved], c=[r["position_error_m"] for r in solved], cmap="viridis", marker="x", s=25, label="每次估计", zorder=8)
        fig.colorbar(scatter, ax=ax, shrink=.7, label="位置误差 / m")
        # 包含场景外的错误估计，不能被固定地图边界裁掉。
        x0, x1, y0, y1 = scene["bounds_m"]
        ax.set(xlim=(min(x0, min(r["estimated_x_m"] for r in solved)) - 1, max(x1, max(r["estimated_x_m"] for r in solved)) + 1),
               ylim=(min(y0, min(r["estimated_y_m"] for r in solved)) - 1, max(y1, max(r["estimated_y_m"] for r in solved)) + 1))
    unsolved = [r for r in rows if r["status"] != "success"]
    if unsolved:
        ax.scatter([r["true_x_m"] for r in unsolved], [r["true_y_m"] for r in unsolved], facecolors="none", edgecolors="#BB5566", s=65, label="含失败或待运行项", zorder=9)
    ax.set_title(f"定位结果：{len(solved)}/{len(rows)} 次已得到结果")
    ax.legend()
    _save(plt, fig, summary_root, "03_localization_map")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
    if solved:
        errors = np.sort([r["position_error_m"] for r in solved])
        axes[0].step(np.r_[0., errors], np.r_[0., np.arange(1, len(errors) + 1) / len(rows)], where="post", color="#4477AA")
        bias_errors = np.array([r["bias_error_ns"] for r in solved])
        axes[1].hist(bias_errors, bins=min(20, max(1, int(np.sqrt(len(solved))))), color="#4477AA", alpha=.8)
        axes[1].axvline(0, color="0.35", ls="--", lw=.8)
    else:
        for ax in axes:
            ax.text(.5, .5, "尚无成功定位结果", transform=ax.transAxes, ha="center")
    axes[0].set(xlabel="位置误差门限 / m", ylabel="占全部计划尝试的比例", ylim=(0, 1.02), title="a  位置误差累计比例")
    axes[1].set(xlabel="估计偏差 − 真实偏差 / ns", ylabel="次数", title="b  有符号偏差误差")
    fig.suptitle(f"计划 {len(rows)} 次；成功 {len(solved)}，失败 {summary['failed_count']}，待运行 {summary['pending_count']}", fontsize=9)
    _save(plt, fig, summary_root, "all_attempt_attainment_and_bias")
    (output / "README.md").write_text(
        "# 按 sample 和执行步骤查看定位结果\n\n"
        "- [总体误差统计](summary/README.md)：CDF、Med、P90、逐次结果表及逐 UE 汇总。\n"
        + "\n".join(f"- [{sample['ue_id']}](samples/{sample['ue_id']}/README.md)" for sample in sample_summaries)
        + ("\n\n本报告仅包含汇总图表及各 UE 结果表，未导出逐步骤图。\n" if summary_only else
           "\n\n每个 UE 的 repeat 子目录含 00–08 步骤，每步保存图、数据与对应函数说明。\n")
        +
        "不绘制候选簇联系。00 真值只作参照，不是定位输入。绘图不重新执行定位。\n"
        "旧结果没有保存的内部历史明确标注缺失；失败和待运行项保留目录和状态，不编造中间输出。\n"
        "summary/position_error_cdf 为成功结果的标准 CDF；all_attempt_attainment_and_bias 的达标比例以全部计划次数为分母。\n"
        "PNG 为 300 dpi，另存 PDF/SVG；新流程椭圆仅表示几何残差近似，未校准，不能当作谱面采样的置信区间。\n",
        encoding="utf-8")
    write_json(output / "report_manifest.json", {"schema_version": 1, "evaluation_only": True,
               "summary_only": summary_only,
               "scene_comparison": "Sionna: exact undirected segment multiset including multiplicity and physical fields; export IDs and object labels ignored only for global background",
               "plotting_code": artifact_record(Path(__file__)),
               "step_plotting_code": artifact_record(Path(__file__).with_name("step_visualization.py")),
               "sources": list({record["path"]: record for record in sources}.values()),
               "artifacts": [artifact_record(path) for path in sorted(output.rglob("*")) if path.is_file()]})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成场景、路径、初次聚类和定位误差图表")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--run-root", type=Path, action="append", help="已有单次结果目录，可重复指定")
    inputs.add_argument("--experiment", type=Path, help="批量实验目录")
    parser.add_argument("--output", type=Path, required=True, help="新的图表输出目录，禁止覆盖")
    parser.add_argument("--summary-only", action="store_true", help="仅汇总图表，不导出逐步骤图；仍校验全部成功结果的来源")
    args = parser.parse_args(argv)
    print(create_report(args.output.resolve(), run_roots=args.run_root, experiment=args.experiment, summary_only=args.summary_only))


if __name__ == "__main__":
    main()
