"""按 UE / 噪声重复 / 实际算法步骤组织可复查的图表和原始输出。"""

from pathlib import Path
import csv
import shutil

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S
from .visualization import _save, _scene_axes, read_json, write_csv, write_json


WORKFLOW = "music_spectrum_sampling_v1"
POINT_WORKFLOW = "music_point_clustering_v2"
FINE_WORKFLOW = "music_fine_spectrum_dbscan_v3"
DIFFRACTION_WORKFLOW = "music_diffraction_cover_v4"

STEPS = [
    ("00_scene_truth", "场景与仿真真值（仅作参照）", "generate_synthetic_measurement / extract_planar_uplink_csi"),
    ("01_csi_input", "一份带噪 CSI 输入", "load_online_measurement_bytes"),
    ("02_music", "二维 MUSIC 谱与原始峰", "music_2d_spectrum / MusicComputer.spectrum → extract_local_music_peaks"),
    ("03_spectrum_sampling", "各峰附近的连续角度与时延采样", "sample_music_spectrum"),
    ("04_reverse_candidates", "全部谱面样本的反向候选轨迹", "generate_reverse_candidates"),
    ("05_first_clustering", "候选聚类、成员与代表", "cluster_reverse_candidates"),
    ("06_joint_solution", "代表候选的位置与共同偏差联合解", "solve_position_and_bias"),
    ("07_forward_check", "预测路径与输入观测的检查", "forward_check"),
    ("08_final_evaluation", "独立真值评价", "evaluate"),
]

POINT_STEPS = [
    *STEPS[:4],
    ("04_initial_candidates", "反向追踪得到的初始位置点", "generate_initial_candidate_points"),
    ("05_point_clustering", "初始位置点聚类、成员与代表点", "cluster_initial_candidate_points"),
    ("06_representative_trajectories", "仅由代表点建立的偏差—位置轨迹", "build_representative_trajectories"),
    ("07_joint_solution", "代表轨迹的位置与共同偏差联合解", "solve_position_and_bias"),
    ("08_forward_check", "预测路径与输入观测的检查", "forward_check_solution"),
    ("09_final_evaluation", "独立真值评价", "evaluate"),
]

FINE_STEPS = [
    *POINT_STEPS[:2],
    ("02_music", "粗谱搜索区域与局部细谱正式峰", "music_2d_spectrum / MusicComputer.spectrum → sample_music_spectrum"),
    ("03_spectrum_sampling", "在正式峰所属的同一局部细谱上采样", "sample_music_spectrum"),
    POINT_STEPS[4],
    ("05_point_clustering", "DBSCAN 点簇、真实代表点与离群点", "cluster_initial_candidate_points"),
    *POINT_STEPS[6:],
]


def _is_fine_workflow(run):
    return run.get("result", {}).get("workflow", run.get("workflow")) in {FINE_WORKFLOW, DIFFRACTION_WORKFLOW}

LEGACY_STEPS = [
    ("00_scene_truth", "场景与仿真真值（仅作参照）", "generate_synthetic_measurement / extract_planar_uplink_csi"),
    ("01_csi_input", "定位输入 CSI", "load_online_measurement_bytes"),
    ("02_music", "二维 MUSIC 谱与原始峰", "CPU: music_2d_spectrum / GPU: MusicComputer.spectrum → extract_local_music_peaks"),
    ("03_peak_perturbations", "CSI 扰动后的峰与配对", "CPU: estimate_music_peak_samples / GPU: estimate_batched_peak_samples → associate_perturbed_peaks → _build_observation_samples"),
    ("04_reverse_candidates", "原始峰的反向候选轨迹", "generate_reverse_candidates"),
    ("05_first_clustering", "第一次聚类", "cluster_reverse_candidates"),
    ("06_joint_solution", "原始峰的位置与偏差联合解", "solve_position_and_bias"),
    ("07_perturbation_solutions", "每次扰动的重复求解", "_bootstrap_joint_solutions"),
    ("08_final_evaluation", "最终汇总位置、偏差与误差", "_distribution_statistics → evaluate"),
]


def _copy(source, destination):
    shutil.copy2(source, destination)


def _table(path, rows):
    if rows:
        write_csv(path, rows)


def _map(plt, run, title):
    fig, ax = plt.subplots(figsize=(7.2, 5.4), layout="constrained")
    _scene_axes(ax, run["scene"], run["bs"], run["boresight"])
    ax.scatter(*run["true"], marker="*", color="#CC9239", s=75, label="UE 真值（仅参照）", zorder=8)
    ax.set_title(title)
    return fig, ax


def _observation_indices(run, observation_ids):
    """沿用 MUSIC 来源编号；某个峰没有合法候选时不压缩后续编号或颜色。"""
    source = run.get("artifacts", {}).get("spectrum_samples")
    regions = read_json(source).get("regions", []) if source else []
    known = {region["observation_id"]: int(region.get("peak_index", index))
             for index, region in enumerate(regions)}
    result = {}
    for fallback, observation_id in enumerate(sorted(observation_ids)):
        if observation_id in known:
            result[observation_id] = known[observation_id]
        elif observation_id.startswith("music_path_") and observation_id.removeprefix("music_path_").isdigit():
            result[observation_id] = int(observation_id.removeprefix("music_path_"))
        else:
            result[observation_id] = fallback
    return result


def _trajectories(plt, run, candidates, directory, *, clustered, title, show_initial_points=False):
    fig, ax = _map(plt, run, title)
    ids = sorted({item["observation_id"] for item in candidates})
    indices = _observation_indices(run, ids) if show_initial_points else {key: index for index, key in enumerate(ids)}
    colors = {key: plt.get_cmap("tab10")(indices[key] % 10) for key in ids}
    rows, extents = [], [run["true"], run["bs"]]
    for index, item in enumerate(candidates):
        interval = item["beta_interval_m"] if clustered else [item["beta_min_m"], item["beta_max_m"]]
        endpoints = np.asarray(item["anchor_m"]) - np.asarray(interval)[:, None] * np.asarray(item["direction"])
        extents.extend(endpoints.tolist())
        metadata = item["metadata"] if clustered else item
        order = len(metadata["reflection_wall_ids"])
        label = f"{'C' if clustered else 'R'}{index + 1}"
        ax.plot(*endpoints.T, color=colors[item["observation_id"]], ls=["-", "--", "-."][order], lw=1.5 if clustered else .9, alpha=.8)
        if show_initial_points:
            ax.scatter(*metadata["initial_position_m"], color=colors[item["observation_id"]], marker="D", s=25,
                       edgecolors="black", linewidths=.4, zorder=8)
        if len(candidates) <= (8 if clustered else 30):
            ax.annotate(label, endpoints.mean(axis=0), fontsize=6, xytext=(3, 3), textcoords="offset points")
        rows.append(dict(label=label, observation_id=item["observation_id"],
                         candidate_id=item.get("candidate_id", ""), sample_id=item.get("sample_id", ""),
                         reflection_order=order, reflection_wall_ids=" | ".join(metadata["reflection_wall_ids"]),
                         anchor_x_m=item["anchor_m"][0], anchor_y_m=item["anchor_m"][1],
                         direction_x=item["direction"][0], direction_y=item["direction"][1],
                         bias_min_ns=interval[0] / SPEED_OF_LIGHT_M_S * 1e9,
                         bias_max_ns=interval[1] / SPEED_OF_LIGHT_M_S * 1e9,
                         member_count=metadata.get("raw_count", 1),
                         representative_sample_id=metadata.get("representative_sample_id", "")))
    for key in ids:
        ax.plot([], [], color=colors[key], label=f"观测 {indices[key] + 1}")
    if show_initial_points:
        ax.scatter([], [], color="0.5", marker="D", s=25, edgecolors="black", linewidths=.4,
                   label="上一步保留的初始代表点")
    xy = np.asarray(extents)
    ax.set(xlim=(xy[:, 0].min() - 3, xy[:, 0].max() + 3), ylim=(xy[:, 1].min() - 3, xy[:, 1].max() + 3))
    ax.legend(fontsize=7)
    fig.suptitle("完整合法轨迹；颜色对应观测，实/虚/点划线对应 0/1/2 次反射；未代入最终估计偏差", fontsize=8)
    _save(plt, fig, directory, "trajectories")
    _table(directory / "trajectories.csv", rows)


def _position(plt, run, solution, directory, title):
    from matplotlib.patches import Ellipse
    fig, ax = _map(plt, run, title)
    estimate, true = np.asarray(solution["mu_m"]), np.asarray(run["true"])
    eigenvalues, eigenvectors = np.linalg.eigh(solution["sigma_m2"])
    if eigenvalues.min() < -1e-9:
        raise ValueError("位置协方差不是半正定矩阵")
    radii = np.sqrt(5.991 * np.maximum(eigenvalues, 0))
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    ellipse_label = ("几何残差近似椭圆（未校准，非采样置信区间）"
                     if run.get("result", {}).get("workflow") in {WORKFLOW, POINT_WORKFLOW, FINE_WORKFLOW, DIFFRACTION_WORKFLOW}
                     else "名义 95% 椭圆（未校准）")
    ax.add_patch(Ellipse(estimate, 2 * radii[1], 2 * radii[0], angle=angle, fill=False,
                        color="#4477AA", label=ellipse_label))
    ax.scatter(*estimate, marker="x", s=50, color="#4477AA", label="本步骤位置输出", zorder=9)
    ax.plot([true[0], estimate[0]], [true[1], estimate[1]], color="0.45", lw=.8)
    margin = max(1., np.linalg.norm(estimate - true), radii.max()) * 1.5
    center = (estimate + true) / 2
    ax.set(xlim=(center[0] - margin, center[0] + margin), ylim=(center[1] - margin, center[1] + margin))
    ax.legend(fontsize=7)
    fig.suptitle(f"偏差输出 {solution['clock_bias_s'] * 1e9:.4f} ns；与真值距离 {np.linalg.norm(estimate - true):.4f} m", fontsize=9)
    _save(plt, fig, directory, "position")


def _export_legacy_steps(plt, run, directory: Path, row: dict) -> None:
    """每一步保存原始数据、图及函数说明，旧产物缺失处明确标注。"""
    directory.mkdir(parents=True, exist_ok=False)
    folders = {}
    index = []
    for name, title, function in LEGACY_STEPS:
        folder = directory / name
        folder.mkdir()
        folders[name] = folder
        state = "available" if run else "not_available"
        note = "原始结果的已记录输出；不会重新运行算法或编造迭代历史。" if run else f"状态：{row['status']}。未得到可核验的完整定位产物，本步骤没有可导出的数据。{row.get('error', '')}"
        (folder / "README.md").write_text(f"# {title}\n\n对应函数：`{function}`。\n\n{note}\n", encoding="utf-8")
        index.append(dict(step=name, title=title, function=function, status=state))
    write_json(directory / "attempt.json", row)
    write_json(directory / "step_index.json", index)
    (directory / "README.md").write_text(
        "# 按执行步骤查看本次定位\n\n"
        + "旧流程（legacy）：CSI 额外扰动后分别求解，再汇总。不是谱面采样主流程。\n\n"
        + "\n".join(f"- [{title}]({name}/README.md)" for name, title, _ in LEGACY_STEPS)
        + "\n\n00 是仿真参照，不是定位器输入。03 的扰动样本进入 07；04–06 使用原始峰。"
        "\n07 是同一份 CSI 的内部扰动，不是 UE 的独立噪声重复。没有保存的历史中间值不重新计算来冒充原始输出。\n", encoding="utf-8")
    if run is None:
        return
    artifacts = run["artifacts"]
    folder = folders["00_scene_truth"]
    _copy(run["input_paths"]["scene_json"], folder / "scene_2d.json")
    _copy(run["input_paths"]["ground_truth"], folder / "ground_truth.npz")
    write_json(folder / "retained_paths.json", run["paths"])
    fig, ax = _map(plt, run, "00  场景、BS、UE 真值与实际参与合成的路径")
    seen = set()
    for path in run["paths"]:
        order = path["order"]
        ax.plot(*np.asarray(path["points"]).T, ls=["-", "--", "-."][order],
                color=["#4477AA", "#66A89F", "#AA7799"][order],
                label=["直射", "一次反射", "二次反射"][order] if order not in seen else None)
        seen.add(order)
    ax.legend(fontsize=7)
    _save(plt, fig, folder, "scene_paths")

    folder = folders["01_csi_input"]
    _copy(run["input_paths"]["online_measurement"], folder / "measurement.npz")
    _copy(run["config_path"], folder / "localization_config.json")
    with np.load(folder / "measurement.npz", allow_pickle=False) as data:
        csi = data["csi_observed"]
        frequencies = data["subcarrier_frequencies_hz"] / 1e6
    write_json(folder / "input_shape.json", dict(csi_shape=list(csi.shape), axes=["snapshot", "bs_antenna", "subcarrier"], truth_used=False))
    # 多快照分别画，避免只展示第一个却暗示全部。
    for i, snapshot in enumerate(csi):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
        for ax, values, title in zip(axes, (np.abs(snapshot), np.angle(snapshot)), ("幅度", "相位 / rad")):
            im = ax.pcolormesh(frequencies, np.arange(snapshot.shape[0]), values, shading="auto", cmap="viridis")
            ax.set(xlabel="子载波基带频率 / MHz", ylabel="BS 阵元编号", title=title)
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"01  在线 CSI，快照 {i}")
        _save(plt, fig, folder, f"csi_snapshot_{i:03d}")

    peaks = read_json(artifacts["music_peaks"])
    folder = folders["02_music"]
    _copy(artifacts["music_spectrum"], folder / "music_spectrum.npz")
    write_json(folder / "nominal_peaks.json", peaks["nominal"])
    _table(folder / "nominal_peaks.csv", [dict(observation_id=f"music_path_{i:02d}", aoa_local_deg=np.degrees(p["aoa_rad"]),
             delay_ns=p["delay_s"] * 1e9, spectrum_value=p["spectrum_value"]) for i, p in enumerate(peaks["nominal"])])
    with np.load(artifacts["music_spectrum"], allow_pickle=False) as data:
        spectrum, angles, delays = data["spectrum"], np.degrees(data["aoa_grid_rad"]), data["delay_grid_s"] * 1e9
    fig, ax = plt.subplots(figsize=(7.2, 5), layout="constrained")
    relative = 10 * np.log10(np.maximum(spectrum / spectrum.max(), 1e-12))
    im = ax.pcolormesh(delays, angles, relative, shading="auto", cmap="viridis", vmin=-60, vmax=0)
    fig.colorbar(im, ax=ax, label="相对 MUSIC 谱 / dB（非概率）")
    for i, peak in enumerate(peaks["nominal"]):
        xy = [peak["delay_s"] * 1e9, np.degrees(peak["aoa_rad"])]
        ax.scatter(*xy, marker="x", color="white", s=35)
        ax.annotate(f"P{i + 1}", xy, xytext=(4, 4), textcoords="offset points", color="white")
    ax.set(xlabel="观测时延 / ns（含未知偏差）", ylabel="阵列局部到达角 / °", title="02  二维 MUSIC 与原始峰")
    _save(plt, fig, folder, "music_spectrum")

    folder = folders["03_peak_perturbations"]
    _copy(artifacts["music_peaks"], folder / "music_peaks_and_associations.json")
    for key in ("nominal_observation_samples", "perturbed_observation_samples"):
        if key in peaks:
            write_json(folder / f"{key}.json", peaks[key])
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    for i, peak in enumerate(peaks["nominal"]):
        ax.scatter(peak["delay_s"] * 1e9, np.degrees(peak["aoa_rad"]), marker="*", s=70, label=f"原始峰 P{i + 1}")
    for repeat, association in enumerate(peaks["associations"]):
        for key, (angle, delay) in association["matches"].items():
            rows.append(dict(perturbation=repeat, nominal_index=int(key), state="matched", aoa_local_deg=np.degrees(angle), delay_ns=delay * 1e9))
            ax.scatter(delay * 1e9, np.degrees(angle), marker="o", s=25, facecolors="none", edgecolors="#4477AA")
        for peak in association["unmatched_peaks"]:
            rows.append(dict(perturbation=repeat, nominal_index=None, state="unmatched", aoa_local_deg=np.degrees(peak["aoa_rad"]), delay_ns=peak["delay_s"] * 1e9))
            ax.scatter(peak["delay_s"] * 1e9, np.degrees(peak["aoa_rad"]), marker="x", color="0.6", s=15)
        for key in association["missed_nominal_indices"]:
            rows.append(dict(perturbation=repeat, nominal_index=key, state="missed", aoa_local_deg=None, delay_ns=None))
    _table(folder / "perturbed_peaks.csv", rows)
    ax.set(xlabel="观测时延 / ns", ylabel="阵列局部到达角 / °", title="03  扰动峰配对：空心圆为已配对，灰叉为未配对")
    ax.legend(fontsize=7)
    _save(plt, fig, folder, "perturbed_peaks")

    for step, key, clustered, title in [("04_reverse_candidates", "raw_reverse_candidates", False, "04  原始峰反向追踪的全部候选"),
                                        ("05_first_clustering", "clustered_candidates", True, "05  第一次聚类的全部代表轨迹")]:
        folder = folders[step]
        _copy(artifacts[key], folder / f"{key}.json")
        _trajectories(plt, run, read_json(artifacts[key]), folder, clustered=clustered, title=title)

    folder = folders["06_joint_solution"]
    central = run["result"]["central_solution"]
    write_json(folder / "central_solution.json", {**central, "selected_candidates": run["result"]["central_selected_candidates"],
               "residuals_m": run["result"]["central_residuals_m"], "diagnostics": run["result"]["diagnostics"]})
    _position(plt, run, central, folder, "06  原始峰的联合求解输出（尚未汇总扰动解）")

    folder = folders["07_perturbation_solutions"]
    _copy(artifacts["bootstrap_diagnostics"], folder / "bootstrap_diagnostics.json")
    _copy(artifacts["uncertainty_solutions"], folder / "uncertainty_solutions.npz")
    diagnostics = read_json(artifacts["bootstrap_diagnostics"])
    diagnostic_rows = []
    for item in diagnostics:
        subdir = folder / f"perturbation_{item['repetition']:03d}"
        subdir.mkdir()
        write_json(subdir / "output.json", item)
        missing = "旧产物没有保存本次扰动的原始候选和聚类中间值；这里只展示当时保存的输出，不重新计算。"
        if "raw_reverse_candidates" in item:
            for key, clustered in [("raw_reverse_candidates", False), ("clustered_candidates", True)]:
                child = subdir / key
                child.mkdir()
                write_json(child / "output.json", item[key])
                _trajectories(plt, run, item[key], child, clustered=clustered, title=f"扰动 {item['repetition']}：{'聚类' if clustered else '反向候选'}")
            missing = "本次扰动的原始候选和聚类输出已保存。"
        (subdir / "README.md").write_text(missing + "\n", encoding="utf-8")
        diagnostic_rows.append(dict(perturbation=item["repetition"], solved=item["solved"],
                         x_m=item.get("position_m", [None, None])[0], y_m=item.get("position_m", [None, None])[1],
                         bias_ns=item["beta_m"] / SPEED_OF_LIGHT_M_S * 1e9 if item["solved"] else None,
                         reason=item.get("reason", "")))
    _table(folder / "perturbation_results.csv", diagnostic_rows)
    fig, ax = _map(plt, run, "07  CSI 内部扰动求解输出")
    solved = [item for item in diagnostics if item["solved"]]
    for i, item in enumerate(solved):
        ax.scatter(*item["position_m"], color="#4477AA", marker="x", label="扰动解" if i == 0 else None)
    points = np.array([run["true"], central["mu_m"], *[item["position_m"] for item in solved]])
    ax.set(xlim=(points[:, 0].min() - 1, points[:, 0].max() + 1), ylim=(points[:, 1].min() - 1, points[:, 1].max() + 1))
    ax.legend(fontsize=7)
    fig.suptitle(f"{len(solved)}/{len(diagnostics)} 次内部扰动得到完整解；相同位置的点可能重叠", fontsize=9)
    _save(plt, fig, folder, "perturbation_positions")

    folder = folders["08_final_evaluation"]
    _copy(artifacts["result"], folder / "localization_result.json")
    write_json(folder / "metrics.json", run["metrics"])
    write_csv(folder / "result.csv", [row])
    _position(plt, run, run["result"], folder, "08  最终位置与误差（扰动汇总之后）")
    write_json(directory / "sources.json", run["sources"])


def _export_scene(plt, run, folder):
    _copy(run["input_paths"]["scene_json"], folder / "scene_2d.json")
    _copy(run["input_paths"]["ground_truth"], folder / "ground_truth.npz")
    write_json(folder / "retained_paths.json", run["paths"])
    fig, ax = _map(plt, run, "00  场景、BS、UE 真值与合成路径（仅作参照）")
    seen = set()
    styles = ["-", "--", "-."]
    labels = ["直射", "一次反射", "二次反射"]
    for path in run["paths"]:
        order = path["order"]
        ax.plot(*np.asarray(path["points"]).T, ls=styles[min(order, 2)],
                color=["#4477AA", "#66A89F", "#AA7799"][min(order, 2)],
                label=labels[order] if order not in seen and order < 3 else None)
        seen.add(order)
    ax.legend(fontsize=7)
    _save(plt, fig, folder, "scene_paths")


def _export_csi(plt, run, folder):
    _copy(run["input_paths"]["online_measurement"], folder / "measurement.npz")
    if run.get("config_path"):
        _copy(run["config_path"], folder / "localization_config.json")
    with np.load(folder / "measurement.npz", allow_pickle=False) as data:
        csi = data["csi_observed"]
        frequencies = data["subcarrier_frequencies_hz"] / 1e6
    if csi.ndim == 2:
        csi = csi[None, ...]
    write_json(folder / "input_shape.json", dict(csi_shape=list(csi.shape),
               axes=["snapshot", "bs_antenna", "subcarrier"], truth_used=False,
               extra_csi_noise_in_localization=False))
    for i, snapshot in enumerate(csi):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
        for ax, values, title in zip(axes, (np.abs(snapshot), np.angle(snapshot)), ("幅度", "相位 / rad")):
            im = ax.pcolormesh(frequencies, np.arange(snapshot.shape[0]), values, shading="auto", cmap="viridis")
            ax.set(xlabel="子载波基带频率 / MHz", ylabel="BS 阵元编号", title=title)
            fig.colorbar(im, ax=ax)
        fig.suptitle(f"01  接收到的带噪 CSI，快照 {i}")
        _save(plt, fig, folder, f"csi_snapshot_{i:03d}")


def _export_music(plt, run, folder):
    artifacts = run["artifacts"]
    peaks = read_json(artifacts["music_peaks"])
    fine = _is_fine_workflow(run)
    indices = peaks.get("nominal_source_indices", list(range(len(peaks["nominal"]))))
    if len(indices) != len(peaks["nominal"]) or len(set(indices)) != len(indices):
        raise ValueError("正式谱峰的来源编号数量不符或重复")
    _copy(artifacts["music_spectrum"], folder / "music_spectrum.npz")
    _copy(artifacts["music_peaks"], folder / "music_peaks.json")
    _table(folder / "nominal_peaks.csv", [dict(observation_id=f"music_path_{i:02d}",
           aoa_local_deg=np.degrees(p["aoa_rad"]), delay_ns=p["delay_s"] * 1e9,
           spectrum_value=p["spectrum_value"]) for i, p in zip(indices, peaks["nominal"], strict=True)])
    if fine:
        _table(folder / "coarse_search_peaks.csv", [dict(source_index=i, aoa_local_deg=np.degrees(p["aoa_rad"]),
               delay_ns=p["delay_s"] * 1e9, spectrum_value=p["spectrum_value"])
               for i, p in enumerate(peaks.get("coarse", []))])
    with np.load(artifacts["music_spectrum"], allow_pickle=False) as data:
        spectrum, angles, delays = data["spectrum"], np.degrees(data["aoa_grid_rad"]), data["delay_grid_s"] * 1e9
    relative = 10 * np.log10(np.maximum(spectrum / max(float(spectrum.max()), np.finfo(float).tiny), 1e-12))
    fig, ax = plt.subplots(figsize=(7.2, 5), layout="constrained")
    im = ax.pcolormesh(delays, angles, relative, shading="auto", cmap="viridis", vmin=-60, vmax=0)
    fig.colorbar(im, ax=ax, label=("粗搜索 MUSIC 谱 / dB（非概率）" if fine else "相对 MUSIC 谱 / dB（非概率）"))
    for i, peak in zip(indices, peaks["nominal"], strict=True):
        xy = [peak["delay_s"] * 1e9, np.degrees(peak["aoa_rad"])]
        ax.scatter(*xy, marker="*" if fine else "x", color="#FFB000" if fine else "white", s=55 if fine else 35)
        ax.annotate(f"P{i + 1}", xy, xytext=(4, 4), textcoords="offset points", color="white")
    if fine:
        ax.scatter([], [], marker="*", color="#FFB000", s=55, label="局部细谱正式峰（位置见 03）")
        ax.legend(fontsize=7)
        fig.suptitle("底图只用于粗搜索；星号来自局部细谱，正式找峰与采样共用该细谱", fontsize=9)
    ax.set(xlabel="观测时延 / ns（含未知偏差）", ylabel="阵列局部到达角 / °",
           title="02  粗搜索谱与局部细谱正式峰" if fine else "02  同一份 CSI 的二维 MUSIC 谱")
    _save(plt, fig, folder, "music_spectrum")


def _export_spectrum_samples(plt, run, folder):
    source = run["artifacts"]["spectrum_samples"]
    _copy(source, folder / "spectrum_samples.json")
    payload = read_json(source)
    fine = _is_fine_workflow(run)
    samples = payload.get("samples", payload.get("records", []))
    rows = []
    for sample in samples:
        rows.append(dict(observation_id=sample["observation_id"], sample_id=sample["sample_id"],
                    source=sample.get("sampling_kind", sample.get("source", "")),
                    aoa_local_deg=float(np.degrees(sample["aoa_local_rad"])),
                    aoa_global_deg=float(np.degrees(sample["aoa_global_rad"])),
                    delay_ns=sample["delay_s"] * 1e9, spectrum_value=sample.get("spectrum_value")))
    _table(folder / "spectrum_samples.csv", rows)
    write_json(folder / "sampling_diagnostics.json", payload.get("diagnostics", {}))
    for index, region in enumerate(payload.get("regions", [])):
        source_index = int(region.get("peak_index", index))
        observation_id = region["observation_id"]
        members = [sample for sample in samples if sample["observation_id"] == observation_id]
        spectrum = np.asarray(region["spectrum"], dtype=float)
        angles = np.degrees(region["aoa_grid_rad"])
        delays = np.asarray(region["delay_grid_s"]) * 1e9
        relative = 10 * np.log10(np.maximum(spectrum / max(float(spectrum.max()), np.finfo(float).tiny), 1e-12))
        fig, ax = plt.subplots(figsize=(7.2, 5), layout="constrained")
        im = ax.pcolormesh(delays, angles, relative, shading="auto", cmap="viridis", vmin=-40, vmax=0)
        fig.colorbar(im, ax=ax, label="局部 MUSIC 谱 / dB（非概率）")
        for nominal, marker, color, label in [(False, ".", "#F2F2F2", "连续谱面采样"),
                                               (True, "*", "#FFB000", "粗网格初始峰")]:
            if fine and nominal:
                # 正式峰即使未作为额外参考样本插入，也始终标在真实细谱节点上。
                continue
            subset = [item for item in members if (item.get("sampling_kind", item.get("source")) == "nominal") == nominal]
            if subset:
                ax.scatter([item["delay_s"] * 1e9 for item in subset],
                           [np.degrees(item["aoa_local_rad"]) for item in subset], marker=marker,
                           s=75 if nominal else 10, color=color, label=label, alpha=.8)
        if fine:
            ax.scatter(region["nominal_delay_s"] * 1e9, np.degrees(region["nominal_aoa_local_rad"]),
                       marker="*", s=95, color="#FFB000", edgecolors="black", linewidths=.4,
                       label="正式峰（本张细谱的最大值）", zorder=9)
            if "coarse_delay_s" in region and "coarse_aoa_local_rad" in region:
                ax.scatter(region["coarse_delay_s"] * 1e9, np.degrees(region["coarse_aoa_local_rad"]),
                           marker="+", s=40, color="#CCCCCC", linewidths=.9, label="粗搜索点（仅确定区域）", zorder=8)
        unique = len({(item["aoa_local_rad"], item["delay_s"]) for item in members})
        ax.set(xlabel="观测时延 / ns（含共同偏差）", ylabel="阵列局部到达角 / °",
               title=f"03  观测 {source_index + 1}：{len(members)} 个样本，{unique} 个不同坐标")
        ax.legend(fontsize=7)
        fig.suptitle("样本用于候选搜索；点的分散程度不是定位置信区间", fontsize=9)
        _save(plt, fig, folder, f"local_spectrum_{source_index:03d}")
        if fine:
            probabilities = np.asarray(region["cell_probabilities"], dtype=float)
            fig, ax = plt.subplots(figsize=(7.2, 5), layout="constrained")
            im = ax.pcolormesh(delays, angles, probabilities, shading="flat", cmap="viridis")
            fig.colorbar(im, ax=ax, label="每个网格单元的采样概率")
            ax.scatter(region["nominal_delay_s"] * 1e9, np.degrees(region["nominal_aoa_local_rad"]),
                       marker="*", s=95, color="#FFB000", edgecolors="black", linewidths=.4,
                       label="同一细谱的正式峰", zorder=9)
            ax.set(xlabel="观测时延 / ns（含共同偏差）", ylabel="阵列局部到达角 / °",
                   title=f"03  观测 {source_index + 1}：由同一细谱生成的采样概率")
            ax.legend(fontsize=7)
            fig.suptitle("概率由单元四角谱值、面积及均匀混合项计算；仅用于候选搜索", fontsize=9)
            _save(plt, fig, folder, f"sampling_probabilities_{source_index:03d}")


def _export_clusters(plt, run, folder):
    clusters = read_json(run["artifacts"]["clustered_candidates"])
    _copy(run["artifacts"]["clustered_candidates"], folder / "clustered_candidates.json")
    _trajectories(plt, run, clusters, folder, clustered=True, title="05  第一次聚类的代表候选轨迹")
    raw = read_json(run["artifacts"]["raw_reverse_candidates"])
    raw_lookup = {(item["observation_id"], item["topology_id"], item["sample_id"]): item for item in raw}
    fig, ax = _map(plt, run, "05  聚类成员（细线）与代表候选（粗线）")
    member_rows, counts, extents = [], [], [run["true"], run["bs"]]
    for i, cluster in enumerate(clusters):
        color = plt.get_cmap("tab20")(i % 20)
        metadata = cluster["metadata"]
        topology_id = metadata.get("topology_id", cluster.get("topology_id"))
        if topology_id is None:
            raise ValueError(f"代表候选缺少反射路径编号：{cluster['candidate_id']}")
        sample_ids = metadata.get("source_sample_ids", [])
        for sample_id in sample_ids:
            item = raw_lookup.get((cluster["observation_id"], topology_id, sample_id))
            if item is None:
                raise ValueError(f"聚类成员没有对应的原始候选：{sample_id}")
            points = np.asarray(item["anchor_m"]) - np.asarray([item["beta_min_m"], item["beta_max_m"]])[:, None] * np.asarray(item["direction"])
            ax.plot(*points.T, color=color, lw=.5, alpha=.25)
            extents.extend(points.tolist())
            member_rows.append(dict(candidate_id=cluster["candidate_id"], observation_id=cluster["observation_id"],
                                    topology_id=topology_id, sample_id=sample_id,
                                    is_representative=sample_id == metadata.get("representative_sample_id")))
        points = np.asarray(cluster["anchor_m"]) - np.asarray(cluster["beta_interval_m"])[:, None] * np.asarray(cluster["direction"])
        ax.plot(*points.T, color=color, lw=2., alpha=.95)
        extents.extend(points.tolist())
        if len(clusters) <= 8:
            ax.annotate(f"C{i + 1} ({len(sample_ids)})", points.mean(axis=0), fontsize=6)
        counts.append(dict(label=f"C{i + 1}", candidate_id=cluster["candidate_id"], observation_id=cluster["observation_id"],
                           topology_id=topology_id, member_count=len(sample_ids),
                           representative_sample_id=metadata.get("representative_sample_id", "")))
    xy = np.asarray(extents)
    ax.set(xlim=(xy[:, 0].min() - 3, xy[:, 0].max() + 3), ylim=(xy[:, 1].min() - 3, xy[:, 1].max() + 3))
    ax.plot([], [], color="0.5", lw=.5, alpha=.5, label="簇内原始成员轨迹")
    ax.plot([], [], color="0.5", lw=2., label="保留的代表候选轨迹")
    ax.legend(fontsize=7)
    fig.suptitle(f"{len(raw)} 条原始候选 → {len(clusters)} 个代表；成员数见簇大小图和表；未代入最终 bias", fontsize=9)
    _save(plt, fig, folder, "members_and_representatives")
    _table(folder / "cluster_members.csv", member_rows)
    _table(folder / "cluster_counts.csv", counts)
    write_json(folder / "counts.json", dict(raw_count=len(raw), representative_count=len(clusters),
               listed_member_count=len(member_rows)))
    fig, ax = plt.subplots(figsize=(max(7.2, .3 * len(counts)), 4.), layout="constrained")
    if counts:
        sizes = [item["member_count"] for item in counts]
        bars = ax.bar([item["label"] for item in counts], sizes,
                      color=[plt.get_cmap("tab20")(i % 20) for i in range(len(counts))])
        ax.bar_label(bars, padding=3, fontsize=7)
        ax.set_ylim(0., max(sizes) * 1.18)
    ax.set(xlabel="代表候选编号（与表一致）", ylabel="簇内原始候选数",
           title="05  聚类压缩数量；颜色与成员及代表轨迹图一致")
    ax.tick_params(axis="x", labelsize=7, rotation=45 if len(counts) > 12 else 0)
    fig.suptitle("成员数用于说明候选压缩，不表示独立观测数或求解权重", fontsize=9)
    _save(plt, fig, folder, "cluster_sizes")


def _export_forward_check(plt, run, folder):
    step_number = folder.name.split("_", 1)[0]
    source = run["artifacts"]["forward_check"]
    _copy(source, folder / "forward_check.json")
    diagnostics = read_json(source)
    peak_label = "正式细谱峰" if _is_fine_workflow(run) else "原始谱峰"
    scalar_rows = [dict(field=key, value=value) for key, value in diagnostics.items()
                   if not isinstance(value, (list, dict))]
    _table(folder / "diagnostics.csv", scalar_rows)
    paths = diagnostics.get("paths", diagnostics.get("per_path", []))
    if paths:
        path_rows = []
        for item in paths:
            prediction = item.get("prediction") or {}
            original = item.get("original_peak_residuals") or {}
            sampled = item.get("sample_residuals") or {}
            path_rows.append(dict(observation_id=item.get("observation_id"), candidate_id=item.get("candidate_id"),
                        valid=item.get("valid"), failure_reasons=" | ".join(item.get("failure_reasons", [])),
                        predicted_aoa_global_deg=prediction.get("aoa_global_deg"),
                        predicted_observed_delay_ns=(prediction["predicted_observed_delay_s"] * 1e9
                                                     if prediction.get("predicted_observed_delay_s") is not None else None),
                        original_peak_aoa_error_deg=original.get("aoa_error_deg"),
                        original_peak_delay_error_ns=original.get("delay_error_ns"),
                        sampled_aoa_error_deg=sampled.get("aoa_error_deg"),
                        sampled_delay_error_ns=sampled.get("delay_error_ns")))
        _table(folder / "path_checks.csv", path_rows)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        x = np.arange(len(paths))
        for ax, key, title in zip(axes, ("original_peak_aoa_error_deg", "original_peak_delay_error_ns"),
                                 (f"预测 − {peak_label}角度 / °", f"预测 − {peak_label}时延 / ns")):
            values = [np.nan if item[key] is None else item[key] for item in path_rows]
            ax.bar(x, values, color=["#4477AA" if item["valid"] else "#BB5566" for item in path_rows])
            ax.axhline(0., color="0.4", lw=.7)
            ax.set_xticks(x, [f"P{i + 1}" for i in x])
            ax.set(xlabel="所选路径（顺序见表）", ylabel=title)
        fig.suptitle(f"{step_number}  所选反射路径正向重算；与{peak_label}比较，未使用真值", fontsize=10)
        _save(plt, fig, folder, "original_peak_residuals")
        fig, ax = _map(plt, run, f"{step_number}  联合解正向重算的所选路径")
        extents = [run["true"], run["bs"]]
        for i, item in enumerate(paths):
            points = np.asarray(item.get("path_nodes_m", []), dtype=float)
            if points.ndim != 2 or len(points) < 2:
                continue
            ax.plot(*points.T, lw=1., ls="-" if item.get("valid") else "--",
                    label=f"P{i + 1}：{'有效' if item.get('valid') else '未通过检查'}")
            extents.extend(points.tolist())
        xy = np.asarray(extents)
        ax.set(xlim=(xy[:, 0].min() - 2, xy[:, 0].max() + 2), ylim=(xy[:, 1].min() - 2, xy[:, 1].max() + 2))
        ax.legend(fontsize=7)
        fig.suptitle("检查已选路径的几何与观测残差；真值星号仅供画图参照", fontsize=8)
        _save(plt, fig, folder, "predicted_paths")
    fig, ax = plt.subplots(figsize=(9, max(3., .28 * min(len(scalar_rows), 20) + 1.5)), layout="constrained")
    ax.axis("off")
    displayed = scalar_rows[:20]
    if displayed:
        table = ax.table(cellText=[[item["field"], str(item["value"])] for item in displayed],
                         colLabels=["检查字段", "本次输出"], cellLoc="left", loc="center", colWidths=[.55, .45])
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1., 1.3)
    else:
        ax.text(.5, .5, f"已记录 {len(paths)} 条路径检查，详见 JSON 和 CSV", ha="center", transform=ax.transAxes)
    ax.set_title(f"{step_number}  对原始观测的检查（不使用真实位置或真实 bias）")
    _save(plt, fig, folder, "forward_check")


def _point_row(point):
    return dict(observation_id=point["observation_id"], sample_id=point["sample_id"],
                topology_id=point["topology_id"], x_m=point["position_m"][0], y_m=point["position_m"][1],
                reference_bias_ns=point["reference_bias_s"] * 1e9,
                aoa_global_deg=float(np.degrees(point["observed_aoa_global_rad"])),
                observed_delay_ns=point["observed_delay_s"] * 1e9,
                reflection_wall_ids=" | ".join(point["reflection_wall_ids"]))


def _point_limits(ax, run, points):
    xy = np.asarray([run["true"], run["bs"], *[point["position_m"] for point in points]], dtype=float)
    ax.set(xlim=(xy[:, 0].min() - 2, xy[:, 0].max() + 2),
           ylim=(xy[:, 1].min() - 2, xy[:, 1].max() + 2))


def _export_initial_points(plt, run, folder):
    source = run["artifacts"]["initial_candidates"]
    _copy(source, folder / "initial_candidates.json")
    payload = read_json(source)
    points = payload["points"]
    _table(folder / "initial_candidates.csv", [_point_row(point) for point in points])
    write_json(folder / "rejected_samples.json", payload.get("rejected_samples", []))
    write_json(folder / "generation_diagnostics.json", payload.get("diagnostics", {}))
    fig, ax = _map(plt, run, "04  每个谱面样本反向追踪后的初始位置点")
    observation_ids = sorted({point["observation_id"] for point in points})
    indices = _observation_indices(run, observation_ids)
    groups = []
    # 颜色区分来源峰，同色的不同符号区分反射墙组合，避免把路径分支看成断开的单条轨迹。
    markers = ("o", "^", "s", "D", "v", "P", "X")
    for observation_id in observation_ids:
        index = indices[observation_id]
        members = [point for point in points if point["observation_id"] == observation_id]
        topologies = sorted({point["topology_id"] for point in members})
        for branch, topology_id in enumerate(topologies):
            group = [point for point in members if point["topology_id"] == topology_id]
            xy = np.asarray([point["position_m"] for point in group])
            label = f"观测 {index + 1} / 墙组合 {branch + 1}（{len(group)} 点）"
            ax.scatter(*xy.T, color=plt.get_cmap("tab10")(index % 10), marker=markers[branch % len(markers)],
                       s=16, alpha=.65, label=label, zorder=7)
            groups.append(dict(label=label, observation_id=observation_id, topology_id=topology_id,
                               point_count=len(group), reflection_wall_ids=" | ".join(group[0]["reflection_wall_ids"])))
    _point_limits(ax, run, points)
    ax.legend(fontsize=6, loc="best")
    fig.suptitle(f"参考 bias = {payload['reference_bias_s'] * 1e9:g} ns，只用于初始化；不是已知或估计的真实 bias；本步未生成轨迹", fontsize=8)
    _save(plt, fig, folder, "initial_points")
    _table(folder / "observation_topologies.csv", groups)


def _export_point_clusters(plt, run, folder):
    initial = read_json(run["artifacts"]["initial_candidates"])
    source = run["artifacts"]["representative_points"]
    _copy(source, folder / "representative_points.json")
    payload = read_json(source)
    fine = _is_fine_workflow(run)
    representatives = payload["representatives"]
    noise_points = payload.get("noise_points", []) if fine else []
    lookup = {(point["observation_id"], point["sample_id"]): point for point in initial["points"]}
    if len(lookup) != len(initial["points"]):
        raise ValueError("初始候选点的来源与样本编号重复")
    if payload["reference_bias_s"] != initial["reference_bias_s"]:
        raise ValueError("初始点与代表点使用的参考 bias 不一致")
    roles = {}
    if fine:
        for item in payload["memberships"]:
            key = (item["observation_id"], item["sample_id"])
            if key in roles or key not in lookup or item["role"] not in {"core", "border", "noise"}:
                raise ValueError("DBSCAN 逐点角色缺失、重复或无效")
            roles[key] = item
        if set(roles) != set(lookup):
            raise ValueError("DBSCAN 逐点角色没有覆盖全部初始点")
    fig, ax = _map(plt, run, "05  初始位置点聚类：圆点为成员，菱形为保留的真实代表点")
    membership, counts, representative_rows, assigned = [], [], [], set()
    clusters = {}
    seen_representative_ids = set()
    for representative in representatives:
        representative_id = representative["candidate_id"]
        if representative_id in seen_representative_ids:
            raise ValueError("代表编号重复")
        seen_representative_ids.add(representative_id)
        cluster_id = representative.get("metadata", {}).get("point_cluster_id", representative_id)
        clusters.setdefault(cluster_id, []).append(representative)
    for index, (candidate_id, cluster_representatives) in enumerate(clusters.items()):
        representative = cluster_representatives[0]
        point, members = representative["point"], representative["members"]
        representative_sample_ids = {item["point"]["sample_id"] for item in cluster_representatives}
        if len(representative_sample_ids) != len(cluster_representatives):
            raise ValueError("同一簇重复保存同一个代表成员")
        for item in cluster_representatives:
            if item["members"] != members or item["point"] not in members:
                raise ValueError("同簇多代表的成员列表不一致，或代表不是实际成员")
        member_ids = {(member["observation_id"], member["sample_id"]) for member in members}
        if (len(member_ids) != len(members) or assigned.intersection(member_ids)
                or (point["observation_id"], point["sample_id"]) not in member_ids):
            raise ValueError(f"点簇成员重复或代表不是簇内成员：{candidate_id}")
        for member in members:
            key = (member["observation_id"], member["sample_id"])
            if key not in lookup or lookup[key] != member:
                raise ValueError(f"点簇成员与初始候选点不一致：{member['sample_id']}")
            if member["observation_id"] != point["observation_id"] or member["topology_id"] != point["topology_id"]:
                raise ValueError(f"点簇混入不同来源峰或反射墙组合：{candidate_id}")
            if fine and (roles[key]["role"] not in {"core", "border"} or roles[key]["candidate_id"] != candidate_id):
                raise ValueError(f"DBSCAN 逐点角色与簇归属不一致：{member['sample_id']}")
            membership.append(dict(candidate_id=candidate_id, **_point_row(member),
                                   is_representative=member["sample_id"] in representative_sample_ids,
                                   **({"role": roles[key]["role"]} if fine else {})))
        if lookup[(point["observation_id"], point["sample_id"])] != point:
            raise ValueError(f"保留的代表点被改写：{candidate_id}")
        assigned.update(member_ids)
        color = plt.get_cmap("tab20")(index % 20)
        xy = np.asarray([member["position_m"] for member in members])
        ax.scatter(*xy.T, s=17, color=color, alpha=.5, zorder=7)
        for item in cluster_representatives:
            actual_point = item["point"]
            ax.scatter(*actual_point["position_m"], s=65, marker="D", color=color, edgecolors="black", linewidths=.7, zorder=9)
            representative_rows.append(dict(candidate_id=item["candidate_id"], point_cluster_id=candidate_id,
                                            **_point_row(actual_point)))
        counts.append(dict(label=f"C{index + 1}", candidate_id=candidate_id, observation_id=point["observation_id"],
                           topology_id=point["topology_id"], member_count=len(members), representative_sample_id=point["sample_id"],
                           representative_count=len(cluster_representatives)))
    noise_rows, noise_ids = [], set()
    for point in noise_points:
        key = (point["observation_id"], point["sample_id"])
        if (key not in lookup or lookup[key] != point or key in assigned or key in noise_ids
                or roles[key]["role"] != "noise" or roles[key]["candidate_id"] is not None):
            raise ValueError("DBSCAN 离群点与初始点或逐点角色不一致")
        noise_ids.add(key)
        noise_rows.append(dict(candidate_id=None, **_point_row(point), is_representative=False, role="noise"))
    if assigned | noise_ids != set(lookup):
        raise ValueError("点聚类没有完整覆盖已保存的初始候选点")
    if noise_points:
        xy = np.asarray([point["position_m"] for point in noise_points])
        ax.scatter(*xy.T, s=23, marker="x", color="0.55", alpha=.75, zorder=8,
                   label=f"离群点（{len(noise_points)} 点，不产生代表）")
    _point_limits(ax, run, initial["points"])
    ax.scatter([], [], color="0.5", s=17, alpha=.5, label="簇内初始位置点")
    ax.scatter([], [], color="0.5", marker="D", s=65, edgecolors="black", label="实际成员中的代表点")
    ax.legend(fontsize=7)
    detail = (f"DBSCAN；{len(noise_points)} 个离群点单独保留" if fine else "旧版按簇内最大跨度划分")
    fig.suptitle(f"{len(initial['points'])} 个初始点 → {len(representatives)} 个代表点；{detail}；尚未建立轨迹", fontsize=8)
    _save(plt, fig, folder, "members_and_representatives")
    _table(folder / "cluster_members.csv", membership)
    _table(folder / "cluster_counts.csv", counts)
    _table(folder / "representative_points.csv", representative_rows)
    if fine:
        write_json(folder / "noise_points.json", noise_points)
        write_json(folder / "clustering_diagnostics.json", payload["diagnostics"])
        # 无离群点时仍输出表头，明确表示本次检查过离群点，而非报告漏了一步。
        point_fields = list(_point_row(initial["points"][0])) if initial["points"] else [
            "observation_id", "sample_id", "topology_id", "x_m", "y_m", "reference_bias_ns",
            "aoa_global_deg", "observed_delay_ns", "reflection_wall_ids"]
        fields = ["candidate_id", *point_fields, "is_representative", "role"]
        for filename, rows in [("noise_points.csv", noise_rows), ("point_memberships.csv", membership + noise_rows)]:
            with (folder / filename).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    write_json(folder / "counts.json", dict(raw_count=len(initial["points"]), representative_count=len(representatives),
               **({"cluster_count": len(clusters)} if run.get("result", {}).get("workflow", run.get("workflow")) == DIFFRACTION_WORKFLOW else {}),
               listed_member_count=len(membership), clustering_space="initial_xy_at_reference_bias",
               reference_bias_s=payload["reference_bias_s"],
               **({"noise_count": len(noise_points), "clustering_algorithm": "dbscan"} if fine else {})))
    fig, ax = plt.subplots(figsize=(max(7.2, .3 * len(counts)), 4.), layout="constrained")
    if counts:
        sizes = [item["member_count"] for item in counts]
        bars = ax.bar([item["label"] for item in counts], sizes,
                      color=[plt.get_cmap("tab20")(i % 20) for i in range(len(counts))])
        ax.bar_label(bars, padding=3, fontsize=7)
        ax.set_ylim(0., max(sizes) * 1.18)
    ax.set(xlabel="点簇编号（与表一致）", ylabel="簇内初始位置点数", title="05  初始位置点聚类压缩数量")
    ax.tick_params(axis="x", labelsize=7, rotation=45 if len(counts) > 12 else 0)
    fig.suptitle("成员数用于说明点集压缩，不表示独立观测数或求解权重", fontsize=9)
    _save(plt, fig, folder, "cluster_sizes")


def _export_point_steps(plt, run, directory, row):
    directory.mkdir(parents=True, exist_ok=False)
    workflow = (run.get("result", {}).get("workflow", run.get("workflow")) if run else row.get("workflow", FINE_WORKFLOW))
    fine = workflow in {FINE_WORKFLOW, DIFFRACTION_WORKFLOW}
    steps = FINE_STEPS if fine else POINT_STEPS
    artifacts = run.get("artifacts", {}) if run else {}
    available = {name: False for name, _, _ in steps}
    if run:
        available.update({
            "00_scene_truth": all(key in run.get("input_paths", {}) for key in ("scene_json", "ground_truth")),
            "01_csi_input": "online_measurement" in run.get("input_paths", {}),
            "02_music": all(key in artifacts for key in ("music_spectrum", "music_peaks")),
            "03_spectrum_sampling": "spectrum_samples" in artifacts,
            "04_initial_candidates": "initial_candidates" in artifacts,
            "05_point_clustering": all(key in artifacts for key in ("initial_candidates", "representative_points")),
            "06_representative_trajectories": all(key in artifacts for key in ("representative_points", "representative_trajectories")),
            "07_joint_solution": bool(run.get("result", {}).get("central_solution")),
            "08_forward_check": "forward_check" in artifacts,
            "09_final_evaluation": bool(run.get("metrics")) and "result" in artifacts,
        })
    notes = {
        "04_initial_candidates": "本步是谱面样本在公开参考 bias 下反向追踪到的实际 XY 端点。参考 bias 只是初始化参数，不是真值，也不是联合估计结果；非法样本单独记录。没有先建立完整候选轨迹。",
        "05_point_clustering": "旧版 v2：在同一来源峰、同一反射墙组合内，按簇内任意两点的最大距离进行贪心划分，选簇内真实成员为代表。此时不比较轨迹距离、不使用轨迹方向或最终估计 bias。这份旧结果不按 DBSCAN 重新解释。成员图的颜色与 cluster_sizes 一致，簇编号及坐标见 CSV。",
        "06_representative_trajectories": "只为上一步保留的代表点建立 bias—位置关系；画出完整合法 bias 范围。其他簇成员不进入轨迹生成。颜色区分来源观测，具体反射墙组合见表。",
    }
    if fine:
        notes.update({
            "02_music": "底图保存的是粗分辨率搜索谱，用于定位局部细谱的搜索区域；星号和 nominal_peaks.csv 是局部细谱重新确定的正式峰。粗搜索点另存 coarse_search_peaks.csv。正式峰可能不在粗底图的最高网格，03 展示它实际对应的细谱。",
            "03_spectrum_sampling": "正式峰取自本张局部细谱的最大值节点。采样概率由同一份细谱每个单元的四角平均谱值、面积及均匀混合项计算，不额外换用其他分辨率的谱。星号始终表示正式细峰，加号仅表示粗搜索点；即使不把正式峰加入候选样本，图上也保留该星号。",
            "05_point_clustering": "v3：在同一来源峰、同一反射墙面顺序内部，对初始 XY 点做 DBSCAN。eps 限制相邻距离，不限制整个簇的长度；min_samples 包含点自身。每簇只保留一个真实成员作为代表。离群点用灰色叉号显示，独立保存到 noise_points.csv，不合并成一个簇，也不各自生成代表。point_memberships.csv 逐点记录 core（核心点）、border（边界点）或 noise（离群点）。所有角色合计必须覆盖原始点集；参数和分组数量见 clustering_diagnostics.json。此时尚未生成轨迹，也不使用真实位置或真实 bias。",
        })
    if workflow == DIFFRACTION_WORKFLOW:
        notes["05_point_clustering"] = (
            "v4：同一来源峰、完整反射与绕射顺序内做 DBSCAN。反射簇保留一个实际成员；"
            "绕射簇按覆盖距离保留多个实际成员，再通过解析区间并集补足整个合法偏差范围的覆盖。"
            "同一簇的成员只统计一次；代表增加不会增加独立观测数或求解权重。"
            "覆盖只针对已生成成员，不保证未采样区域覆盖或最终定位精度。")
    folders, index = {}, []
    for name, title, function in steps:
        folder = directory / name
        folder.mkdir()
        folders[name] = folder
        state = "available" if available[name] else "not_available"
        note = ("只导出本次运行保存的输出，不重新执行定位。" if available[name] else
                f"状态：{row['status']}。本步骤没有保存可核验的输出。{row.get('error', '')}")
        (folder / "README.md").write_text(f"# {title}\n\n对应函数：`{function}`。\n\n{note}\n\n{notes.get(name, '')}\n", encoding="utf-8")
        index.append(dict(step=name, title=title, function=function, status=state))
    write_json(directory / "attempt.json", row)
    write_json(directory / "step_index.json", index)
    (directory / "README.md").write_text(
        "# 按执行步骤查看本次定位\n\n"
        + ("未记录工作流来源；下列为当前流程占位，不表示这些步骤已经运行。\n\n" if run is None and "workflow" not in row else "")
        + ("工作流 v4：带噪 CSI → 统一细谱采样 → 反射与一次绕射初始点 → DBSCAN → 绕射簇按覆盖距离选多个实际代表 → 代表轨迹 → 共享偏差求解。\n\n" if workflow == DIFFRACTION_WORKFLOW else "工作流 v3：带噪 CSI → 粗谱搜索区域 → 局部细谱正式找峰与采样 → 反向追踪初始点 → DBSCAN 与每簇一个真实代表 → 仅为代表点建立轨迹 → 一次联合求解。\n\n" if fine else
           "旧工作流 v2：带噪 CSI → 粗谱峰附近的细谱采样 → 反向追踪初始位置点 → 按簇内最大跨度划分与真实代表点 → 仅为代表点建立轨迹 → 一次联合求解。\n\n")
        + "\n".join(f"- [{title}]({name}/README.md)" for name, title, _ in steps)
        + "\n\n04、05 只展示初始位置点，06 才展示代表点的偏差—位置轨迹。参考 bias 是公开初始化参数，不读取真实 bias，也不是最终估计值。"
        "\n01 的 CSI 已带噪，定位器不额外加噪；03 的样本用于候选搜索，不表示独立观测或定位置信区间。"
        "\n00、09 在独立评估侧读取真值；08 正向检查不使用真值。失败时已保存的步骤照常导出。\n", encoding="utf-8")
    if run is None:
        return
    for name, exporter in [("00_scene_truth", _export_scene), ("01_csi_input", _export_csi),
                           ("02_music", _export_music), ("03_spectrum_sampling", _export_spectrum_samples),
                           ("04_initial_candidates", _export_initial_points), ("05_point_clustering", _export_point_clusters),
                           ("08_forward_check", _export_forward_check)]:
        if available[name]:
            exporter(plt, run, folders[name])
    if available["06_representative_trajectories"]:
        folder = folders["06_representative_trajectories"]
        _copy(artifacts["representative_trajectories"], folder / "representative_trajectories.json")
        trajectories = read_json(artifacts["representative_trajectories"])
        representatives = read_json(artifacts["representative_points"])["representatives"]
        lookup = {representative["candidate_id"]: representative for representative in representatives}
        trajectory_ids = {trajectory["candidate_id"] for trajectory in trajectories}
        if len(trajectory_ids) != len(trajectories) or trajectory_ids != set(lookup):
            raise ValueError("代表轨迹与上一步代表点没有一一对应")
        mapping = []
        for trajectory in trajectories:
            point = lookup[trajectory["candidate_id"]]["point"]
            metadata = trajectory["metadata"]
            if (metadata["representative_sample_id"] != point["sample_id"]
                    or trajectory["observation_id"] != point["observation_id"]
                    or metadata["initial_position_m"] != point["position_m"]
                    or metadata["reflection_wall_ids"] != point["reflection_wall_ids"]):
                raise ValueError("代表轨迹的来源或初始位置与代表点不一致")
            at_reference = np.asarray(trajectory["anchor_m"]) - point["reference_bias_s"] * SPEED_OF_LIGHT_M_S * np.asarray(trajectory["direction"])
            if not np.allclose(at_reference, point["position_m"], rtol=0., atol=1e-8):
                raise ValueError("代表轨迹在参考 bias 下没有经过代表点")
            mapping.append(dict(candidate_id=trajectory["candidate_id"], **_point_row(point)))
        _table(folder / "representative_to_trajectory.csv", mapping)
        _trajectories(plt, run, trajectories, folder, clustered=True, title="06  仅由聚类代表点生成的合法偏差—位置轨迹",
                      show_initial_points=True)
    if available["07_joint_solution"]:
        folder = folders["07_joint_solution"]
        result = run["result"]
        central = result["central_solution"]
        write_json(folder / "joint_solution.json", {**central,
                   "selected_candidates": result.get("central_selected_candidates", []),
                   "residuals_m": result.get("central_residuals_m", []), "diagnostics": result.get("diagnostics", {})})
        _position(plt, run, central, folder, "07  代表轨迹的唯一联合求解结果")
    if available["09_final_evaluation"]:
        folder = folders["09_final_evaluation"]
        _copy(artifacts["result"], folder / "localization_result.json")
        write_json(folder / "metrics.json", run["metrics"])
        write_csv(folder / "result.csv", [row])
        _position(plt, run, run["result"], folder, "09  联合解的独立真值评价")
    write_json(directory / "sources.json", run.get("sources", []))


def export_steps(plt, run, directory: Path, row: dict) -> None:
    """按保存的工作流导出步骤；失败运行也保留已完成阶段。"""
    workflow = (run.get("result", {}).get("workflow", run.get("workflow"))
                if run else row.get("workflow", FINE_WORKFLOW))
    if workflow in {POINT_WORKFLOW, FINE_WORKFLOW, DIFFRACTION_WORKFLOW}:
        return _export_point_steps(plt, run, directory, row)
    if workflow != WORKFLOW:
        if workflow is not None:
            raise ValueError(f"未知的定位工作流：{workflow}")
        return _export_legacy_steps(plt, run, directory, row)
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = run.get("artifacts", {}) if run else {}
    available = {name: False for name, _, _ in STEPS}
    if run:
        available.update({
            "00_scene_truth": all(key in run.get("input_paths", {}) for key in ("scene_json", "ground_truth")),
            "01_csi_input": "online_measurement" in run.get("input_paths", {}),
            "02_music": all(key in artifacts for key in ("music_spectrum", "music_peaks")),
            "03_spectrum_sampling": "spectrum_samples" in artifacts,
            "04_reverse_candidates": "raw_reverse_candidates" in artifacts,
            "05_first_clustering": all(key in artifacts for key in ("raw_reverse_candidates", "clustered_candidates")),
            "06_joint_solution": bool(run.get("result", {}).get("central_solution")),
            "07_forward_check": "forward_check" in artifacts,
            "08_final_evaluation": bool(run.get("metrics")) and "result" in artifacts,
        })
    folders, index = {}, []
    for name, title, function in STEPS:
        folder = directory / name
        folder.mkdir()
        folders[name] = folder
        note = ("只导出本次运行保存的输出，不重新执行定位。" if available[name] else
                f"状态：{row['status']}。本步骤没有保存可核验的输出。{row.get('error', '')}")
        (folder / "README.md").write_text(f"# {title}\n\n对应函数：`{function}`。\n\n{note}\n", encoding="utf-8")
        index.append(dict(step=name, title=title, function=function,
                          status="available" if available[name] else "not_available"))
    write_json(directory / "attempt.json", row)
    write_json(directory / "step_index.json", index)
    (directory / "README.md").write_text(
        "# 按执行步骤查看本次定位\n\n"
        + ("未记录工作流来源；下列为当前步骤占位，不表示这些步骤已经运行。\n\n"
           if run is None and "workflow" not in row else "")
        + "工作流：谱面采样 → 汇集反向候选 → 聚类代表 → 一次联合求解。\n\n"
        + "\n".join(f"- [{title}]({name}/README.md)" for name, title, _ in STEPS)
        + "\n\n01 的 CSI 已带噪；定位器不再额外加噪。03 的所有谱面样本进入 04，05 汇集后聚类，06 联合求解共同 bias 与位置。"
        "\n03 的散点用于候选搜索，不是独立观测或置信区间。候选轨迹保留 bias 变化，不用真值 bias 画初始位置。"
        "\n00、08 在独立评估侧读取真值；07 只检查原始观测。失败步骤保留状态，已完成步骤照常导出。\n",
        encoding="utf-8")
    if run is None:
        return
    for name, exporter in [("00_scene_truth", _export_scene), ("01_csi_input", _export_csi),
                           ("02_music", _export_music), ("03_spectrum_sampling", _export_spectrum_samples),
                           ("05_first_clustering", _export_clusters), ("07_forward_check", _export_forward_check)]:
        if available[name]:
            exporter(plt, run, folders[name])
    if available["04_reverse_candidates"]:
        folder = folders["04_reverse_candidates"]
        _copy(artifacts["raw_reverse_candidates"], folder / "raw_reverse_candidates.json")
        _trajectories(plt, run, read_json(artifacts["raw_reverse_candidates"]), folder, clustered=False,
                      title="04  全部谱面样本反向追踪的候选轨迹")
    if available["06_joint_solution"]:
        folder = folders["06_joint_solution"]
        result = run["result"]
        central = result["central_solution"]
        write_json(folder / "joint_solution.json", {**central,
                   "selected_candidates": result.get("central_selected_candidates", []),
                   "residuals_m": result.get("central_residuals_m", []), "diagnostics": result.get("diagnostics", {})})
        _position(plt, run, central, folder, "06  聚类代表候选的唯一联合求解结果")
    if available["08_final_evaluation"]:
        folder = folders["08_final_evaluation"]
        _copy(artifacts["result"], folder / "localization_result.json")
        write_json(folder / "metrics.json", run["metrics"])
        write_csv(folder / "result.csv", [row])
        _position(plt, run, run["result"], folder, "08  联合解的独立真值评价")
    write_json(directory / "sources.json", run.get("sources", []))


def per_sample_statistics(rows):
    result = []
    for ue_id in dict.fromkeys(row["ue_id"] for row in rows):
        group = [row for row in rows if row["ue_id"] == ue_id]
        errors = [row["position_error_m"] for row in group if row["status"] == "success"]
        result.append(dict(ue_id=ue_id, true_x_m=group[0]["true_x_m"], true_y_m=group[0]["true_y_m"],
                           planned_count=len(group), solved_count=len(errors),
                           failed_count=sum(row["status"].endswith("failed") for row in group),
                           pending_count=sum(row["status"] == "pending" for row in group),
                           median_error_m=float(np.median(errors)) if errors else None,
                           p90_error_m=float(np.quantile(errors, .9)) if errors else None))
    return result


def export_summary(plt, rows, summary, directory):
    directory.mkdir(parents=True, exist_ok=True)
    samples = per_sample_statistics(rows)
    write_csv(directory / "per_attempt.csv", rows)
    write_csv(directory / "per_sample.csv", samples)
    write_csv(directory / "summary.csv", [summary])
    write_json(directory / "summary.json", summary)
    errors = np.array([row["position_error_m"] for row in rows if row["status"] == "success"])
    # 标准经验 CDF 以有误差值的成功尝试为分母。失败数量同时展示。
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    if len(errors):
        ordered = np.sort(errors)
        ax.step(np.r_[0., ordered], np.r_[0., np.arange(1, len(errors) + 1) / len(errors)], where="post", label=f"成功定位误差 CDF，n={len(errors)}", color="#4477AA")
        for quantile, name, color in [(.5, "Med", "#66A89F"), (.9, "P90", "#AA7799")]:
            value = float(np.quantile(errors, quantile))
            ax.axvline(value, ls="--", lw=1, color=color, label=f"{name} = {value:.4f} m")
        write_csv(directory / "cdf_source.csv", [dict(position_error_m=float(value), cumulative_fraction=(i + 1) / len(errors)) for i, value in enumerate(ordered)])
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, "没有可计算误差的成功结果", ha="center", transform=ax.transAxes)
    ax.set(xlabel="定位欧式距离误差 / m", ylabel="累计比例（CDF）", ylim=(0, 1.02), title="全部 sample 的定位误差 CDF")
    fig.suptitle(f"{len(samples)} 个 UE；成功 {len(errors)}/{len(rows)} 次；失败 {summary['failed_count']}，待运行 {summary['pending_count']}", fontsize=9)
    _save(plt, fig, directory, "position_error_cdf")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    if len(errors):
        values = [float(np.median(errors)), float(np.quantile(errors, .9))]
        axes[0].bar(["Med", "P90"], values, color=["#66A89F", "#AA7799"])
        for i, value in enumerate(values):
            axes[0].annotate(f"{value:.4f}", (i, value), ha="center", xytext=(0, 4), textcoords="offset points")
        axes[0].set_ylim(0, max(max(values) * 1.25, .001))
    for key, label, marker in [("median_error_m", "每个 UE 的 Med", "o"), ("p90_error_m", "每个 UE 的 P90", "^")]:
        axes[1].plot(range(len(samples)), [np.nan if row[key] is None else row[key] for row in samples], marker=marker, ls="none", ms=4, label=label)
    axes[0].set(title="全部成功尝试的 Med / P90", ylabel="定位误差 / m")
    axes[1].set(title="每个 sample 的噪声重复统计", ylabel="定位误差 / m", xlabel="UE sample 编号")
    axes[1].set_xticks(range(len(samples)), [row["ue_id"] for row in samples], rotation=90, fontsize=6)
    axes[1].legend(fontsize=7)
    _save(plt, fig, directory, "position_error_med_p90")
    (directory / "README.md").write_text(
        "# 全部 sample 汇总\n\nCDF、整体 Med/P90 使用所有成功定位尝试的欧式距离误差；"
        "每个 UE 的噪声重复分别计入，同时列出成功、失败、待运行数。"
        "\nper_sample.csv 和右侧图按 UE 分组，在该 UE 的成功噪声重复内计算 Med/P90。"
        "\n失败项没有误差，不能作为零填入 CDF；不是把五次重复的估计坐标先平均再计算误差。"
        "\n所有分位数使用 numpy.quantile 的默认线性插值；CDF 是经验累计分布，二者的小样本插值约定不同。\n", encoding="utf-8")
    return samples
