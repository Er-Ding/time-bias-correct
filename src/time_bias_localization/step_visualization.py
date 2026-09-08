"""按 UE / 噪声重复 / 实际算法步骤组织可复查的图表和原始输出。"""

from pathlib import Path
import shutil

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S
from .visualization import _save, _scene_axes, read_json, write_csv, write_json


STEPS = [
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


def _trajectories(plt, run, candidates, directory, *, clustered, title):
    fig, ax = _map(plt, run, title)
    ids = sorted({item["observation_id"] for item in candidates})
    colors = {key: plt.get_cmap("tab10")(i % 10) for i, key in enumerate(ids)}
    rows, extents = [], [run["true"], run["bs"]]
    for index, item in enumerate(candidates):
        interval = item["beta_interval_m"] if clustered else [item["beta_min_m"], item["beta_max_m"]]
        endpoints = np.asarray(item["anchor_m"]) - np.asarray(interval)[:, None] * np.asarray(item["direction"])
        extents.extend(endpoints.tolist())
        metadata = item["metadata"] if clustered else item
        order = len(metadata["reflection_wall_ids"])
        label = f"{'C' if clustered else 'R'}{index + 1}"
        ax.plot(*endpoints.T, color=colors[item["observation_id"]], ls=["-", "--", "-."][order], lw=1.5 if clustered else .9, alpha=.8)
        ax.annotate(label, endpoints.mean(axis=0), fontsize=6, xytext=(3, 3), textcoords="offset points")
        rows.append(dict(label=label, observation_id=item["observation_id"],
                         candidate_id=item.get("candidate_id", ""), sample_id=item.get("sample_id", ""),
                         reflection_order=order, reflection_wall_ids=" | ".join(metadata["reflection_wall_ids"]),
                         anchor_x_m=item["anchor_m"][0], anchor_y_m=item["anchor_m"][1],
                         direction_x=item["direction"][0], direction_y=item["direction"][1],
                         bias_min_ns=interval[0] / SPEED_OF_LIGHT_M_S * 1e9,
                         bias_max_ns=interval[1] / SPEED_OF_LIGHT_M_S * 1e9,
                         member_count=metadata.get("raw_count", 1)))
    for i, key in enumerate(ids):
        ax.plot([], [], color=colors[key], label=f"观测 {i + 1}")
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
    ax.add_patch(Ellipse(estimate, 2 * radii[1], 2 * radii[0], angle=angle, fill=False,
                        color="#4477AA", label="名义 95% 椭圆（未校准）"))
    ax.scatter(*estimate, marker="x", s=50, color="#4477AA", label="本步骤位置输出", zorder=9)
    ax.plot([true[0], estimate[0]], [true[1], estimate[1]], color="0.45", lw=.8)
    margin = max(1., np.linalg.norm(estimate - true), radii.max()) * 1.5
    center = (estimate + true) / 2
    ax.set(xlim=(center[0] - margin, center[0] + margin), ylim=(center[1] - margin, center[1] + margin))
    ax.legend(fontsize=7)
    fig.suptitle(f"偏差输出 {solution['clock_bias_s'] * 1e9:.4f} ns；与真值距离 {np.linalg.norm(estimate - true):.4f} m", fontsize=9)
    _save(plt, fig, directory, "position")


def export_steps(plt, run, directory: Path, row: dict) -> None:
    """每一步保存原始数据、图及函数说明，旧产物缺失处明确标注。"""
    directory.mkdir(parents=True, exist_ok=False)
    folders = {}
    index = []
    for name, title, function in STEPS:
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
        + "\n".join(f"- [{title}]({name}/README.md)" for name, title, _ in STEPS)
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
