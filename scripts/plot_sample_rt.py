"""将已有 RT 核验结果画在二维场景中；只读输入，不运行定位或改动候选。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=103)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    scene_path, audit_root, output = args.scene.resolve(), args.audit_root.resolve(), args.output.resolve()
    sample_id = f"SAMPLE_{args.sample:06d}"
    sample_dir = audit_root / sample_id
    files = [scene_path, sample_dir / "propagation_hypotheses.json",
             sample_dir / "offline_fixed_position_geometry.json", audit_root / "provenance.json"]
    inputs = {str(path): digest(path) for path in files}
    scene, bank, checked, audit = [json.loads(path.read_text()) for path in files]
    if audit["inputs_sha256"].get(str(scene_path)) != inputs[str(scene_path)]:
        raise ValueError("场景与 RT 核验记录使用的场景不一致。")
    paths = checked["valid_paths"]
    if not paths or len(paths) != checked["valid_hypotheses"]:
        raise ValueError("该样本没有可绘制路径，或有效路径数量记录不一致。")
    by_id = {item["hypothesis_id"]: item for item in bank["hypotheses"]}
    for path in paths:
        nodes = np.asarray(path["nodes_m"], float)
        if nodes.shape != (len(path["interactions"]) + 2, 2) or not np.isfinite(nodes).all():
            raise ValueError("路径节点格式错误。")
        np.testing.assert_allclose(nodes[0], checked["position_m"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(nodes[-1], bank["receiver_m"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum(),
                                   path["length_m"], atol=1e-6, rtol=0)
        if by_id[path["hypothesis_id"]]["interactions_ue_to_bs"] != path["interactions"]:
            raise ValueError("已保存路径与候选中的反射／绕射顺序不一致。")
    output.mkdir(parents=True, exist_ok=False)
    available = {item.name for item in font_manager.fontManager.ttflist}
    fonts = [name for name in ("WenQuanYi Micro Hei", "Noto Sans CJK SC", "Droid Sans Fallback")
             if name in available]
    if not fonts:
        raise RuntimeError("没有可用的中文绘图字体。")
    plt.rcParams.update({"font.family": fonts[0], "axes.unicode_minus": False,
                         "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    wall_segments = [[wall["start_m"], wall["end_m"]] for wall in scene["walls"]]
    wall_by_id = {wall["wall_id"]: wall for wall in scene["walls"]}
    colors = ["#1574B8", "#D66B16", "#7D459D", "#12856C"]
    all_nodes = np.concatenate([np.asarray(path["nodes_m"]) for path in paths])
    lower, upper = all_nodes.min(axis=0) - 13, all_nodes.max(axis=0) + 13
    full_bounds = scene["bounds_m"]
    fig, axes = plt.subplots(1, len(paths) + 1, figsize=(7 + 4 * len(paths), 8.3),
                             gridspec_kw={"width_ratios": [1.4] + [1] * len(paths)})
    fig.subplots_adjust(left=.055, right=.98, top=.82, bottom=.14, wspace=.25)
    fig.suptitle(f"样本 {args.sample:06d} · 当前二维 RT 的有效路径", fontsize=21, y=.97, weight="bold")
    ue, bs = np.asarray(checked["position_m"]), np.asarray(bank["receiver_m"])
    fig.text(.5, .914,
             f"固定真实位置核验：{len(bank['hypotheses'])} 个候选中有 {len(paths)} 条走通   |   "
             f"UE ({ue[0]:.2f}, {ue[1]:.2f}) m   ·   BS ({bs[0]:.2f}, {bs[1]:.2f}) m",
             ha="center", color="#46515B", fontsize=11)

    def endpoints(ax):
        for point, label, marker, color in ((ue, "UE", "*", "#198457"), (bs, "BS", "^", "#1B2430")):
            ax.scatter(*point, s=105 if marker == "*" else 64, marker=marker, color=color,
                       edgecolor="white", linewidth=.8, zorder=8)
            ax.annotate(label, point, xytext=(7, -15), textcoords="offset points", weight="bold",
                        color=color, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1.2}, zorder=9)

    def draw_path(ax, path, index, detailed):
        color, nodes = colors[index % len(colors)], np.asarray(path["nodes_m"])
        ax.plot(nodes[:, 0], nodes[:, 1], color=color, lw=2.1, zorder=5)
        for start, end in zip(nodes[:-1], nodes[1:]):
            ax.annotate("", xy=start + .61 * (end - start), xytext=start + .47 * (end - start),
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5, "mutation_scale": 11}, zorder=6)
        counts = {"reflection": 0, "diffraction": 0}
        for node, (kind, key) in zip(nodes[1:-1], path["interactions"]):
            counts[kind] += 1
            ax.scatter(*node, s=36, marker="o" if kind == "reflection" else "D", facecolor="white",
                       edgecolor=color, linewidth=1.5, zorder=7)
            if detailed:
                label = ("反射" if kind == "reflection" else "绕射") + str(counts[kind])
                ax.annotate(label, node, xytext=(8, 8 if kind == "reflection" else -15),
                            textcoords="offset points", color=color, fontsize=9,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1.2}, zorder=9)
                if kind == "reflection":
                    wall = wall_by_id[key]
                    ends = np.array([wall["start_m"], wall["end_m"]])
                    ax.plot(ends[:, 0], ends[:, 1], color=color, alpha=.48, lw=3, zorder=3)

    for index, ax in enumerate(axes):
        ax.set_facecolor("#FAFBFC")
        ax.add_collection(LineCollection(wall_segments, colors="#A9B1B8", linewidths=.8, zorder=2))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        ax.grid(color="#E4E8EB", lw=.5, zorder=0)
        for spine in ax.spines.values():
            spine.set_color("#C7CED4")
        if index == 0:
            ax.set_xlim(full_bounds[:2])
            ax.set_ylim(full_bounds[2:])
            ax.set_title("场景总览", fontsize=14, pad=13)
            ax.add_patch(Rectangle(lower, *(upper - lower), fill=False, linestyle="--",
                                   linewidth=1, edgecolor="#65717D", zorder=4))
            for i, path in enumerate(paths):
                draw_path(ax, path, i, False)
            ax.legend([Line2D([0], [0], color=colors[i % len(colors)], lw=2) for i in range(len(paths))],
                      [f"路径 {i+1}" for i in range(len(paths))], loc="lower left", framealpha=.95)
        else:
            path = paths[index - 1]
            reflections = sum(kind == "reflection" for kind, _ in path["interactions"])
            diffractions = sum(kind == "diffraction" for kind, _ in path["interactions"])
            title = f"{reflections} 次反射" + (f" + {diffractions} 次绕射" if diffractions else "")
            ax.set_title(f"路径 {index} · {title}\n长度 {path['length_m']:.2f} m", fontsize=12,
                         color=colors[(index - 1) % len(colors)], pad=13)
            ax.set_xlim(lower[0], upper[0])
            ax.set_ylim(lower[1], upper[1])
            draw_path(ax, path, index - 1, True)
        endpoints(ax)
    fig.text(.5, .075, "灰线：墙体   ·   圆点：反射   ·   菱形：绕射   ·   箭头：UE → BS   ·   虚框：右侧放大范围",
             ha="center", fontsize=11, color="#46515B")
    fig.text(.5, .034, f"俯视图，固定高度 {scene['fixed_height_m']:.2f} m。"
             "读取已保存的几何核验结果；UE 为真实位置，不代表定位输出。", ha="center", fontsize=10, color="#65717D")
    print(f"[绘图] {sample_id}：{len(paths)} 条路径。", flush=True)
    stem = f"sample_{args.sample:06d}_rt"
    artifacts = []
    for extension in ("png", "svg", "pdf"):
        target = output / f"{stem}.{extension}"
        fig.savefig(target, dpi=args.dpi, facecolor="white")
        artifacts.append(str(target))
    plt.close(fig)
    if any(digest(Path(path)) != value for path, value in inputs.items()):
        raise RuntimeError("绘图期间输入发生变化，请保留本轮记录并在新目录重试。")
    metadata = {"sample_id": sample_id, "inputs_sha256": inputs, "candidate_count": len(bank["hypotheses"]),
                "valid_paths": paths, "true_position_m": ue.tolist(), "bs_position_m": bs.tolist(),
                "scope": checked["scope"], "position_is_ground_truth_not_estimate": True,
                "source_sha256": {str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
                "artifacts": artifacts}
    (output / "plot_inputs.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print("[完成] " + artifacts[0], flush=True)


if __name__ == "__main__":
    main()
