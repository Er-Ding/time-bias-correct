"""三臂对照实验：只改一个开关，其余配置逐字节相同。

臂定义
  B0  基线：伪峰剔除与幅值加权都关闭（等于当前行为）
  M   B0 + music.spurious_peak_filter.enabled
  W   B0 + localization.amplitude_weighting.enabled
两次比较共用 B0，所以效果量为 M−B0 与 W−B0，各自只含一个自变量。

样本集分三组，第三组是安慰剂对照：不含边界伪峰的样本不应受 M 影响。
只读原实验的 CSI 与地图，输出写入独立目录，不修改原实验。
"""
from __future__ import annotations

# 必须在导入 numpy 之前限制线程数。BLAS 在导入时读取这些变量并缓存线程池，
# 之后再改环境变量无效；此前把设置写在工作函数里导致每个进程仍开满线程，
# 88 个进程在 96 核机器上产生上万线程、负载超订近 4 倍，反而慢约 8 倍。
import os

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import signal
import sys
import time

import numpy as np


ARMS: dict[str, dict] = {
    "B0": {"music": {"spurious_peak_filter": {"enabled": False}},
           "localization": {"amplitude_weighting": {"enabled": False}}},
    "M": {"music": {"spurious_peak_filter": {"enabled": True}},
          "localization": {"amplitude_weighting": {"enabled": False}}},
    "W": {"music": {"spurious_peak_filter": {"enabled": False}},
          "localization": {"amplitude_weighting": {"enabled": True}}},
}
BOUNDARY_DEG = 88.5


def process_thread_count() -> int:
    """本进程当前的操作系统线程数，用于核验线程限制确实生效。"""
    with open("/proc/self/status") as stream:
        for line in stream:
            if line.startswith("Threads:"):
                return int(line.split()[1])
    return -1


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arm_config(base: dict, arm: str) -> dict:
    """只改变两个已声明的开关，不依赖输入配置里的开关状态。"""
    result = deepcopy(base)
    for section, settings in ARMS[arm].items():
        for name, value in settings.items():
            result[section][name].update(value)
    return result


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout("trial_timeout")


def run_task(payload):
    """单次定位；线程数已在模块导入前限制，这里只降优先级并记录线程数。"""
    os.nice(10)
    src = str(payload.pop("_src"))
    if src not in sys.path:
        sys.path.insert(0, src)
    from time_bias_localization import pipeline
    from contextlib import redirect_stdout, redirect_stderr

    began = time.perf_counter()
    sample_id, arm = payload["sample_id"], payload["arm"]
    root = Path(payload["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    threads = process_thread_count()
    try:
        view = deepcopy(payload["config"])
        view["output"]["root"] = str(root)
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(payload["timeout_s"]))
        try:
            with (root / "online.log").open("a", buffering=1) as log, \
                    redirect_stdout(log), redirect_stderr(log):
                result = pipeline.localize(view, scene_json=payload["scene_json"],
                                           online_input=payload["online_npz"],
                                           generation_manifest=payload["generation_manifest"],
                                           output_root=root)
        finally:
            signal.alarm(0)
        elapsed = time.perf_counter() - began
        return {"sample_id": sample_id, "arm": arm, "status": result["status"],
                "reason": result.get("reason"), "mu_m": result.get("mu_m"),
                "clock_bias_s": result.get("clock_bias_s"),
                "output_type": result.get("output_type"),
                "processing_seconds": result.get("processing_seconds", elapsed),
                "wall_seconds": elapsed, "process_threads": threads,
                "diagnostics": {
                    "nominal_music_peak_count": (result.get("diagnostics") or {}
                                                 ).get("nominal_music_peak_count"),
                    "hypothesis_count": (result.get("diagnostics") or {}).get("hypothesis_count"),
                    "spurious_peak_filter": (result.get("diagnostics") or {}
                                            ).get("spurious_peak_filter"),
                    "angle_branch_search": {
                        key: value for key, value in
                        ((result.get("diagnostics") or {}).get("angle_branch_search") or {}).items()
                        if key in ("branch_count", "selected_branch", "all_branches_failed",
                                   "failure_reason")},
                    "observation_scales": (result.get("diagnostics") or {}
                                           ).get("observation_scales"),
                },
                "error": None}
    except _Timeout:
        return {"sample_id": sample_id, "arm": arm, "status": "timeout",
                "reason": "trial_timeout_s", "mu_m": None, "clock_bias_s": None,
                "output_type": None, "processing_seconds": time.perf_counter() - began,
                "wall_seconds": time.perf_counter() - began, "diagnostics": {},
                "process_threads": threads, "error": "timeout"}
    except Exception as error:  # 单样本失败不能拖垮整批
        return {"sample_id": sample_id, "arm": arm, "status": "error",
                "reason": f"{type(error).__name__}: {error}", "mu_m": None,
                "clock_bias_s": None, "output_type": None,
                "processing_seconds": time.perf_counter() - began,
                "wall_seconds": time.perf_counter() - began, "diagnostics": {},
                "process_threads": threads, "error": f"{type(error).__name__}: {error}"}


def classify_samples(source):
    """把样本分成：无位置、含边界伪峰的成功样本、不含边界伪峰的成功样本。"""
    groups = {"excluded": [], "success_with_boundary": [], "success_without_boundary": []}
    records = {}
    for path in sorted((source / "samples").glob("SAMPLE_*/result.json")):
        row = json.loads(path.read_text())
        sample_id = path.parent.name
        status = row["status"]
        if status not in ("excluded_observation", "success"):
            continue
        attempt = Path(row["attempt_dir"])
        if status == "success":
            peak_files = [attempt / "localization/music_peaks.json"]
        else:
            peak_files = list(attempt.glob("localization_unavailable/*/music_peaks.json"))
        if len(peak_files) != 1 or not peak_files[0].is_file():
            raise ValueError(f"当前结果没有唯一的 MUSIC 峰记录：{path}")
        peaks = json.loads(peak_files[0].read_text())
        boundary = sum(1 for peak in peaks.get("nominal") or []
                       if abs(np.rad2deg(peak["aoa_rad"])) >= BOUNDARY_DEG)
        records[sample_id] = {"status": status, "boundary_peaks": boundary,
                             "rt_path_count": row["rt_path_count"],
                             "true_position_m": row["true_position_m"],
                             "true_clock_bias_ns": row["true_clock_bias_ns"],
                             "attempt_dir": row["attempt_dir"]}
        if status == "excluded_observation":
            groups["excluded"].append(sample_id)
        elif boundary > 0:
            groups["success_with_boundary"].append(sample_id)
        else:
            groups["success_without_boundary"].append(sample_id)
    return groups, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--placebo-count", type=int, default=100)
    parser.add_argument("--arms", default="B0,M,W")
    parser.add_argument("--limit-per-group", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout_s < 1 or args.placebo_count < 0 or args.limit_per_group < 0:
        parser.error("进程数和时限必须大于零，抽样数量不能为负数")
    arms = [value.strip() for value in args.arms.split(",") if value.strip()]
    if not arms or len(set(arms)) != len(arms) or set(arms) - ARMS.keys():
        parser.error("实验组必须从 B0,M,W 中选择，不能为空或重复")
    from time_bias_localization.config import load_config, localization_config_view
    from time_bias_localization.boundary_experiment import source_fingerprint, write_json
    from time_bias_localization.monte_carlo_experiment import verify_record
    public_config = localization_config_view(load_config(args.config))
    public_config.pop("_config_path", None)
    configs = {arm: arm_config(public_config, arm) for arm in arms}
    source, output = args.input.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("输出必须与只读输入分开")
    groups, records = classify_samples(source)
    rng = np.random.default_rng(args.seed)
    placebo = sorted(groups["success_without_boundary"])
    indices = rng.choice(len(placebo), size=min(args.placebo_count, len(placebo)), replace=False)
    selection = {
        "excluded": sorted(groups["excluded"]),
        "success_with_boundary": sorted(groups["success_with_boundary"]),
        "success_without_boundary": sorted(placebo[index] for index in indices),
    }
    if args.limit_per_group:
        selection = {name: ids[:args.limit_per_group] for name, ids in selection.items()}
    tasks = []
    for group, ids in selection.items():
        for sample_id in ids:
            observation = json.loads((source / "samples" / sample_id / "observation.json").read_text())
            for key in ("scene_json", "online_npz", "generation_manifest"):
                verify_record(observation["artifacts"][key])
            online = observation["artifacts"]["online_npz"]["path"]
            manifest = observation["artifacts"]["generation_manifest"]["path"]
            for arm in arms:
                tasks.append({
                    "sample_id": sample_id, "group": group, "arm": arm,
                    "scene_json": observation["artifacts"]["scene_json"]["path"], "online_npz": online,
                    "generation_manifest": manifest,
                    "config": configs[arm],
                    "timeout_s": args.timeout_s,
                    "output_root": str(output / "runs" / arm / sample_id),
                    "_src": str(Path(__file__).resolve().parents[1] / "src"),
                })
    if not tasks:
        raise ValueError("没有选出可对照的样本")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(source), "config": str(args.config.resolve()),
        "config_sha256": digest(args.config), "arms": arms,
        "public_configs": configs, "source": source_fingerprint(),
        "runner_sha256": digest(__file__),
        "arm_definitions": ARMS, "selection": selection,
        "selection_counts": {name: len(ids) for name, ids in selection.items()},
        "task_count": len(tasks), "workers": args.workers,
        "timeout_s": args.timeout_s, "classify_boundary_deg": BOUNDARY_DEG,
        "records": records, "single_variable_per_comparison": True,
        "note": "M−B0 与 W−B0 各自只含一个自变量；B0 是两次比较的公共基线",
    })

    print(f"[资源] 本进程线程数 {process_thread_count()}，"
          f"线程上限变量 {os.environ['OMP_NUM_THREADS']}/{os.environ['OPENBLAS_NUM_THREADS']}/"
          f"{os.environ['MKL_NUM_THREADS']}，CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}", flush=True)
    print(f"[计划] 样本 {sum(len(v) for v in selection.values())} 个，"
          f"分组 {({k: len(v) for k, v in selection.items()})}，任务 {len(tasks)} 个，"
          f"并发 {args.workers}", flush=True)
    results, began = [], time.perf_counter()
    task_groups = {(task["sample_id"], task["arm"]): task["group"] for task in tasks}
    context = mp.get_context("spawn")
    with context.Pool(processes=args.workers) as pool:
        for index, payload in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1), 1):
            payload.pop("_src", None)
            payload["group"] = task_groups[payload["sample_id"], payload["arm"]]
            results.append(payload)
            write_json(output / "results.json", {"results": results, "completed": index, "total": len(tasks)})
            if index % 20 == 0 or index == len(tasks):
                elapsed = time.perf_counter() - began
                rate = index / max(elapsed, 1e-9)
                remaining = (len(tasks) - index) / max(rate, 1e-9)
                print(f"[进度] {index}/{len(tasks)} 用时 {elapsed/60:.1f} min，"
                      f"预计剩余 {remaining/60:.1f} min；"
                      f"状态 {dict(Counter(r['status'] for r in results))}；"
                      f"工作进程线程数 {dict(Counter(r['process_threads'] for r in results))}", flush=True)
    print(f"[完成] {len(results)} 个任务，用时 {(time.perf_counter()-began)/60:.1f} min", flush=True)
    if any(row["status"] in ("error", "localization_failed", "experiment_error") for row in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
