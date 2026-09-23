"""单场景随机位置实验：先固定有信号的样本，再由独立进程定位，最后评估。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
import fcntl
import json
import os
from pathlib import Path
import threading
import time
import traceback

import numpy as np
import yaml

from .boundary_experiment import PersistentWorker, freeze_experiment, read_json, unique_stamp, write_json
from .config import load_config, localization_config_view, validate_config
from .path_policy import PATH_TYPE_NAMES, path_type_counts, selected_path_type_counts
from .provenance import artifact_record, file_sha256


def seed_for(seed: int, index: int, purpose: int) -> int:
    return int(np.random.SeedSequence([seed, index, purpose]).generate_state(1)[0])


def load_settings(path: Path, *, sample_count=None, workers=None, backend=None, compute_backend=None) -> tuple[dict, dict]:
    raw = yaml.safe_load(path.read_text())
    settings = dict(raw.get("experiment", {}))
    defaults = dict(sample_count=1000, max_proposals=100000, random_seed=20260915,
                    bias_min_ns=-50.0, bias_max_ns=50.0, workers=4,
                    trial_timeout_s=1800.0, channel_backend="sionna", plots=True,
                    legal_region={"wall_clearance_m": .5, "bs_min_distance_m": 2.0})
    if set(settings) - set(defaults):
        raise ValueError(f"未知实验参数：{sorted(set(settings) - set(defaults))}")
    settings = {**defaults, **settings}
    for key, value in (("sample_count", sample_count), ("workers", workers), ("channel_backend", backend)):
        if value is not None:
            settings[key] = value
    for key in ("sample_count", "max_proposals", "workers"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} 必须是正整数")
    if type(settings["random_seed"]) is not int or settings["random_seed"] < 0:
        raise ValueError("random_seed 必须是非负整数")
    for key in ("bias_min_ns", "bias_max_ns", "trial_timeout_s"):
        if isinstance(settings[key], bool) or not np.isfinite(float(settings[key])):
            raise ValueError(f"{key} 必须是有限数")
        settings[key] = float(settings[key])
    if settings["bias_min_ns"] > settings["bias_max_ns"] or settings["trial_timeout_s"] <= 0:
        raise ValueError("时间偏置下界不能超过上界，单样本超时必须大于零")
    if settings["max_proposals"] < settings["sample_count"]:
        raise ValueError("max_proposals 不能小于 sample_count")
    if settings["channel_backend"] not in ("sionna", "synthetic_fixture"):
        raise ValueError("channel_backend 只能为 sionna 或 synthetic_fixture")
    if type(settings["plots"]) is not bool or not isinstance(settings["legal_region"], dict):
        raise ValueError("plots 必须是布尔值，legal_region 必须是映射")
    config = load_config(path)
    config.pop("experiment", None)
    config.pop("_config_path", None)  # 完整生成配置的路径不进入定位子进程。
    if compute_backend is not None:
        config["compute"]["backend"] = compute_backend
    config["project"]["random_seed"] = settings["random_seed"]
    if config["localization"]["solver_method"] != "continuous":
        raise ValueError("本实验只支持连续定位流程")
    if config["scene"].get("diffraction_position") != "last_from_bs":
        raise ValueError("本实验要求 last_from_bs 六类路径规则")
    if config["scene"]["max_reflections"] != 2 or config["scene"].get("max_diffractions") != 1:
        raise ValueError("本实验要求最多两次反射、一次绕射")
    if settings["channel_backend"] == "sionna":
        if config["scene"]["source"] != "sionna_builtin":
            raise ValueError("真实 RT 实验需要 Sionna 内置场景")
        if config["scene"].get("localization_bounds_m") != config["scene"]["bounds_m"]:
            raise ValueError("采样范围和定位地图范围必须显式一致")
    low, high = settings["bias_min_ns"] * 1e-9, settings["bias_max_ns"] * 1e-9
    if low < float(config["localization"]["bias_min_s"]) or high > float(config["localization"]["bias_max_s"]):
        raise ValueError("注入时间偏置范围超出了公开求解范围")
    if float(config["music"]["delay_min_s"]) > low:
        raise ValueError("MUSIC 时延下界必须覆盖最小公共时间偏置，不能漏掉负观测时延")
    if config["music"].get("path_detection", {}).get("enabled", False):
        raise ValueError("本实验使用原始 CSI 的 MUSIC 细化峰，不启用逐路径残差检测")
    from .data import build_subcarrier_frequencies
    from .pipeline import _validate_unambiguous_delay_window
    _validate_unambiguous_delay_window(config["music"], build_subcarrier_frequencies(
        bandwidth_hz=float(config["radio"]["bandwidth_hz"]), num_subcarriers=config["radio"]["num_subcarriers"]))
    validate_config(config)
    return settings, config


def verify_record(record: dict) -> None:
    if file_sha256(record["path"]) != record["sha256"]:
        raise ValueError(f"保存文件已发生变化：{record['path']}")


def prepare_samples(root: Path, settings: dict, config: dict) -> dict:
    from .boundary_channel import load_probe, make_boundary_channel, save_probe
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        if len(plan["samples"]) != settings["sample_count"]:
            raise ValueError("已固定的样本数量与配置不一致")
        for sample in plan["samples"]:
            verify_record(sample["proposal_record"])
        return plan
    proposals = root / "sampling" / "proposals"
    accepted, counts, provider = [], Counter(), None
    bounds = np.asarray(config["scene"]["bounds_m"], float)
    print(f"[采样] 目标 {settings['sample_count']} 个有路径的位置；最多检查 {settings['max_proposals']} 个提案。", flush=True)
    try:
        for index in range(settings["max_proposals"]):
            if len(accepted) == settings["sample_count"]:
                break
            record_path = proposals / f"proposal_{index:07d}.json"
            position_seed = seed_for(settings["random_seed"], index, 0)
            rt_seed = seed_for(settings["random_seed"], index, 1)
            point = np.random.default_rng(position_seed).uniform(bounds[[0, 2]], bounds[[1, 3]])
            if record_path.exists():
                record = read_json(record_path)
                if record["position_m"] != point.tolist() or record["seed"] != rt_seed:
                    raise ValueError("已有提案与确定的采样种子不一致")
            else:
                cache_root = root / "sampling" / "channels" / f"proposal_{index:07d}"
                complete = sorted(cache_root.glob("*/probe.json"))
                if complete:
                    # 恢复已完成 RT、但尚未来得及登记的提案，避免再抽一个点。
                    attempt = complete[0].parent
                    probe = load_probe(attempt)
                    if (attempt / "context.json").exists():
                        setup_root = Path(read_json(attempt / "context.json")["setup_root"])
                    else:
                        # 也覆盖 probe.json 已提交、context.json 尚未提交时的中断。
                        matches = [p.parent for p in (root / "channel_setups").glob("*/channel_setup.json")
                                   if file_sha256(p) == probe.channel_setup_sha256]
                        if len(matches) != 1:
                            raise ValueError("缓存信道无法匹配其原始公开设置")
                        setup_root = matches[0]
                        write_json(attempt / "context.json", {"setup_root": str(setup_root)})
                    if probe.seed != rt_seed or not np.array_equal(probe.position_m, point):
                        raise ValueError("缓存信道与提案不一致")
                else:
                    if provider is None:
                        print("[射线追踪] 正在加载场景和公开地图。", flush=True)
                        provider = make_boundary_channel(config, root / "channel_setups" / unique_stamp(),
                            backend=settings["channel_backend"], legal_region=settings["legal_region"])
                    setup_root = provider.setup_root
                    probe = provider.probe(point, rt_seed)
                    attempt = cache_root / unique_stamp()
                    if probe.status in ("covered", "no_signal"):
                        save_probe(probe, attempt)
                        write_json(attempt / "context.json", {"setup_root": str(setup_root)})
                record = {"proposal_index": index, "position_seed": position_seed,
                          **probe.summary(), "setup_root": str(setup_root)}
                if probe.status == "unknown":
                    error_path = root / "sampling" / "errors" / f"proposal_{index:07d}_{unique_stamp()}.json"
                    write_json(error_path, record)
                    raise RuntimeError(f"RT 技术错误，保留同一提案等待重试，不能当作无路径补采：{error_path}")
                if probe.status in ("covered", "no_signal"):
                    record.update(probe_root=str(attempt), probe_record=artifact_record(attempt / "probe.json"))
                if probe.status == "covered":
                    record["path_type_counts"] = path_type_counts(probe.path_metadata["interactions"], probe.path_metadata["retained_mask"])
                if probe.status not in ("covered", "no_signal", "illegal"):
                    raise RuntimeError(f"无法识别的提案状态：{probe.status}")
                write_json(record_path, record)
            counts[record["status"]] += 1
            if record["status"] == "covered":
                verify_record(record["probe_record"])
                sample_index = len(accepted)
                bias_seed = seed_for(settings["random_seed"], index, 2)
                bias_ns = float(np.random.default_rng(bias_seed).uniform(settings["bias_min_ns"], settings["bias_max_ns"]))
                accepted.append({**record, "sample_id": f"SAMPLE_{sample_index + 1:06d}",
                    "proposal_record": artifact_record(record_path), "bias_seed": bias_seed,
                    "noise_seed": seed_for(settings["random_seed"], index, 3), "clock_bias_ns": bias_ns})
            progress = {"stage": "sampling", "target": settings["sample_count"], "proposal_count": index + 1,
                        "accepted_count": len(accepted), "proposal_status_counts": dict(counts),
                        "samples_with_at_least_two_paths": sum(s["retained_path_count"] >= 2 for s in accepted)}
            write_json(root / "progress.json", progress)
            print(f"[采样] 提案 {index + 1}，已接受 {len(accepted)}/{settings['sample_count']}，状态={record['status']}，路径={record['retained_path_count']}", flush=True)
    finally:
        if provider is not None:
            provider.close()
    if len(accepted) != settings["sample_count"]:
        raise RuntimeError(f"达到提案上限，仅接受 {len(accepted)}/{settings['sample_count']}；实验尚未完成。")
    plan = {"schema_version": 1, "path_type_direction": "BS_to_UE", "path_type_names": list(PATH_TYPE_NAMES),
            "sampling": "uniform_xy_in_declared_outdoor_region_conditioned_on_at_least_one_supported_path",
            "proposal_count": sum(counts.values()), "proposal_status_counts": dict(counts),
            "replacement_rule": "illegal_or_zero_supported_paths_only_never_localization_failure",
            "samples": accepted}
    write_json(plan_path, plan)
    return plan


def prepare_observations(root: Path, plan: dict, settings: dict, config: dict) -> dict:
    from .boundary_channel import load_probe, write_observation_bundle
    observations = {}
    for index, sample in enumerate(plan["samples"]):
        directory = root / "samples" / sample["sample_id"]
        pointer = directory / "observation.json"
        if pointer.exists():
            record = read_json(pointer)
            for artifact in record["artifacts"].values():
                verify_record(artifact)
        else:
            verify_record(sample["probe_record"])
            probe = load_probe(sample["probe_root"])
            cfg = deepcopy(config)
            cfg["simulation"].update(ue_position_m=sample["position_m"], clock_bias_s=sample["clock_bias_ns"] * 1e-9)
            attempt = directory / "observation_attempts" / unique_stamp()
            generated = write_observation_bundle(probe, cfg, setup_root=sample["setup_root"],
                                                 output_root=attempt, noise_seed=sample["noise_seed"])
            record = {"sample_id": sample["sample_id"], "path_type_counts": sample["path_type_counts"],
                "artifacts": {key: artifact_record(generated[key]) for key in
                              ("online_npz", "truth_npz", "scene_json", "generation_manifest")}}
            write_json(pointer, record)
        observations[sample["sample_id"]] = record
        write_json(root / "progress.json", {"stage": "csi_generation", "completed": index + 1, "target": len(plan["samples"])})
        print(f"[CSI] {index + 1}/{len(plan['samples'])} {sample['sample_id']} 已保存，信噪比={config['radio']['snr_db']} dB。", flush=True)
    return observations


def online_worker(connection) -> None:
    """spawn 新进程；工作请求只含公开配置、地图、CSI 和其来源清单。"""
    from .pipeline import localize
    from .timing import collect_timings
    connection.send({"ready": True, "pid": os.getpid()})
    while True:
        try:
            job = connection.recv()
        except EOFError:
            return
        if job is None:
            return
        root = Path(job["output_root"])
        began = time.perf_counter()
        with (root / "online.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            try:
                with collect_timings(snapshot_path=job["snapshot_path"]):
                    result = localize(job["config"], scene_json=job["scene_json"], online_input=job["online_input"],
                        generation_manifest=job["generation_manifest"], output_root=root)
                diag = result.get("diagnostics", {})
                payload = {"status": result["status"], "reason": result.get("reason"),
                    "mu_m": result.get("mu_m"), "clock_bias_s": result.get("clock_bias_s"),
                    "selected_path_type_counts": selected_path_type_counts(result.get("selected_paths", [])),
                    "music_observation_count": diag.get("nominal_music_peak_count", diag.get("observation_count")),
                    "hypothesis_search_incomplete": diag.get("hypothesis_search_incomplete"),
                    "scientific_validation_status": result.get("scientific_validation_status", "not_validated")}
            except Exception as error:
                traceback.print_exc()
                payload = {"status": "localization_failed", "reason": f"{type(error).__name__}: {error}",
                           "mu_m": None, "clock_bias_s": None, "selected_path_type_counts": [0] * 6}
        payload["processing_seconds"] = time.perf_counter() - began
        # 先提交可恢复结果，再通知父进程，避免结束瞬间断线造成已完成样本重跑。
        write_json(root / "worker_result.json", payload)
        connection.send(payload)


def evaluate_sample(sample: dict, observation: dict, payload: dict, attempt: Path) -> dict:
    """在线结果已经提交后才读取真值。失败和单路径样本也保留一行。"""
    row = {"sample_id": sample["sample_id"], "proposal_index": sample["proposal_index"],
           "rt_path_count": sample["retained_path_count"], "path_type_counts": sample["path_type_counts"],
           "selected_path_type_counts": payload.get("selected_path_type_counts", [0] * 6),
           "status": payload["status"], "reason": payload.get("reason", payload.get("error")),
           "true_position_m": sample["position_m"], "true_clock_bias_ns": sample["clock_bias_ns"],
           "estimated_position_m": payload.get("mu_m"), "estimated_clock_bias_ns": None,
           "position_error_m": None, "clock_bias_error_ns": None, "clock_bias_signed_error_ns": None,
           "music_observation_count": payload.get("music_observation_count"),
           "processing_seconds": payload.get("processing_seconds"), "attempt_dir": str(attempt),
           "hypothesis_search_incomplete": payload.get("hypothesis_search_incomplete"),
           "truth_used_in_online_solve": False, "scientific_validation_status": "not_validated"}
    if payload["status"] == "success":
        truth_record = observation["artifacts"]["truth_npz"]
        verify_record(truth_record)
        with np.load(truth_record["path"], allow_pickle=False) as truth:
            true_position, true_bias = truth["ue_position_m"], float(truth["clock_bias_s"])
        position = np.asarray(payload["mu_m"], float)
        bias = float(payload["clock_bias_s"])
        if position.shape != (2,) or not np.all(np.isfinite(position)) or not np.isfinite(bias):
            raise ValueError("success 结果必须包含有限位置和公共时间偏置")
        row.update(estimated_clock_bias_ns=bias * 1e9,
                   position_error_m=float(np.linalg.norm(position - true_position)),
                   clock_bias_error_ns=abs(bias - true_bias) * 1e9,
                   clock_bias_signed_error_ns=(bias - true_bias) * 1e9)
    write_json(attempt / "evaluation.json", row)
    return row


def numeric_summary(values: list[float]) -> dict:
    a = np.asarray(values, float)
    if not len(a):
        return {"count": 0, "mean": None, "median": None, "rmse": None, "p90": None, "p95": None, "max": None}
    return {"count": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "rmse": float(np.sqrt(np.mean(a*a))), "p90": float(np.quantile(a, .9)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def summarize(plan: dict, results: dict[str, dict]) -> dict:
    samples = plan["samples"]
    status = Counter(results.get(s["sample_id"], {}).get("status", "pending") for s in samples)
    accepted_results = [results[s["sample_id"]] for s in samples
                        if results.get(s["sample_id"], {}).get("status") == "success"]
    groups = {}
    for name, predicate in (("all_samples", lambda s: True),
                            ("at_least_two_rt_paths", lambda s: s["retained_path_count"] >= 2),
                            ("single_rt_path", lambda s: s["retained_path_count"] == 1)):
        selected = [s for s in samples if predicate(s)]
        good = [results[s["sample_id"]] for s in selected if results.get(s["sample_id"], {}).get("status") == "success"]
        groups[name] = {"sample_count": len(selected), "success_count": len(good),
            "success_rate": len(good)/len(selected) if selected else None,
            "position_error_m": numeric_summary([r["position_error_m"] for r in good]),
            "clock_bias_error_ns": numeric_summary([r["clock_bias_error_ns"] for r in good])}
    counts = np.asarray([s["path_type_counts"] for s in samples], dtype=int).reshape(-1, 6)
    selected_counts = np.asarray([r["selected_path_type_counts"] for r in accepted_results], dtype=int).reshape(-1, 6)
    return {"schema_version": 1, "accepted_sample_count": len(samples), "completed_sample_count": len(results),
        "all_samples_finished": len(results) == len(samples), "status_counts": dict(status),
        "proposal_count": plan["proposal_count"], "proposal_status_counts": plan["proposal_status_counts"],
        "samples_with_at_least_two_paths": sum(s["retained_path_count"] >= 2 for s in samples),
        "samples_with_more_than_two_paths": sum(s["retained_path_count"] > 2 for s in samples),
        "path_type_direction": "BS_to_UE", "path_type_names": list(PATH_TYPE_NAMES),
        "rt_path_type_counts": counts.sum(axis=0).tolist(), "samples_containing_path_type": (counts > 0).sum(axis=0).tolist(),
        "selected_path_type_counts_on_success": selected_counts.sum(axis=0).tolist(),
        "accuracy": groups, "accuracy_scope": "conditional_on_success; success_rate_denominator_includes_failures_and_pending",
        "scientific_validation_status": "monte_carlo_statistics_not_full_path_set_acceptance"}


def publish_report(root: Path, plan: dict, results: dict[str, dict], *, plots=False) -> dict:
    report = root / "report"
    summary = summarize(plan, results)
    write_json(report / "summary.json", summary)
    fields = ["sample_id", "proposal_index", "rt_path_count", "status", "reason", "true_position_m", "estimated_position_m",
              "true_clock_bias_ns", "estimated_clock_bias_ns", "position_error_m", "clock_bias_error_ns",
              "clock_bias_signed_error_ns", "path_type_counts", "selected_path_type_counts", "music_observation_count",
              "processing_seconds", "attempt_dir", "hypothesis_search_incomplete"]
    temporary = report / "samples.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for sample in plan["samples"]:
            row = results.get(sample["sample_id"], {"sample_id": sample["sample_id"], "proposal_index": sample["proposal_index"],
                "rt_path_count": sample["retained_path_count"], "status": "pending", "true_position_m": sample["position_m"],
                "true_clock_bias_ns": sample["clock_bias_ns"], "path_type_counts": sample["path_type_counts"]})
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    temporary.replace(report / "samples.csv")
    accuracy = summary["accuracy"]["all_samples"]
    lines = ["# 单场景蒙特卡罗实验", "", f"- 已接受位置：{len(plan['samples'])}",
        f"- 已完成定位尝试：{len(results)}", f"- 状态计数：{summary['status_counts']}",
        f"- 至少两条 RT 路径的位置：{summary['samples_with_at_least_two_paths']}",
        f"- 位置误差（米，仅成功样本）：{accuracy['position_error_m']}",
        f"- 时间偏置绝对误差（ns，仅成功样本）：{accuracy['clock_bias_error_ns']}",
        "", "## 六维路径数量", "", f"顺序按 BS→UE：{list(PATH_TYPE_NAMES)}", "",
        f"- 所有接受样本进入 CSI 的路径总数：{summary['rt_path_type_counts']}",
        f"- 成功定位时选中的路径总数：{summary['selected_path_type_counts_on_success']}",
        "", "## 统计口径", "", "仅非法放置位置和没有有效路径的位置补采；单路径、定位失败、歧义和超时均保留。",
        "误差统计只含成功输出，成功率分母包含所有接受样本；至少两条路径的子组另列在 summary.json。",
        "同一样本所有路径共用一个时间偏置；位置、RT、偏置和 CSI 噪声使用分别派生的随机种子。",
        "绕射限制在 CSI 合成前实施；上行 UE→BS 顺序中的绕射必须是第一次交互。",
        "场景采用固定高度的二维定位、阵列正面角度范围和露天可放置区域；噪声按每个样本的目标信噪比设置。",
        "定位使用有限搜索预算；success 表示选中路径满足当前接受条件，不表示已验证全局唯一性或完整路径集合。",
        "", f"逐样本表：{report / 'samples.csv'}", f"冻结采样计划：{root / 'plan.json'}", ""]
    (report / "summary.md").write_text("\n".join(lines))
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, field, label in zip(axes, ("position_error_m", "clock_bias_error_ns"),
                                    ("Position error (m)", "Clock bias absolute error (ns)")):
            values = np.sort([r[field] for r in results.values() if r["status"] == "success"])
            if len(values):
                ax.step(values, np.arange(1, len(values)+1)/len(values), where="post")
            else:
                ax.text(.5, .5, "No successful estimates", ha="center", transform=ax.transAxes)
            ax.set(xlabel=label, ylabel="CDF among successful estimates", ylim=(0, 1.02))
            ax.grid(alpha=.3)
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(report / f"error_cdf.{extension}", dpi=160)
        plt.close(fig)
    return summary


def solve_samples(root: Path, plan: dict, observations: dict, settings: dict, config: dict) -> dict:
    results = {}
    for sample in plan["samples"]:
        pointer = root / "samples" / sample["sample_id"] / "result.json"
        if pointer.exists():
            row = read_json(pointer)
            verify_record(row["evaluation_record"])
            verify_record(row["solve_record"])
            original = read_json(Path(row["evaluation_record"]["path"]))
            if original != {key: value for key, value in row.items() if key not in ("evaluation_record", "solve_record")}:
                raise ValueError(f"已有汇总行与评估记录不一致：{pointer}")
            results[sample["sample_id"]] = row
    publish_report(root, plan, results)
    local, workers, workers_lock = threading.local(), [], threading.Lock()

    def execute_impl(sample):
        directory = root / "samples" / sample["sample_id"]
        observation = observations[sample["sample_id"]]
        completed = sorted((directory / "localization_attempts").glob("*/solve_record.json"))
        if completed:
            attempt, payload = completed[0].parent, read_json(completed[0])
        else:
            recovered = sorted((directory / "localization_attempts").glob("*/worker_result.json"))
            if recovered:
                attempt, payload = recovered[0].parent, read_json(recovered[0])
                if payload.get("processing_seconds", float("inf")) > settings["trial_timeout_s"]:
                    payload = {"status": "timeout", "reason": "recovered_worker_exceeded_declared_budget",
                               "processing_seconds": payload["processing_seconds"]}
                write_json(attempt / "solve_record.json", payload)
                return commit_evaluation(sample, observation, payload, attempt)
            attempt = directory / "localization_attempts" / unique_stamp()
            attempt.mkdir(parents=True, exist_ok=False)
            public = localization_config_view(config)
            public["project"]["random_seed"] = seed_for(settings["random_seed"], sample["proposal_index"], 4)
            public["output"]["root"] = str(attempt)
            job = {"config": public, "scene_json": observation["artifacts"]["scene_json"]["path"],
                "online_input": observation["artifacts"]["online_npz"]["path"],
                "generation_manifest": observation["artifacts"]["generation_manifest"]["path"],
                "output_root": str(attempt), "snapshot_path": str(attempt / "timing_live.json")}
            write_json(attempt / "online_request.json", job)
            print(f"[定位开始] {sample['sample_id']}，详细日志={attempt / 'online.log'}", flush=True)
            worker = getattr(local, "worker", None)
            if worker is None or worker._closed:
                worker = PersistentWorker(target=online_worker)
                local.worker = worker
                with workers_lock:
                    workers.append(worker)
            payload = worker.run(job, settings["trial_timeout_s"])
            # 超时判定由父进程固定；不能被截止时刻附近子进程刚写出的结果覆盖。
            write_json(attempt / "solve_record.json", payload)
        return commit_evaluation(sample, observation, payload, attempt)

    def commit_evaluation(sample, observation, payload, attempt):
        row = evaluate_sample(sample, observation, payload, attempt)
        row.update(evaluation_record=artifact_record(attempt / "evaluation.json"),
                   solve_record=artifact_record(attempt / "solve_record.json"))
        write_json(root / "samples" / sample["sample_id"] / "result.json", row)
        return row

    def execute(sample):
        try:
            return execute_impl(sample)
        except Exception as error:
            # 保留逐样本技术异常并继续其余固定样本；结束后整体退出码为 2。
            attempt = root / "samples" / sample["sample_id"] / "system_errors" / unique_stamp()
            attempt.mkdir(parents=True, exist_ok=False)
            (attempt / "error.txt").write_text(traceback.format_exc())
            payload = {"status": "experiment_error", "reason": f"{type(error).__name__}: {error}"}
            write_json(attempt / "solve_record.json", payload)
            print(f"[样本技术错误] {sample['sample_id']}：{payload['reason']}；现场={attempt}", flush=True)
            return commit_evaluation(sample, observations[sample["sample_id"]], payload, attempt)

    try:
        with ThreadPoolExecutor(max_workers=settings["workers"]) as pool:
            pending = {pool.submit(execute, sample): sample["sample_id"] for sample in plan["samples"] if sample["sample_id"] not in results}
            while pending:
                done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                if not done:
                    print(f"[定位进行中] 已完成 {len(results)}/{len(plan['samples'])}；当前阶段见各样本 online.log。", flush=True)
                for future in done:
                    sample_id = pending.pop(future)
                    row = future.result()
                    results[sample_id] = row
                    summary = publish_report(root, plan, results)
                    write_json(root / "progress.json", {"stage": "localization", "completed": len(results),
                        "target": len(plan["samples"]), "status_counts": summary["status_counts"]})
                    print(f"[定位完成] {len(results)}/{len(plan['samples'])} {sample_id}，状态={row['status']}，位置误差={row['position_error_m']} m，偏置误差={row['clock_bias_error_ns']} ns", flush=True)
    finally:
        for worker in workers:
            worker.close()
    return publish_report(root, plan, results, plots=settings["plots"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--rt-backend", choices=("sionna", "synthetic_fixture"))
    parser.add_argument("--compute-backend", choices=("numpy", "cuda"), help="MUSIC 计算设备；不改变射线追踪设备或连续求解模型")
    parser.add_argument("--prepare-only", action="store_true", help="只完成采样和 CSI；相同参数再次运行可继续定位")
    args = parser.parse_args()
    settings, config = load_settings(args.config, sample_count=args.samples, workers=args.workers,
                                    backend=args.rt_backend, compute_backend=args.compute_backend)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".experiment.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("同一输出目录已有实验在运行。") from None
        freeze_experiment(root, settings, config)
        print(f"[实验开始] 场景={config['scene']['name']}，样本={settings['sample_count']}，输出={root}", flush=True)
        plan = prepare_samples(root, settings, config)
        observations = prepare_observations(root, plan, settings, config)
        if args.prepare_only:
            existing = {sample["sample_id"]: read_json(root / "samples" / sample["sample_id"] / "result.json")
                        for sample in plan["samples"] if (root / "samples" / sample["sample_id"] / "result.json").exists()}
            publish_report(root, plan, existing)
            print("本次只准备采样和 CSI；已有定位结果继续保留。", flush=True)
            return
        summary = solve_samples(root, plan, observations, settings, config)
        write_json(root / "progress.json", {"stage": "completed", "completed": summary["completed_sample_count"],
            "target": settings["sample_count"], "status_counts": summary["status_counts"]})
        print(f"[实验完成] 状态={summary['status_counts']}；汇总={root / 'report/summary.json'}", flush=True)
        if any(summary["status_counts"].get(key, 0) for key in ("localization_failed", "experiment_error")):
            raise SystemExit(2)  # 技术异常区别于约束不足、歧义或声明预算内未解出。


if __name__ == "__main__":
    main()
