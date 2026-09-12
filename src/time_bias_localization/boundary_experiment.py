"""覆盖区随机取点、成对定位、分阶段计时。在线子进程仅接收公开配置和 CSI 路径。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import yaml

from .config import load_config, localization_config_view
from .provenance import file_sha256, canonical_json_sha256

STRATEGIES = ("single", "coverage")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def append_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def unique_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + f"_{os.getpid()}"


def derived_seed(seed: int, cohort: str, index: int, purpose: int) -> int:
    return int(np.random.SeedSequence([seed, 0 if cohort == "pilot" else 1, index, purpose]).generate_state(1)[0])


def load_settings(path: Path) -> tuple[dict, dict]:
    settings = yaml.safe_load(path.read_text())
    allowed = {"generation_config", "channel_backend", "pilot_ue_count", "formal_ue_count", "noise_repeats",
               "random_seed", "max_proposals", "trial_timeout_s", "warmup_per_strategy", "timing_snapshots",
               "plots", "legal_region", "public_scene_json"}
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"实验配置包含未知字段：{sorted(unknown)}")
    defaults = dict(channel_backend="sionna", pilot_ue_count=30, formal_ue_count=300, noise_repeats=5,
                    random_seed=20260911, max_proposals=100000, trial_timeout_s=600.0,
                    warmup_per_strategy=5, timing_snapshots=True, plots=True, legal_region={})
    settings = {**defaults, **settings}
    for key in ("pilot_ue_count", "formal_ue_count", "noise_repeats", "max_proposals"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    for key in ("random_seed", "warmup_per_strategy"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"{key} 必须为非负整数")
    if not np.isfinite(settings["trial_timeout_s"]) or settings["trial_timeout_s"] <= 0:
        raise ValueError("trial_timeout_s 必须为正数")
    for key in ("timing_snapshots", "plots"):
        if type(settings[key]) is not bool:
            raise ValueError(f"{key} 必须是布尔值")
    if settings["channel_backend"] not in {"sionna", "synthetic_fixture"}:
        raise ValueError("channel_backend 无效")
    for key in ("generation_config", "public_scene_json"):
        if settings.get(key):
            source = Path(settings[key]).expanduser()
            settings[key] = str((path.parent / source).resolve() if not source.is_absolute() else source.resolve())
    config = load_config(settings["generation_config"])
    if config["scene"].get("max_diffractions") != 1:
        raise ValueError("本对照实验要求 max_diffractions=1，两组均启用绕射")
    if config['scene'].get('source') == 'sionna_builtin':
        if config['scene'].get('localization_bounds_m') != config['scene']['bounds_m']:
            raise ValueError('Sionna bounds_m 与 localization_bounds_m 必须显式一致，避免从默认房间范围采样')
    return settings, config


def source_fingerprint() -> dict:
    files = {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for package in ('numpy', 'scipy', 'sionna-rt', 'sionna', 'mitsuba', 'drjit', 'cupy-cuda12x', 'matplotlib', 'PyYAML'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"versions": versions, "sha256": canonical_json_sha256(files), "files": files,
            "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "cpu_threads": os.environ.get("OMP_NUM_THREADS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER")}


def freeze_experiment(root: Path, settings: dict, config: dict) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    frozen_config = {key: value for key, value in config.items() if not key.startswith("_")}
    record = {"schema_version": 1, "settings": settings, "generation_config": frozen_config,
              "source": source_fingerprint(), "coverage_mode": "nonzero_supported_rt_signal_fixed_snr",
              "scientific_validation_status": "planned_not_a_result"}
    # Public input files referenced by the configuration are content bound too.
    if settings.get("public_scene_json"):
        record["public_scene_sha256"] = file_sha256(settings["public_scene_json"])
    path = root / "experiment.json"
    if path.exists():
        if read_json(path) != record:
            raise ValueError("代码、环境或实验参数与冻结记录不同；请使用新输出目录，旧结果保留")
    else:
        write_json(path, record)
    return record


def prepare_cohort(root: Path, cohort: str, settings: dict, config: dict) -> dict:
    from .boundary_channel import make_boundary_channel, save_probe, load_probe, write_observation_bundle
    directory = root / cohort
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / "plan.json"
    if plan_path.exists():
        return read_json(plan_path)
    # Freeze qualified positions before any localization; a failed solver never triggers replacement.
    ledger_path = directory / "coverage_proposals.jsonl"
    ledger = rows(ledger_path)
    completed = {row["proposal_index"]: row for row in ledger if row["status"] in {"covered", "no_signal", "illegal"}}
    accepted = [row for _, row in sorted(completed.items()) if row["status"] == "covered"]
    required = settings[f"{cohort}_ue_count"]
    provider = None
    if len(accepted) < required:
        provider = make_boundary_channel(config, directory / "channel_setups" / unique_stamp(),
                                         backend=settings["channel_backend"],
                                         public_scene_json=settings.get("public_scene_json"),
                                         legal_region=settings["legal_region"])
    try:
        bounds = np.asarray(config["scene"]["bounds_m"], dtype=float)
        for index in range(settings["max_proposals"]):
            if len(accepted) >= required:
                break
            if index in completed:
                continue
            seed = derived_seed(settings["random_seed"], cohort, index, 0)
            rng = np.random.default_rng(seed)
            point = rng.uniform(bounds[[0, 2]], bounds[[1, 3]])
            base_probe_root = directory / "frozen_channels" / f"proposal_{index:07d}"
            saved = sorted(base_probe_root.parent.glob(base_probe_root.name + "*")) if base_probe_root.parent.exists() else []
            complete_saved = [path for path in saved if (path / "probe.json").is_file()]
            if complete_saved:
                # Recover a probe committed before its ledger row, retaining its original public setup.
                probe_root = complete_saved[0]
                probe = load_probe(probe_root)
                if probe.seed != seed or not np.array_equal(probe.position_m, point):
                    raise ValueError("已有信道缓存与本次提案不一致")
                matches = [path.parent for path in (directory / "channel_setups").glob("*/channel_setup.json")
                           if file_sha256(path) == probe.channel_setup_sha256]
                if len(matches) != 1:
                    raise ValueError("已冻结信道的原始公开设置无法唯一匹配，不能换设置继续")
                setup_root = matches[0]
            else:
                probe = provider.probe(point, seed)
                setup_root = provider.setup_root
                probe_root = base_probe_root if not base_probe_root.exists() else base_probe_root.with_name(
                    base_probe_root.name + "_attempt_" + unique_stamp())
                if probe.status == "covered":
                    # An incomplete old directory is preserved; the same proposal gets a new attempt.
                    save_probe(probe, probe_root)
            entry = {"proposal_index": index, **probe.summary(), "setup_root": str(setup_root)}
            if probe.status == "covered":
                entry["probe_root"] = str(probe_root)
                accepted.append(entry)
            append_json(ledger_path, entry)
            print(f"[{cohort} 覆盖] 提案 {index + 1}，合格 {len(accepted)}/{required}，{probe.status}: {probe.reason}", flush=True)
            if probe.status not in {"covered", "no_signal", "illegal"}:
                raise RuntimeError("RT 技术错误导致覆盖未知，已保留现场；重新运行会重试同一提案，不会当作无信号跳过")
    finally:
        if provider is not None:
            provider.close()
    if len(accepted) != required:
        raise RuntimeError(f"覆盖采样未达到计划数量：{len(accepted)}/{required}，不得以较小样本冒充完成")
    points = []
    for index, entry in enumerate(accepted):
        point = {"cohort": cohort, "ue_id": f"{cohort.upper()}_{index + 1:04d}",
                 "position_m": entry["position_m"], "probe_root": entry["probe_root"],
                 "setup_root": entry["setup_root"], "channel_category": entry["channel_category"],
                 "has_diffraction": entry["has_diffraction"], "retained_path_count": entry["retained_path_count"],
                 "noise_seeds": [derived_seed(settings["random_seed"], cohort, index, 100 + j) for j in range(settings["noise_repeats"])],
                 "mc_seeds": [derived_seed(settings["random_seed"], cohort, index, 200 + j) for j in range(settings["noise_repeats"])]}
        points.append(point)
    write_json(directory / "frozen_points.json", {"points": points, "coverage_frozen_before_localization": True})
    # Noisy replicas use saved geometry; separate process finishes this phase before timing starts.
    for point in points:
        point["observations"] = []
        probe = load_probe(point["probe_root"])
        for repeat, seed in enumerate(point["noise_seeds"]):
            bundle_root = directory / "observations" / point["ue_id"] / f"repeat_{repeat:03d}"
            receipt = bundle_root.parent / f"repeat_{repeat:03d}_receipt.json"
            if receipt.exists():
                bundle = read_json(receipt)
                for name, digest in (("online_npz", "input_sha256"), ("truth_npz", "truth_sha256"),
                                     ("scene_json", "scene_sha256"), ("generation_manifest", "manifest_sha256")):
                    if file_sha256(bundle[name]) != bundle[digest]:
                        raise ValueError(f"冻结观测文件已改变：{name}")
            else:
                # If an earlier attempt left a partial bundle, preserve it in a new attempt path.
                if bundle_root.exists():
                    bundle_root = bundle_root.with_name(bundle_root.name + "_attempt_" + unique_stamp())
                bundle = write_observation_bundle(probe, config, setup_root=point["setup_root"],
                                                 output_root=bundle_root, noise_seed=seed)
                bundle["input_sha256"] = file_sha256(bundle["online_npz"])
                bundle["truth_sha256"] = file_sha256(bundle["truth_npz"])
                bundle["scene_sha256"] = file_sha256(bundle["scene_json"])
                bundle["manifest_sha256"] = file_sha256(bundle["generation_manifest"])
                write_json(receipt, bundle)
            point["observations"].append(bundle)
        print(f"[{cohort} CSI] {point['ue_id']}：{len(point['observations'])} 份噪声观测已保存", flush=True)
    plan = {"schema_version": 1, "cohort": cohort, "points": points,
            "coverage_frozen_before_localization": True, "expected_requests": required * settings["noise_repeats"] * 2}
    write_json(plan_path, plan)
    return plan


def worker_loop(connection) -> None:
    """仅导入在线模块。job 没有 UE 坐标、噪声等级或真值文件内容。"""
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
        began = time.perf_counter()
        recorder = None
        try:
            with collect_timings(snapshot_path=job.get("snapshot_path")) as recorder:
                result = localize(job["config"], scene_json=job["scene_json"], online_input=job["online_input"],
                                  generation_manifest=job["generation_manifest"], output_root=job["output_root"])
            payload = {"status": "success", "position_m": np.asarray(result["mu_m"]).tolist(),
                       "clock_bias_s": float(result["clock_bias_s"]), "diagnostics": result["diagnostics"],
                       "forward_valid": result["forward_check"]["all_selected_paths_valid"]}
        except Exception as error:
            payload = {"status": "localization_failed", "error": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()}
        if payload["status"] == "success":
            # 额外统计失败不能覆盖已经得到的定位结果；该读取不属于在线计算。
            try:
                representatives = read_json(Path(job["output_root"]) / "localization" / "representative_points.json")
                payload["diffraction_representative_count"] = sum(
                    any(kind == "diffraction" for kind, _ in row["point"].get("propagation_interactions", []))
                    for row in representatives.get("representatives", []))
            except Exception as error:
                payload["diagnostic_read_error"] = f"{type(error).__name__}: {error}"
        payload["timings"] = recorder.to_dict() if recorder is not None else {}
        # Collector ends at localize return including publication, excluding result transfer/evaluation.
        payload["processing_seconds"] = payload["timings"].get("elapsed_s", time.perf_counter() - began)
        connection.send(_jsonable(payload))


class PersistentWorker:
    def __init__(self, *, target=None, startup_timeout_s=60.0, shutdown_grace_s=5.0):
        context = mp.get_context("spawn")
        self._closed = False
        self.shutdown_grace_s = shutdown_grace_s
        self.connection, remote = context.Pipe()
        self.process = context.Process(target=target or worker_loop, args=(remote,))
        began = time.perf_counter()
        self.process.start()
        remote.close()
        if not self.connection.poll(startup_timeout_s):
            self.close()
            raise RuntimeError("在线进程初始化超时")
        try:
            self.ready = self.connection.recv()
        except (EOFError, OSError):
            self.close()
            raise RuntimeError("在线进程在初始化完成前退出") from None
        if not isinstance(self.ready, dict) or self.ready.get("ready") is not True:
            self.close()
            raise RuntimeError("在线进程返回了无效的初始化记录")
        self.startup_seconds = time.perf_counter() - began

    def run(self, job: dict, timeout_s: float) -> dict:
        began = time.perf_counter()
        try:
            self.connection.send(job)
            if self.connection.poll(timeout_s):
                return self.connection.recv()
        except (BrokenPipeError, EOFError, OSError) as error:
            return self._interrupted_payload(job, began, "localization_failed",
                                             f"online_worker_exited: {type(error).__name__}")
        return self._interrupted_payload(job, began, "timeout", "online_trial_timeout")

    def _interrupted_payload(self, job: dict, began: float, status: str, error: str) -> dict:
        # 先固定请求截止时刻，进程回收耗时另存，不能混入定位耗时。
        elapsed = time.perf_counter() - began
        stopped = time.perf_counter()
        self.close()
        shutdown_seconds = time.perf_counter() - stopped
        snapshot = Path(job["snapshot_path"]) if job.get("snapshot_path") else None
        timing_error = None
        try:
            timing = read_json(snapshot) if snapshot and snapshot.exists() else {}
        except (OSError, ValueError) as error:
            timing = {}
            timing_error = f"{type(error).__name__}: {error}"
        payload = {"status": status, "error": error, "timings": timing,
                "processing_seconds": elapsed, "timing_snapshot_incomplete": True,
                "timing_snapshot_error": timing_error,
                "failure_at_perf_counter_s": began + elapsed,
                "worker_shutdown_seconds": shutdown_seconds,
                "worker_exitcode": getattr(getattr(self, "process", None), "exitcode", None)}
        if status == "timeout":
            payload.update(timeout_request_started_perf_counter_s=began,
                           timeout_at_perf_counter_s=began + elapsed,
                           timeout_scope="request_roundtrip_including_input_publication_and_diagnostics")
        return payload

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(self.shutdown_grace_s)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(self.shutdown_grace_s)
        else:
            # 收割已退出的子进程，避免留下僵尸进程。
            self.process.join()
        self.connection.close()


def make_job(config: dict, observation: dict, strategy: str, seed: int, output: Path, snapshots: bool) -> dict:
    public = localization_config_view(config)
    public["project"]["random_seed"] = int(seed)
    public["localization"]["diffraction_representative_policy"] = strategy
    public["output"]["root"] = str(output)
    return {"config": public, "scene_json": observation["scene_json"], "online_input": observation["online_npz"],
            "generation_manifest": observation["generation_manifest"], "output_root": str(output),
            "snapshot_path": str(output / "timing_snapshot.json") if snapshots else None}


def trial_record(point: dict, repeat: int, strategy: str, payload: dict, observation: dict, output: Path) -> dict:
    timing = payload.get("timings", {})
    marks = timing.get("marks", {})
    position = payload.get("position_m", timing.get("mark_data", {}).get("position_available", {}).get("mu_m"))
    bias = payload.get("clock_bias_s", timing.get("mark_data", {}).get("position_available", {}).get("clock_bias_s"))
    # Ground truth is first opened here in the parent, after online timing has stopped.
    if file_sha256(observation["truth_npz"]) != observation["truth_sha256"]:
        raise ValueError("评估真值与冻结输入摘要不一致，拒绝产生精度结果")
    with np.load(observation["truth_npz"], allow_pickle=False) as truth:
        truth_position = np.asarray(truth["ue_position_m"], dtype=float)
        truth_bias = float(truth["clock_bias_s"])
    estimate = np.asarray(position, dtype=float) if position is not None else None
    valid_position = estimate is not None and estimate.shape == (2,) and np.all(np.isfinite(estimate))
    error = float(np.linalg.norm(estimate - truth_position)) if valid_position else None
    if error is not None and not np.isfinite(error):
        error = None
    def duration(end):
        return max(0.0, marks[end] - marks["csi_map_ready"]) if end in marks and "csi_map_ready" in marks else None
    failed_online = duration("online_failed")
    # perf_counter 在本机父子进程共享同一个单调时钟，可精确衔接硬超时边界。
    failure_at = payload.get("failure_at_perf_counter_s", payload.get("timeout_at_perf_counter_s"))
    if (payload["status"] != "success" and "csi_map_ready" in marks
            and timing.get("started_perf_counter_s") is not None
            and failure_at is not None):
        failed_online = max(0.0, failure_at
                            - timing["started_perf_counter_s"] - marks["csi_map_ready"])
    localization_seconds = duration("position_available")
    checked_seconds = duration("checked_complete")
    if localization_seconds is None:
        localization_seconds = failed_online
    if checked_seconds is None:
        checked_seconds = failed_online
    diagnostics = payload.get("diagnostics", {})
    initial_snapshot = timing.get("mark_data", {}).get("initial_candidates_available", {})
    representative_snapshot = timing.get("mark_data", {}).get("representatives_available", {})
    active_stages = [event.get("name") for event in timing.get("events", []) if event.get("status") == "running"]
    stage_rows = timing.get("events", timing.get("stages", []))
    if isinstance(stage_rows, dict):
        stage_rows = list(stage_rows.values())
    return {"cohort": point["cohort"], "ue_id": point["ue_id"], "repeat_index": repeat, "strategy": strategy,
            "noise_seed": point["noise_seeds"][repeat], "mc_seed": point["mc_seeds"][repeat],
            "input_sha256": observation["input_sha256"], "status": payload["status"],
            "position_error_m": error, "clock_bias_error_ns": abs(float(bias) - truth_bias) * 1e9 if bias is not None else None,
            "localization_seconds": localization_seconds, "checked_seconds": checked_seconds,
            "failed_online_seconds": failed_online, "processing_seconds": payload.get("processing_seconds"),
            "forward_valid": payload.get("forward_valid"),
            "identifiable": diagnostics.get("diffraction_physical_constraints", {}).get("locally_identifiable"),
            "initial_count": diagnostics.get("initial_candidate_count", initial_snapshot.get("count")),
            "cluster_count": diagnostics.get("point_clustering", representative_snapshot.get("diagnostics", {})).get("cluster_count"),
            "representative_count": diagnostics.get("representative_point_count", representative_snapshot.get("count")),
            "diffraction_representative_count": payload.get("diffraction_representative_count", representative_snapshot.get("diffraction_count")),
            "initial_candidate_diagnostics": diagnostics.get("initial_candidate_generation", initial_snapshot.get("diagnostics")),
            "representative_diagnostics": diagnostics.get("point_clustering", representative_snapshot.get("diagnostics")),
            "stage_timings": stage_rows, "timing_snapshot_incomplete": payload.get("timing_snapshot_incomplete", False),
            "snapshot_write_s": timing.get("snapshot_write_s"), "result_dir": str(output),
            "channel_category": point["channel_category"], "has_diffraction": point["has_diffraction"],
            "failed_step": timing.get("mark_data", {}).get("online_failed", {}).get("failed_step"),
            "interrupted_stages": active_stages,
            "diagnostic_read_error": payload.get("diagnostic_read_error"),
            "worker_pid": payload.get("worker_pid"), "worker_generation": payload.get("worker_generation"),
            "execution_state": payload.get("execution_state"), "warmup_state": payload.get("warmup_state"),
            "completed_warmups": payload.get("completed_warmups"),
            "worker_requests_before": payload.get("worker_requests_before"),
            "pair_cache_comparable": payload.get("pair_cache_comparable", False),
            "worker_shutdown_seconds": payload.get("worker_shutdown_seconds"),
            "worker_exitcode": payload.get("worker_exitcode"),
            "error": payload.get("error")}


def update_report(root: Path, cohort: str, settings: dict) -> dict:
    from .boundary_report import create_boundary_report
    plan = read_json(root / cohort / "plan.json")
    records = rows(root / cohort / "trials.jsonl")
    return create_boundary_report(root / cohort / "report", records, plan["points"],
                                  metadata={"plots": settings["plots"], "noise_repeats": settings["noise_repeats"],
                                            "scene_json": plan["points"][0]["observations"][0]["scene_json"],
                                            "channel_backend": settings["channel_backend"],
                                            "timing_snapshots": settings["timing_snapshots"]})


def benchmark_cohort(root: Path, cohort: str, settings: dict, config: dict) -> None:
    plan = read_json(root / cohort / "plan.json")
    points = plan["points"]
    log = root / cohort / "trials.jsonl"
    existing = rows(log)
    keys = {(r["ue_id"], r["repeat_index"], r["strategy"]) for r in existing}
    expected_keys = {(p["ue_id"], repeat, strategy) for p in points
                     for repeat in range(settings["noise_repeats"]) for strategy in STRATEGIES}
    if len(keys) != len(existing) or not keys <= expected_keys:
        raise ValueError("已有定位记录包含重复请求或计划外请求；拒绝改变冻结分母")
    if len(expected_keys) != plan["expected_requests"]:
        raise ValueError("UE 清单、重复次数与冻结请求总数不一致")
    rng = np.random.default_rng(derived_seed(settings["random_seed"], cohort, 0, 900))
    pairs = [(p, r) for p in points for r in range(settings["noise_repeats"])]
    rng.shuffle(pairs)
    # Balanced alternating order (with a seeded starting arm); both compute full CSI pipeline.
    first = int(rng.integers(2))
    workers = {}
    worker_states = {}
    generation = 0
    warmup_failed = False
    attempt = root / cohort / "benchmark_attempts" / unique_stamp()
    attempt.mkdir(parents=True)

    def close_workers():
        for worker in workers.values():
            if worker is not None:
                worker.close()
        workers.clear()

    def start_workers():
        nonlocal generation
        generation += 1
        worker_states.clear()
        for strategy in STRATEGIES:
            started = time.perf_counter()
            target = attempt / f"workers_{generation:03d}" / strategy
            try:
                worker = PersistentWorker()
            except Exception as error:
                workers[strategy] = None
                worker_states[strategy] = {
                    "worker_pid": None, "worker_generation": generation,
                    "startup_error": f"{type(error).__name__}: {error}",
                    "startup_seconds": time.perf_counter() - started,
                    "worker_requests_before": 0, "completed_warmups": 0,
                    "warmup_state": "worker_unavailable"}
            else:
                workers[strategy] = worker
                worker_states[strategy] = {
                    "worker_pid": worker.ready["pid"], "worker_generation": generation,
                    "startup_seconds": worker.startup_seconds,
                    "worker_requests_before": 0, "completed_warmups": 0,
                    "warmup_state": ("disabled_after_failure" if warmup_failed else
                                     "disabled_by_config" if settings["warmup_per_strategy"] == 0 else
                                     "pending" if generation == 1 else "skipped_after_worker_restart")}
            write_json(target / "startup.json", worker_states[strategy])

    def initialize_pair():
        nonlocal warmup_failed
        start_workers()
        # 每次尝试最多执行一轮指定预热。失败后不再反复消耗同一困难输入的预算。
        if generation != 1 or settings["warmup_per_strategy"] == 0:
            return
        if any(worker is None for worker in workers.values()):
            for strategy, worker in workers.items():
                if worker is not None:
                    worker_states[strategy]["warmup_state"] = "skipped_after_startup_failure"
            return
        warm_point = (read_json(root / "pilot" / "plan.json")["points"][0]
                      if cohort == "formal" else points[0])
        for warm_index in range(settings["warmup_per_strategy"]):
            for strategy in STRATEGIES:
                worker = workers[strategy]
                target = attempt / f"workers_{generation:03d}" / strategy / f"warmup_{warm_index:03d}"
                job = make_job(config, warm_point["observations"][0], strategy,
                               warm_point["mc_seeds"][0], target, settings["timing_snapshots"])
                print(f"[{cohort} 预热开始] {warm_index+1}/{settings['warmup_per_strategy']} "
                      f"{strategy} UE={warm_point['ue_id']} PID={worker.ready['pid']} "
                      f"预算={settings['trial_timeout_s']} 秒", flush=True)
                payload = worker.run(job, settings["trial_timeout_s"])
                payload.update(strategy=strategy, warmup_index=warm_index,
                               worker_pid=worker.ready["pid"], worker_generation=generation,
                               result_dir=str(target), phase="warmup")
                write_json(target / "warmup_result.json", payload)
                append_json(attempt / "warmups.jsonl", payload)
                print(f"[{cohort} 预热结束] {strategy}: {payload['status']}", flush=True)
                if payload["status"] != "success" or not worker.process.is_alive():
                    warmup_failed = True
                    write_json(attempt / "warmup_failure.json", {
                        "status": payload["status"], "strategy": strategy,
                        "result_dir": str(target), "continue_planned_requests": True,
                        "remaining_warmups": "skipped; both strategy workers restarted cold"})
                    print(f"[{cohort} 预热失败] 已保留现场；两组重新启动，跳过剩余预热，继续固定 UE 的实际请求。", flush=True)
                    close_workers()
                    start_workers()
                    return
                worker_states[strategy]["completed_warmups"] += 1
        for state in worker_states.values():
            state["warmup_state"] = "completed"

    try:
        for pair_index, (point, repeat) in enumerate(pairs):
            strategies = STRATEGIES if (pair_index + first) % 2 == 0 else STRATEGIES[::-1]
            pending = [strategy for strategy in strategies if (point["ue_id"], repeat, strategy) not in keys]
            if not pending:
                continue
            if not workers:
                initialize_pair()
            # 隔离两组缓存，同一配对开始前执行次数和预热次数相等才标为计时可比。
            comparable = (len(pending) == len(STRATEGIES)
                          and all(worker is not None and worker.process.is_alive() for worker in workers.values())
                          and len({(s["worker_requests_before"], s["completed_warmups"], s["warmup_state"])
                                   for s in worker_states.values()}) == 1)
            restart_after_pair = len(pending) != len(STRATEGIES)
            for strategy in strategies:
                key = (point["ue_id"], repeat, strategy)
                if key in keys:
                    continue
                worker = workers[strategy]
                state = worker_states[strategy]
                observation = point["observations"][repeat]
                for name, digest in (("online_npz", "input_sha256"), ("scene_json", "scene_sha256"),
                                     ("generation_manifest", "manifest_sha256")):
                    if file_sha256(observation[name]) != observation[digest]:
                        raise ValueError(f"输入文件与冻结指纹不一致：{name}")
                target = attempt / point["ue_id"] / f"repeat_{repeat:03d}" / strategy
                job = make_job(config, observation, strategy, point["mc_seeds"][repeat], target, settings["timing_snapshots"])
                execution_state = ("unavailable" if worker is None else
                                   "reused" if state["worker_requests_before"] else
                                   "prewarmed" if state["completed_warmups"] else "cold")
                print(f"[{cohort} 定位] {len(keys)+1}/{plan['expected_requests']} {point['ue_id']} "
                      f"噪声 {repeat+1} {strategy} PID={state['worker_pid']} 进程状态={execution_state}", flush=True)
                if worker is None:
                    payload = {"status": "localization_failed", "timings": {},
                               "error": "online_worker_startup_failed: " + state["startup_error"],
                               "processing_seconds": None}
                else:
                    payload = worker.run(job, settings["trial_timeout_s"])
                payload.update({key: state[key] for key in (
                    "worker_pid", "worker_generation", "worker_requests_before", "completed_warmups", "warmup_state")})
                payload.update(execution_state=execution_state, pair_cache_comparable=comparable)
                state["worker_requests_before"] += 1
                write_json(target / "worker_result.json", payload)
                record = trial_record(point, repeat, strategy, payload, observation, target)
                append_json(log, record)
                keys.add(key)
                restart_after_pair |= payload["status"] != "success" or worker is None or not worker.process.is_alive()
                print(f"[{cohort} 已记录] {len(keys)}/{plan['expected_requests']} {record['status']} 误差={record['position_error_m']} 米", flush=True)
            if restart_after_pair:
                print(f"[{cohort} 进程恢复] 当前配对已保留；两组共同重新启动，下一配对标记冷启动。", flush=True)
                close_workers()
    finally:
        close_workers()
        update_report(root, cohort, settings)
    if keys != expected_keys:
        raise RuntimeError(f"计划请求尚未全部记录：{len(keys)}/{len(expected_keys)}")
    write_json(root / cohort / "cohort_completion.json", {"status": "completed", "planned_requests": plan["expected_requests"],
               "recorded_requests": len(keys), "finished_at": datetime.now(timezone.utc).isoformat(),
               "meaning": "所有计划请求均有记录；定位失败、超时仍作为实验结果保留"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-prepared-from", type=Path,
                        help="在新输出目录复用另一实验已冻结的UE和CSI，不重采样、不复制旧定位结果")
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--action", choices=("run", "prepare", "benchmark", "report"), default="run")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_handle = None
    if args.action == "run":
        import fcntl
        lock_handle = (root / ".experiment_run.lock").open("a")
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("同一实验目录已有任务运行，拒绝并发修改")
    settings, config = load_settings(args.config.resolve())
    if args.action != "report":
        from .gpu_runtime import validate_gpu_selection
        validate_gpu_selection(config, channel_backend=settings["channel_backend"])
    freeze_experiment(root, settings, config)
    if args.reuse_prepared_from is not None:
        from .boundary_reuse import reuse_prepared_cohort
        reuse_prepared_cohort(args.reuse_prepared_from, root, args.phase, settings, config)
    if args.phase == "formal":
        completed = root / "pilot" / "cohort_completion.json"
        if not completed.exists() or read_json(completed)["status"] != "completed":
            raise ValueError("正式实验需要同一目录内已完成的预跑记录；先运行 --phase pilot")
        freeze_path = root / "formal_parameters_frozen.json"
        if not freeze_path.exists():
            write_json(freeze_path, {"frozen_at": datetime.now(timezone.utc).isoformat(),
                       "experiment_sha256": file_sha256(root / "experiment.json"),
                       "pilot_completion_sha256": file_sha256(completed), "parameters": localization_config_view(config)})
    if args.action == "run":
        # RT interpreter exits before localization. No generation GPU memory or work overlaps timing.
        for action in ("prepare", "benchmark"):
            subprocess.run([sys.executable, "-u", "-m", __package__ + ".boundary_experiment",
                            "--config", str(args.config.resolve()), "--output", str(root),
                            "--phase", args.phase, "--action", action], check=True)
    elif args.action == "prepare":
        prepare_cohort(root, args.phase, settings, config)
    elif args.action == "benchmark":
        benchmark_cohort(root, args.phase, settings, config)
    else:
        update_report(root, args.phase, settings)
    print(f"完成 {args.phase}/{args.action}，结果目录：{root / args.phase}", flush=True)


if __name__ == "__main__":
    main()
