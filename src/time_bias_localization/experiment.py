"""固定 UE 采样计划，复用每点信道生成独立噪声重复，再调用原定位流程。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import math
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import sys
import time
import traceback

import numpy as np
import yaml

from .config import load_config, load_localization_config, localization_config_view
from .pipeline import WORKFLOW, generate_data, localize, evaluate
from .provenance import (artifact_record, exclusive_output_root_lock, generation_bundle_id,
                         load_generation_manifest, verify_generation_artifact)
from .scene import Scene2D
from .signal import apply_common_delay_bias
from .visualization import checked_record, read_json, write_json


def sample_positions(scene: Scene2D, bounds, *, count: int, seed: int, bs, wall_clearance: float, bs_clearance: float):
    """在用户指定的单个空旷矩形内均匀采样，绝不按定位结果筛点。

    整个矩形（含墙边距）不允许与墙段包围盒重叠，采用保守检查。
    场景墙线不能判断封闭区域的用途；矩形必须由用户指定为可通行区域。
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("采样范围必须为四个有限数 [xmin,xmax,ymin,ymax]")
    x0, x1, y0, y1 = bounds
    sx0, sx1, sy0, sy1 = scene.bounds_m
    if not (sx0 <= x0 < x1 <= sx1 and sy0 <= y0 < y1 <= sy1):
        raise ValueError("采样矩形必须位于固定场景范围内")
    if count < 1 or not np.isfinite(wall_clearance) or wall_clearance < 0 or not np.isfinite(bs_clearance) or bs_clearance < 0:
        raise ValueError("采样数必须为正，距离限制必须非负且有限")
    for wall in scene.walls:
        low, high = np.minimum(wall.start, wall.end), np.maximum(wall.start, wall.end)
        if high[0] >= x0 - wall_clearance and low[0] <= x1 + wall_clearance and high[1] >= y0 - wall_clearance and low[1] <= y1 + wall_clearance:
            raise ValueError(f"采样矩形及墙边距与墙段包围盒重叠：{wall.wall_id}；请缩小或调整区域")
    rng = np.random.default_rng(seed)
    positions = []
    for _ in range(count * 1000):
        point = rng.uniform([x0, y0], [x1, y1])
        if np.linalg.norm(point - np.asarray(bs)) >= bs_clearance:
            positions.append(point.tolist())
        if len(positions) == count:
            return positions
    raise ValueError("无法在指定区域满足 BS 最小距离限制")


def prepare_experiment(config_path: Path, output: Path) -> dict:
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {"generation_config", "scene_json", "sampling_bounds_m", "ue_count", "noise_repeats", "random_seed", "wall_clearance_m", "bs_clearance_m"}
    if set(spec) != required:
        raise ValueError(f"实验配置必须严格包含：{sorted(required)}")
    def resolve(value):
        path = Path(value).expanduser()
        return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
    generation_path = resolve(spec["generation_config"])
    generation = load_config(generation_path)
    scene_path = resolve(spec["scene_json"])
    scene = Scene2D.load(scene_path)
    bounds = generation["scene"].get("localization_bounds_m") or generation["scene"]["bounds_m"]
    if not np.allclose(scene.bounds_m, bounds) or not np.isclose(scene.fixed_height_m, generation["scene"]["fixed_height_m"]):
        raise ValueError("采样地图与生成配置的固定范围或高度不一致")
    for name in ("ue_count", "noise_repeats", "random_seed"):
        if type(spec[name]) is not int or spec[name] < (0 if name == "random_seed" else 1):
            raise ValueError(f"{name} 必须为合法整数")
    positions = sample_positions(scene, spec["sampling_bounds_m"], count=spec["ue_count"], seed=spec["random_seed"],
                                 bs=generation["simulation"]["bs_position_m"], wall_clearance=spec["wall_clearance_m"], bs_clearance=spec["bs_clearance_m"])
    output.mkdir(parents=True, exist_ok=False)
    generation.pop("_config_path", None)
    generation_snapshot = output / "generation_template.yaml"
    generation_snapshot.write_text(yaml.safe_dump(generation, allow_unicode=True, sort_keys=False), encoding="utf-8")
    # 采样计划在任何信道生成或定位之前固定，所有未执行项也进入统计分母。
    plan = dict(schema_version=1, workflow=WORKFLOW, scene=artifact_record(scene_path), generation_template=artifact_record(generation_snapshot),
                source_experiment_config=artifact_record(config_path), source_generation_config=artifact_record(generation_path),
                sampling_bounds_m=spec["sampling_bounds_m"], random_seed=spec["random_seed"], noise_repeats=spec["noise_repeats"],
                bs_position_m=generation["simulation"]["bs_position_m"], bs_boresight_rad=math.radians(generation["radio"]["bs_boresight_deg"]),
                clock_bias_s=generation["simulation"]["clock_bias_s"], snr_db=generation["radio"]["snr_db"],
                sampling_rule="uniform_in_declared_free_rectangle_without_outcome_filtering",
                wall_clearance_m=spec["wall_clearance_m"], bs_clearance_m=spec["bs_clearance_m"],
                points=[dict(ue_id=f"UE{i + 1:03d}", position_m=point,
                             channel_seed=spec["random_seed"] + 10000 + i,
                             noise_seeds=[spec["random_seed"] + 1000000 + i * spec["noise_repeats"] + j for j in range(spec["noise_repeats"])])
                        for i, point in enumerate(positions)])
    write_json(output / "experiment_plan.json", plan)
    return plan


def make_noise_repeat(parent_manifest: Path, destination: Path, *, seed: int) -> Path:
    """生成侧操作：保留同一几何信道和偏差，只替换独立噪声。"""
    manifest, _, _ = load_generation_manifest(parent_manifest)
    for key, record in manifest["artifact_hashes"].items():
        verify_generation_artifact(manifest, key, record["path"])
    with np.load(manifest["artifact_hashes"]["online_measurement"]["path"], allow_pickle=False) as data:
        online = {key: data[key] for key in data.files}
    truth_source = Path(manifest["artifact_hashes"]["ground_truth"]["path"])
    with np.load(truth_source, allow_pickle=False) as data:
        truth = {key: data[key] for key in data.files}
    online["csi_observed"] = apply_common_delay_bias(
        truth["csi_geometric"], online["subcarrier_frequencies_hz"], float(truth["clock_bias_s"]),
        noise_std=float(truth["injected_noise_std"]), seed=seed)
    data_root = destination / "data"
    data_root.mkdir(parents=True, exist_ok=False)
    online_path, truth_path = data_root / "online/measurement.npz", data_root / "truth/ground_truth.npz"
    online_path.parent.mkdir()
    truth_path.parent.mkdir()
    np.savez_compressed(online_path, **online)
    np.savez_compressed(truth_path, **truth)
    shutil.copy2(truth_source.with_suffix(".json"), truth_path.with_suffix(".json"))
    online_source = Path(manifest["artifact_hashes"]["online_measurement"]["path"])
    shutil.copy2(online_source.parent / "manifest.json", online_path.parent / "manifest.json")
    manifest["artifact_hashes"]["online_measurement"] = artifact_record(online_path)
    manifest["artifact_hashes"]["ground_truth"] = artifact_record(truth_path)
    if manifest["stage"] == "synthetic_csi_generation":
        manifest.update(online_input=str(online_path), truth_input=str(truth_path))
    else:
        manifest["data_artifacts"] = {"online_npz": str(online_path), "truth_npz": str(truth_path)}
    manifest["bundle_id"] = generation_bundle_id(manifest)
    target = destination / "generation_manifest.json"
    write_json(target, manifest)
    # 严格生成清单保持原协议，额外来源信息独立保存，不能传给定位器。
    write_json(destination / "noise_generation.json", dict(parent_manifest=artifact_record(parent_manifest), noise_seed=seed,
               generation_manifest=artifact_record(target), rule="same_geometric_csi_new_independent_noise"))
    return target


def batch_localization_config(generation: dict) -> dict:
    """从生成配置提取公开输入，并使用 Sionna 明确声明的固定地图范围。"""
    config = localization_config_view(generation)
    if generation["scene"]["source"] == "sionna_builtin":
        # load_config 合并默认值后可能仍含合成房间 bounds_m；Sionna 的真实
        # 公共地图范围来自 localization_bounds_m，不能沿用房间默认范围。
        config["scene"]["bounds_m"] = deepcopy(generation["scene"]["localization_bounds_m"])
    return config


def _run_point(output: Path, point: dict, noise_repeats: int, template: dict,
               scene_path: Path, compute: dict, worker: dict) -> None:
    """每个 UE 独占一个输出目录，原来的逐重复保存和不覆盖规则保持不变。"""
    ue_root = output / point["ue_id"]
    ue_root.mkdir(exist_ok=True)
    if all((ue_root / f"repeat_{repeat:03d}/attempt.json").exists() for repeat in range(noise_repeats)):
        return
    source_root = ue_root / "channel"
    generation = deepcopy(template)
    generation.pop("_config_path", None)
    generation["compute"] = deepcopy(compute)
    generation["simulation"]["ue_position_m"] = point["position_m"]
    generation["project"]["random_seed"] = point["channel_seed"]
    generation["output"]["root"] = str(source_root)
    generation["simulation"]["allow_overwrite"] = False
    generation["simulation"]["deepmimo_scenario_name"] = f"{output.name}_{point['ue_id']}".lower()
    channel_config = ue_root / "generation.yaml"
    if not channel_config.exists():
        channel_config.write_text(yaml.safe_dump(generation, allow_unicode=True, sort_keys=False), encoding="utf-8")
    parent_manifest = source_root / ("generation_manifest.json" if generation["scene"]["source"] == "sionna_builtin" else "data/generation_manifest.json")
    channel_error = None
    if not parent_manifest.exists():
        try:
            print(f"{point['ue_id']}: 生成固定信道", flush=True)
            if generation["scene"]["source"] == "sionna_builtin":
                from .sionna_generation import generate_sionna_deepmimo_bundle
                generate_sionna_deepmimo_bundle(load_config(channel_config), output_root=source_root)
            else:
                generate_data(load_config(channel_config), scene_json=scene_path)
        except Exception as error:
            channel_error = f"{type(error).__name__}: {error}"
            failure_path = ue_root / f"channel_failure_{time.time_ns()}.txt"
            failure_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"{point['ue_id']}: 信道生成失败：{channel_error}；详细日志：{failure_path}", flush=True)
    for repeat, seed in enumerate(point["noise_seeds"]):
        root = ue_root / f"repeat_{repeat:03d}"
        status_path = root / "attempt.json"
        if status_path.exists():
            continue
        if root.exists():
            # 不覆盖中断现场，也不偷偷重跑改变结果。
            write_json(status_path, dict(worker=worker, status="interrupted_failed", error="该目录有中断残留，已保留；新实验目录可重新运行"))
            continue
        root.mkdir()
        if channel_error:
            write_json(status_path, dict(worker=worker, status="generation_failed", error=channel_error))
            continue
        stage = "generation"
        try:
            manifest_path = make_noise_repeat(parent_manifest, root, seed=seed)
            manifest = read_json(manifest_path)
            # 先保存过滤后的公开配置，再经严格入口读取。
            config = batch_localization_config(generation)
            config.pop("_config_path", None)
            config["output"]["root"] = str(root)
            config["project"]["random_seed"] = seed + 100000000
            config_path = root / "localization.yaml"
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            print(f"{point['ue_id']} / {repeat + 1}: 定位", flush=True)
            stage = "localization"
            start = time.perf_counter()
            result = localize(load_localization_config(config_path),
                scene_json=manifest["artifact_hashes"]["scene_json"]["path"],
                online_input=root / "data/online/measurement.npz", generation_manifest=manifest_path,
                run_receipt=root / "receipt.json")
            elapsed = time.perf_counter() - start
            stage = "evaluation"
            evaluate(result_json=root / "localization/localization_result.json", truth_npz=root / "data/truth/ground_truth.npz",
                     output_json=root / "evaluation/metrics.json", expected_run_id=result["localization_run_id"])
            write_json(status_path, dict(worker=worker, status="success", run_id=result["localization_run_id"], localization_seconds=elapsed, noise_seed=seed))
        except Exception as error:
            (root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            failure = dict(worker=worker, status=f"{stage}_failed", error=f"{type(error).__name__}: {error}", noise_seed=seed)
            if getattr(error, "failure_progress", None):
                failure["failure_progress"] = error.failure_progress
            write_json(status_path, failure)
            stage_label = {"generation": "噪声生成", "localization": "定位", "evaluation": "评估"}[stage]
            print(f"{point['ue_id']} / {repeat + 1}: {stage_label}失败：{error}；详细日志：{root / 'failure.txt'}", flush=True)


_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def _execution_options(compute_backend, gpu_ids, workers, music_batch_size,
                       music_angle_chunk_size, cpu_threads) -> dict:
    if compute_backend not in {"numpy", "cuda"}:
        raise ValueError("compute_backend 只能为 numpy 或 cuda")
    for name, value in dict(workers=workers, music_batch_size=music_batch_size,
                            music_angle_chunk_size=music_angle_chunk_size,
                            cpu_threads=cpu_threads).items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} 必须为正整数")
    devices = [value.strip() for value in str(gpu_ids).split(",")]
    if any(not value.isdecimal() for value in devices):
        raise ValueError("gpu_ids 必须为不重复的 GPU 编号，例如 0,1")
    devices = [str(int(value)) for value in devices]
    if len(set(devices)) != len(devices):
        raise ValueError("gpu_ids 必须为不重复的 GPU 编号，例如 0,1")
    return dict(backend=compute_backend, gpu_ids=devices,
                worker_count=len(devices) if compute_backend == "cuda" else workers,
                batch_size=music_batch_size, angle_chunk_size=music_angle_chunk_size,
                cpu_threads=cpu_threads)


@contextmanager
def _spawn_environment(options: dict, device: str | None):
    """spawn 启动新解释器前设置环境，早于 NumPy/Sionna 的首次导入。"""
    values = {name: str(options["cpu_threads"]) for name in _THREAD_VARIABLES}
    values["TBC_REQUIRE_CUDA"] = "1" if device is not None else "0"
    if device is not None:
        values["CUDA_VISIBLE_DEVICES"] = device
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _worker_record(worker_id: str, options: dict, device: str | None) -> dict:
    return dict(worker_id=worker_id, pid=os.getpid(), backend=options["backend"],
                requested_gpu_id=device, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                local_device_id=0 if device is not None else None,
                thread_limit_method=("existing_process_not_reconfigured" if worker_id == "main_process"
                                     else "environment_before_python_start"),
                cwd=str(Path.cwd()), thread_environment={name: os.environ.get(name) for name in _THREAD_VARIABLES})


def _worker_compute(options: dict) -> dict:
    return dict(backend=options["backend"], device_id=0, batch_size=options["batch_size"],
                angle_chunk_size=options["angle_chunk_size"])


def _point_status(output: Path, point: dict, worker: dict) -> dict:
    statuses = [read_json(output / point["ue_id"] / f"repeat_{i:03d}/attempt.json")["status"]
                for i in range(len(point["noise_seeds"]))]
    return dict(ue_id=point["ue_id"], worker=worker, status="completed",
                success_count=statuses.count("success"), failed_count=len(statuses) - statuses.count("success"),
                repeat_statuses=statuses)


def _fail_point(output: Path, point: dict, error: str, worker: dict | None = None) -> None:
    """进程异常退出也保存终态，不删中间产物、不重写已有成功/失败记录。"""
    for repeat, seed in enumerate(point["noise_seeds"]):
        status_path = output / point["ue_id"] / f"repeat_{repeat:03d}/attempt.json"
        if not status_path.exists():
            write_json(status_path, dict(status="worker_failed", error=error, worker=worker, noise_seed=seed))


def _experiment_worker(task_queue, event_queue, output: Path, run_root: Path,
                       options: dict, worker_id: str, device: str | None) -> None:
    """一张 GPU 对应一个常驻进程，依次处理 UE，所有绘图留给主进程。"""
    worker = None
    try:
        workdir = run_root / "workers" / worker_id
        workdir.mkdir(parents=True)
        os.chdir(workdir)
        worker = _worker_record(worker_id, options, device)
        compute = _worker_compute(options)
        if options["backend"] == "cuda":
            from .compute import ComputeSettings, MusicComputer
            # 明确要求 GPU；无 CUDA/CuPy/显存时直接报告启动失败，绝不退回 CPU。
            computer = MusicComputer(ComputeSettings(**compute))
            worker["compute_device"] = computer.metadata()
        plan = read_json(output / "experiment_plan.json")
        if plan.get("workflow") != WORKFLOW:
            raise ValueError("实验计划不是当前点聚类流程，不能混入不同流程的结果；请新建实验输出目录")
        template = load_config(checked_record(plan["generation_template"]))
        scene_path = checked_record(plan["scene"])
        write_json(workdir / "worker.json", worker)
        event_queue.put(dict(kind="ready", worker=worker))
        while True:
            point = task_queue.get()
            if point is None:
                return
            event_queue.put(dict(kind="started", ue_id=point["ue_id"], worker=worker))
            try:
                _run_point(output, point, plan["noise_repeats"], template, scene_path, compute, worker)
                event_queue.put(dict(kind="completed", **_point_status(output, point, worker)))
            except Exception as error:
                detail = f"{type(error).__name__}: {error}"
                failure_path = workdir / f"{point['ue_id']}_failure.txt"
                failure_path.write_text(traceback.format_exc(), encoding="utf-8")
                _fail_point(output, point, detail, worker)
                event_queue.put(dict(kind="completed", **_point_status(output, point, worker), error=detail))
                print(f"{point['ue_id']}: 工作进程任务失败：{detail}；详细日志：{failure_path}", flush=True)
    except BaseException as error:
        event_queue.put(dict(kind="worker_failed", worker_id=worker_id, worker=worker,
                             error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
        raise


def _run_parallel(output: Path, run_root: Path, points: list[dict], options: dict) -> None:
    context = multiprocessing.get_context("spawn")
    task_queue, event_queue = context.Queue(), context.Queue()
    processes = {}
    devices = options["gpu_ids"] if options["backend"] == "cuda" else [None] * options["worker_count"]
    done, started, ready = set(), {}, set()
    try:
        for index, device in enumerate(devices):
            worker_id = f"worker_{index:03d}"
            process = context.Process(target=_experiment_worker,
                args=(task_queue, event_queue, output, run_root, options, worker_id, device),
                name=f"ue-{worker_id}")
            # 不能只在 initializer 设置线程数；spawn 会先导入模块再调用函数。
            with _spawn_environment(options, device):
                process.start()
            processes[worker_id] = process
        # 所有指定 GPU 都通过预检后才发布任务，避免一张坏卡造成半批实验。
        while len(ready) < len(processes):
            try:
                event = event_queue.get(timeout=.25)
            except queue.Empty:
                dead = [name for name, process in processes.items() if process.exitcode is not None]
                if dead:
                    raise RuntimeError(f"工作进程启动失败：{dead}，详见 {run_root / 'workers'}")
                continue
            if event["kind"] == "worker_failed":
                write_json(run_root / "workers" / event["worker_id"] / "failure.json", event)
                raise RuntimeError(f"工作进程启动失败：{event['error']}")
            if event["kind"] == "ready":
                ready.add(event["worker"]["worker_id"])
        for point in points:
            task_queue.put(point)
        for _ in processes:
            task_queue.put(None)
        while len(done) < len(points):
            try:
                event = event_queue.get(timeout=.25)
            except queue.Empty:
                if all(process.exitcode is not None for process in processes.values()):
                    break
                continue
            if event["kind"] == "started":
                started[event["ue_id"]] = event["worker"]
                write_json(run_root / "tasks" / f"{event['ue_id']}.json", dict(event, status="running"))
            elif event["kind"] == "completed":
                done.add(event["ue_id"])
                write_json(run_root / "tasks" / f"{event['ue_id']}.json", event)
                print(f"{event['ue_id']}: 完成，成功 {event['success_count']}，失败 {event['failed_count']}", flush=True)
            elif event["kind"] == "worker_failed":
                write_json(run_root / "workers" / event["worker_id"] / "failure.json", event)
        if len(done) != len(points):
            detail = "工作进程异常退出，未完成任务已保留为失败；请查看 execution_runs 中的工作进程日志"
            for point in points:
                if point["ue_id"] not in done:
                    _fail_point(output, point, detail, started.get(point["ue_id"]))
                    write_json(run_root / "tasks" / f"{point['ue_id']}.json",
                               dict(ue_id=point["ue_id"], status="worker_failed", error=detail,
                                    worker=started.get(point["ue_id"])))
            raise RuntimeError(detail)
    finally:
        for worker_id, process in processes.items():
            stopped_by_parent = False
            process.join(timeout=1)
            if process.is_alive():
                stopped_by_parent = True
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            write_json(run_root / "workers" / worker_id / "exit.json",
                       dict(pid=process.pid, exitcode=process.exitcode, stopped_by_parent=stopped_by_parent))
        # 异常退出时队列中可能留有未消费任务，不能等待 feeder 而无限阻塞。
        task_queue.cancel_join_thread()
        task_queue.close()
        event_queue.close()


def run_experiment(output: Path, *, compute_backend: str = "numpy", gpu_ids: str = "0",
                   workers: int = 1, music_batch_size: int = 4,
                   music_angle_chunk_size: int = 32, cpu_threads: int = 1) -> None:
    """默认保持串行；GPU 每卡一个常驻进程，CPU 可按 UE 分进程。"""
    output = Path(output).resolve()
    options = _execution_options(compute_backend, gpu_ids, workers, music_batch_size,
                                 music_angle_chunk_size, cpu_threads)
    with exclusive_output_root_lock(output):
        plan = read_json(output / "experiment_plan.json")
        template = load_config(checked_record(plan["generation_template"]))
        scene_path = checked_record(plan["scene"])
        points = [point for point in plan["points"] if not all(
            (output / point["ue_id"] / f"repeat_{i:03d}/attempt.json").exists()
            for i in range(plan["noise_repeats"]))]
        run_root = output / "execution_runs" / f"{time.time_ns()}_{os.getpid()}"
        run_root.mkdir(parents=True)
        write_json(run_root / "execution_config.json", dict(options, experiment_plan=artifact_record(output / "experiment_plan.json"),
                   requested_points=[point["ue_id"] for point in points], python_executable=sys.executable,
                   plotting="main_process_after_all_localization", started_ns=time.time_ns()))
        for point in points:
            write_json(run_root / "tasks" / f"{point['ue_id']}.json", dict(ue_id=point["ue_id"], status="pending"))
        try:
            if points and (compute_backend == "cuda" or workers > 1):
                _run_parallel(output, run_root, points, options)
            else:
                worker = _worker_record("main_process", options, None)
                write_json(run_root / "workers/main_process/worker.json", worker)
                for point in points:
                    task_path = run_root / "tasks" / f"{point['ue_id']}.json"
                    write_json(task_path, dict(ue_id=point["ue_id"], status="running", worker=worker))
                    _run_point(output, point, plan["noise_repeats"], template, scene_path, _worker_compute(options), worker)
                    write_json(task_path, _point_status(output, point, worker))
            write_json(run_root / "status.json", dict(status="completed", finished_ns=time.time_ns()))
        except BaseException as error:
            # 启动失败和人为中断也结束本轮调度记录，但不凭空生成未执行样本的
            # attempt.json；修复设备或恢复运行后，未开始的 UE 仍可继续。
            for point in points:
                task_path = run_root / "tasks" / f"{point['ue_id']}.json"
                task = read_json(task_path)
                if task["status"] in {"pending", "running"}:
                    task.update(status="interrupted" if task["status"] == "running" else "cancelled",
                                error=f"{type(error).__name__}: {error}", finished_ns=time.time_ns())
                    write_json(task_path, task)
            write_json(run_root / "status.json", dict(status="failed", error=f"{type(error).__name__}: {error}", finished_ns=time.time_ns()))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="固定 BS、多 UE、独立噪声重复的小实验")
    parser.add_argument("--config", type=Path, help="新实验配置；继续已有实验时省略")
    parser.add_argument("--output", type=Path, required=True, help="实验目录；已有计划可继续")
    parser.add_argument("--plan-only", action="store_true", help="只固定采样计划，不执行射线追踪和定位")
    parser.add_argument("--report-output", type=Path, help="新图表目录；省略时不绘图")
    parser.add_argument("--compute-backend", choices=("numpy", "cuda"), default="numpy", help="MUSIC 使用 CPU 或 GPU；请求 GPU 后不自动回退")
    parser.add_argument("--gpu-ids", default="0", help="物理 GPU 编号，逗号分隔；每卡一个工作进程")
    parser.add_argument("--workers", type=int, default=1, help="CPU 模式的 UE 并行进程数")
    parser.add_argument("--music-batch-size", type=int, default=4, help="同时计算的 MUSIC 输入数量")
    parser.add_argument("--music-angle-chunk-size", type=int, default=32, help="每次计算的 MUSIC 角度网格数，控制显存")
    parser.add_argument("--cpu-threads", type=int, default=1, help="每个子进程的 CPU 数值计算线程数；直接单进程调用沿用已有线程设置")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.config:
        prepare_experiment(args.config.resolve(), output)
    elif not (output / "experiment_plan.json").exists():
        parser.error("新实验必须提供 --config")
    if not args.plan_only:
        run_experiment(output, compute_backend=args.compute_backend, gpu_ids=args.gpu_ids,
                       workers=args.workers, music_batch_size=args.music_batch_size,
                       music_angle_chunk_size=args.music_angle_chunk_size, cpu_threads=args.cpu_threads)
    if args.report_output:
        from .visualization import create_report
        create_report(args.report_output.resolve(), experiment=output)
    print(output)


if __name__ == "__main__":
    main()
