"""按 UE 分进程后仍保留相同输入、终态和失败现场。"""

from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from time_bias_localization.config import DEFAULT_CONFIG
from time_bias_localization.experiment import (
    _execution_options, _fail_point, _run_parallel, _spawn_environment,
    prepare_experiment, run_experiment,
)
from time_bias_localization.pipeline import prepare_scene
from time_bias_localization.provenance import file_sha256
from time_bias_localization.visualization import read_json, write_json


@pytest.fixture
def small_experiment(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    config["output"]["root"] = str(tmp_path / "scene")
    config["radio"].update(num_subcarriers=80, num_bs_antennas=8)
    config["music"].update(angle_step_deg=2., delay_step_s=2e-9,
                            spatial_subarray_size=6, frequency_subarray_size=12)
    config["music"]["spectrum_sampling"].update(samples_per_peak=8, local_grid_points_per_axis=9)
    scene_files = prepare_scene(config)
    generation_file = tmp_path / "generation.yaml"
    generation_file.write_text(yaml.safe_dump(config))
    spec_file = tmp_path / "experiment.yaml"
    spec_file.write_text(yaml.safe_dump(dict(generation_config=str(generation_file),
        scene_json=scene_files["scene_json"], sampling_bounds_m=[10., 16., 3., 6.],
        ue_count=2, noise_repeats=1, random_seed=20260908,
        wall_clearance_m=.5, bs_clearance_m=2.)))
    return spec_file


def test_spawned_ue_runs_preserve_inputs_results_and_resume(small_experiment, tmp_path):
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    prepare_experiment(small_experiment, serial)
    prepare_experiment(small_experiment, parallel)
    original_plan = file_sha256(parallel / "experiment_plan.json")
    run_experiment(serial)
    run_experiment(parallel, workers=2, cpu_threads=1)
    for ue in ("UE001", "UE002"):
        first, second = (root / ue / "repeat_000" for root in (serial, parallel))
        status = read_json(second / "attempt.json")
        assert status["status"] == read_json(first / "attempt.json")["status"] == "success"
        assert status["worker"]["pid"] != os.getpid()
        assert status["worker"]["thread_environment"]["OPENBLAS_NUM_THREADS"] == "1"
        assert status["worker"]["thread_limit_method"] == "environment_before_python_start"
        assert read_json(first / "attempt.json")["worker"]["thread_limit_method"] == "existing_process_not_reconfigured"
        assert "/workers/worker_" in status["worker"]["cwd"]
        with np.load(first / "data/online/measurement.npz") as a, np.load(second / "data/online/measurement.npz") as b:
            np.testing.assert_array_equal(a["csi_observed"], b["csi_observed"])
        a, b = (read_json(root / "evaluation/metrics.json") for root in (first, second))
        np.testing.assert_allclose(a["localization_error_m"], b["localization_error_m"], atol=1e-6, rtol=1e-6)
    terminal_before = {str(path): file_sha256(path) for path in parallel.glob("UE*/repeat_*/attempt.json")}
    run_experiment(parallel, workers=2)
    assert file_sha256(parallel / "experiment_plan.json") == original_plan
    assert terminal_before == {path: file_sha256(path) for path in terminal_before}
    runs = sorted((parallel / "execution_runs").iterdir())
    assert read_json(runs[-1] / "execution_config.json")["requested_points"] == []
    assert read_json(runs[0] / "status.json")["status"] == "completed"


def test_spawn_environment_is_set_before_start_and_restored(monkeypatch):
    options = _execution_options("cuda", "2,5", 9, 4, 32, 2)
    assert options["worker_count"] == 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "12")
    with _spawn_environment(options, "5"):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
        assert os.environ["TBC_REQUIRE_CUDA"] == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "12"


@pytest.mark.parametrize("ids", ("0,0", "0,00", "", "-1", "0,abc"))
def test_gpu_ids_reject_invalid_or_duplicate_devices(ids):
    with pytest.raises(ValueError, match="GPU"):
        _execution_options("cuda", ids, 1, 4, 32, 1)


def test_worker_failure_preserves_terminal_and_partial_artifacts(tmp_path):
    point = dict(ue_id="UE001", noise_seeds=[10, 11, 12])
    success = tmp_path / "UE001/repeat_000/attempt.json"
    write_json(success, dict(status="success", run_id="existing"))
    original = file_sha256(success)
    partial = tmp_path / "UE001/repeat_001/localization/partial.txt"
    partial.parent.mkdir(parents=True)
    partial.write_text("retain evidence")
    _fail_point(tmp_path, point, "worker died")
    assert file_sha256(success) == original
    assert partial.read_text() == "retain evidence"
    assert read_json(tmp_path / "UE001/repeat_001/attempt.json")["status"] == "worker_failed"
    assert read_json(tmp_path / "UE001/repeat_002/attempt.json")["noise_seed"] == 12


def _crashing_worker(task_queue, event_queue, output, run_root, options, worker_id, device):
    # 使用真实 spawn 和真实异常退出，验证父进程不永远等待丢失的结果。
    event_queue.put(dict(kind="ready", worker=dict(worker_id=worker_id)))
    task_queue.get()
    os._exit(17)


def test_dead_worker_is_detected_and_unfinished_points_are_recorded(tmp_path, monkeypatch):
    import time_bias_localization.experiment as experiment
    monkeypatch.setattr(experiment, "_experiment_worker", _crashing_worker)
    points = [dict(ue_id=f"UE{i:03d}", noise_seeds=[i]) for i in (1, 2)]
    options = _execution_options("numpy", "0", 1, 4, 32, 1)
    run_root = tmp_path / "execution_runs/test"
    run_root.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="异常退出"):
        _run_parallel(tmp_path, run_root, points, options)
    for point in points:
        assert read_json(tmp_path / point["ue_id"] / "repeat_000/attempt.json")["status"] == "worker_failed"
        assert read_json(run_root / "tasks" / f"{point['ue_id']}.json")["status"] == "worker_failed"
    assert read_json(run_root / "workers/worker_000/exit.json")["exitcode"] == 17


def test_unavailable_gpu_fails_before_publishing_ue_tasks(small_experiment, tmp_path):
    output = tmp_path / "unavailable_gpu"
    prepare_experiment(small_experiment, output)
    original_plan = file_sha256(output / "experiment_plan.json")
    # 不存在的设备编号在有 GPU 和无 GPU 的测试机上都必须明确失败。
    with pytest.raises(RuntimeError, match="启动失败"):
        run_experiment(output, compute_backend="cuda", gpu_ids="999999")
    assert file_sha256(output / "experiment_plan.json") == original_plan
    assert not list(output.glob("UE*/repeat_*/attempt.json"))
    run_root = next((output / "execution_runs").iterdir())
    assert read_json(run_root / "status.json")["status"] == "failed"
    assert all(read_json(path)["status"] == "cancelled" for path in (run_root / "tasks").glob("*.json"))


def test_keyboard_interrupt_closes_task_status_without_fabricating_attempts(small_experiment, tmp_path, monkeypatch):
    import time_bias_localization.experiment as experiment
    output = tmp_path / "interrupted"
    prepare_experiment(small_experiment, output)

    def interrupt_run(output, run_root, points, options):
        write_json(run_root / "tasks/UE001.json", dict(ue_id="UE001", status="running"))
        partial = output / "UE001/repeat_000/partial.txt"
        partial.parent.mkdir(parents=True)
        partial.write_text("intermediate output")
        raise KeyboardInterrupt()

    monkeypatch.setattr(experiment, "_run_parallel", interrupt_run)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(output, workers=2)
    run_root = next((output / "execution_runs").iterdir())
    assert read_json(run_root / "tasks/UE001.json")["status"] == "interrupted"
    assert read_json(run_root / "tasks/UE002.json")["status"] == "cancelled"
    assert not list(output.glob("UE*/repeat_*/attempt.json"))
    assert (output / "UE001/repeat_000/partial.txt").read_text() == "intermediate output"
