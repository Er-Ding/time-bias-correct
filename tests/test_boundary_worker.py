"""在线进程的输入边界、超时记录和失败延迟不能改变实验分母。"""

from copy import deepcopy
import json
from pathlib import Path
import os
import signal
import time

import numpy as np
import pytest

from time_bias_localization import boundary_experiment as experiment
from time_bias_localization.config import DEFAULT_CONFIG


def test_job_excludes_ground_truth_noise_level_and_generation_parameters(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    observation = {"scene_json": "public_scene.json", "online_npz": "observed.npz",
                   "generation_manifest": "manifest.json", "truth_npz": "secret_truth.npz",
                   "true_position": [91, 23], "snr_db": 5.0}
    job = experiment.make_job(config, observation, "single", 17, tmp_path, True)
    assert set(job) == {"config", "scene_json", "online_input", "generation_manifest",
                        "output_root", "snapshot_path"}
    assert "simulation" not in job["config"]
    assert "snr_db" not in job["config"]["radio"]
    assert "secret_truth.npz" not in json.dumps(job)
    assert job["config"]["localization"]["diffraction_representative_policy"] == "single"
    assert job["config"]["project"]["random_seed"] == 17


def test_postprocessing_read_failure_preserves_success_and_serializable_payload(tmp_path, monkeypatch):
    import time_bias_localization.pipeline as pipeline

    class Connection:
        def __init__(self):
            self.jobs = iter([{"config": {}, "scene_json": "scene", "online_input": "observed",
                               "generation_manifest": "manifest", "output_root": str(tmp_path)}, None])
            self.sent = []

        def recv(self):
            return next(self.jobs)

        def send(self, value):
            self.sent.append(value)

    monkeypatch.setattr(pipeline, "localize", lambda *args, **kwargs: {
        "mu_m": np.asarray([1., 2.]), "clock_bias_s": 3e-9,
        "diagnostics": {"joint_covariance": np.eye(3)},
        "forward_check": {"all_selected_paths_valid": True}})
    connection = Connection()
    experiment.worker_loop(connection)
    payload = connection.sent[-1]
    assert payload["status"] == "success"
    assert payload["position_m"] == [1., 2.]
    assert "FileNotFoundError" in payload["diagnostic_read_error"]
    assert payload["diagnostics"]["joint_covariance"] == np.eye(3).tolist()
    json.dumps(payload, allow_nan=False)


def _truth_and_point(tmp_path):
    truth_path = tmp_path / "truth.npz"
    np.savez(truth_path, ue_position_m=[1., 2.], clock_bias_s=np.asarray(0.0))
    observation = {"truth_npz": str(truth_path), "truth_sha256": experiment.file_sha256(truth_path), "input_sha256": "public_hash"}
    point = {"cohort": "pilot", "ue_id": "PILOT_0001", "noise_seeds": [1], "mc_seeds": [2],
             "channel_category": "diffraction_only", "has_diffraction": True}
    return observation, point


def test_failed_request_has_primary_online_latency_and_event_provenance(tmp_path):
    observation, point = _truth_and_point(tmp_path)
    event = {"name": "T12_solver", "status": "failed", "elapsed_s": 4., "exclusive_s": 4.,
             "start_s": 2., "end_s": 6., "event_id": 9, "parent_event_id": None}
    payload = {"status": "localization_failed", "processing_seconds": 9.,
               "timings": {"marks": {"csi_map_ready": 1., "online_failed": 6.},
                           "mark_data": {"online_failed": {"failed_step": "07_joint_solution"}},
                           "events": [event], "stages": [{"name": "should_not_use_aggregate"}]}}
    record = experiment.trial_record(point, 0, "coverage", payload, observation, tmp_path)
    assert record["position_error_m"] is None
    assert record["localization_seconds"] == 5.
    assert record["checked_seconds"] == record["failed_online_seconds"] == 5.
    assert record["stage_timings"] == [event]
    assert record["failed_step"] == "07_joint_solution"


def test_hard_timeout_uses_shared_clock_not_stale_snapshot_elapsed(tmp_path):
    observation, point = _truth_and_point(tmp_path)
    payload = {"status": "timeout", "processing_seconds": 11., "timeout_at_perf_counter_s": 110.,
               "timing_snapshot_incomplete": True,
               "timings": {"started_perf_counter_s": 100., "elapsed_s": 3.,
                           "marks": {"csi_map_ready": 2.}, "events": [
                               {"name": "T12_solver", "status": "running", "elapsed_s": None,
                                "exclusive_s": None, "start_s": 3., "end_s": None}]}}
    record = experiment.trial_record(point, 0, "single", payload, observation, tmp_path)
    assert record["localization_seconds"] == record["checked_seconds"] == 8.
    assert record["stage_timings"][0]["elapsed_s"] is None
    assert record["stage_timings"][0]["status"] == "running"


def test_geometry_check_failure_keeps_position_error_and_position_latency(tmp_path):
    observation, point = _truth_and_point(tmp_path)
    payload = {"status": "localization_failed", "processing_seconds": 9.,
               "timings": {"marks": {"csi_map_ready": 1., "position_available": 4., "online_failed": 6.},
                           "mark_data": {"position_available": {"mu_m": [4., 6.], "clock_bias_s": 2e-9}}}}
    record = experiment.trial_record(point, 0, "single", payload, observation, tmp_path)
    assert record["position_error_m"] == 5.
    assert record["localization_seconds"] == 3.
    assert record["checked_seconds"] == 5.


@pytest.mark.parametrize("failure_location", ["send", "recv"])
def test_dead_worker_records_current_failure_and_closes(failure_location):
    class Connection:
        def send(self, value):
            if failure_location == "send":
                raise BrokenPipeError("gone")

        def poll(self, timeout):
            return True

        def recv(self):
            raise EOFError("gone")

    worker = object.__new__(experiment.PersistentWorker)
    worker.connection = Connection()
    closed = []
    worker.close = lambda: closed.append(True)
    result = worker.run({}, 1.)
    assert result["status"] == "localization_failed"
    assert result["processing_seconds"] >= 0
    assert closed == [True]


def test_nested_numpy_metadata_can_be_saved(tmp_path):
    path = tmp_path / "result.json"
    experiment.write_json(path, {"covariance": np.eye(2), "count": np.int64(3)})
    assert experiment.read_json(path) == {"covariance": [[1., 0.], [0., 1.]], "count": 3}


def _interrupted_process(connection):
    """真实 spawn 子进程：写下现场后退出，或故意忽略 TERM 检查硬回收。"""
    connection.send({"ready": True, "pid": os.getpid()})
    job = connection.recv()
    experiment.write_json(Path(job["snapshot_path"]), {
        "marks": {"csi_map_ready": .01}, "events": [{"name": "T08_reverse_rt", "status": "running"}],
        "mark_data": {"initial_candidates_available": {"count": 27}}})
    if job.get("crash"):
        os._exit(9)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(.05)


@pytest.mark.parametrize("crash", [False, True])
def test_real_process_interruption_reaps_child_and_preserves_snapshot(tmp_path, monkeypatch, crash):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    worker = experiment.PersistentWorker(target=_interrupted_process, startup_timeout_s=15., shutdown_grace_s=.1)
    try:
        payload = worker.run({"snapshot_path": str(tmp_path / "timing.json"), "crash": crash}, .4)
        assert payload["status"] == ("localization_failed" if crash else "timeout")
        assert payload["timings"]["mark_data"]["initial_candidates_available"]["count"] == 27
        assert payload["timing_snapshot_incomplete"] is True
        assert not worker.process.is_alive()
        assert worker.process.exitcode == (9 if crash else -signal.SIGKILL)
        assert payload["processing_seconds"] < 5.
        assert payload["worker_shutdown_seconds"] < 5.
        assert payload["failure_at_perf_counter_s"] > 0.
    finally:
        worker.close()  # 反复关闭不会误操作进程或抛出异常。


def test_failed_request_keeps_candidate_counts_from_timing_snapshot(tmp_path):
    observation, point = _truth_and_point(tmp_path)
    payload = {"status": "timeout", "timings": {"mark_data": {
        "initial_candidates_available": {"count": 123, "diagnostics": {"branches": 47}},
        "representatives_available": {"count": 17, "diffraction_count": 12,
                                        "diagnostics": {"cluster_count": 8}}},
        "events": [{"name": "T12_solver", "status": "running"}]}}
    record = experiment.trial_record(point, 0, "single", payload, observation, tmp_path)
    assert record["initial_count"] == 123
    assert record["initial_candidate_diagnostics"] == {"branches": 47}
    assert record["representative_count"] == 17
    assert record["cluster_count"] == 8
    assert record["diffraction_representative_count"] == 12
    assert record["interrupted_stages"] == ["T12_solver"]


def _mock_benchmark(tmp_path, monkeypatch, *, fail_warmup=False, fail_request=False, fail_startup=False, warmups=1):
    points = []
    for index in range(2):
        directory = tmp_path / f"input_{index}"
        directory.mkdir()
        observation, point = _truth_and_point(directory)
        public = directory / "public"
        public.write_text("same observed input for both representative policies")
        for name, digest in (("online_npz", "input_sha256"), ("scene_json", "scene_sha256"),
                             ("generation_manifest", "manifest_sha256")):
            observation[name] = str(public)
            observation[digest] = experiment.file_sha256(public)
        point.update(ue_id=f"PILOT_{index:04d}", observations=[observation])
        points.append(point)
    experiment.write_json(tmp_path / "pilot" / "plan.json", {"points": points, "expected_requests": 4})
    calls, created = [], []
    state = {"warm_failed": False, "request_failed": False, "startup_failed": False}

    class FakeWorker:
        def __init__(self):
            if fail_startup and not state["startup_failed"]:
                state["startup_failed"] = True
                raise RuntimeError("test worker startup failure")
            self.alive = True
            self.process = self
            self.ready = {"ready": True, "pid": 100 + len(created)}
            self.startup_seconds = .01
            created.append(self)

        def is_alive(self):
            return self.alive

        def close(self):
            self.alive = False

        def run(self, job, timeout_s):
            is_warmup = Path(job["output_root"]).name.startswith("warmup_")
            strategy = job["config"]["localization"]["diffraction_representative_policy"]
            calls.append((self.ready["pid"], strategy, is_warmup))
            if is_warmup and fail_warmup and not state["warm_failed"]:
                state["warm_failed"] = True
                self.alive = False
                return {"status": "timeout", "processing_seconds": timeout_s, "timings": {}}
            if not is_warmup and fail_request and not state["request_failed"]:
                state["request_failed"] = True
                self.alive = False
                return {"status": "timeout", "processing_seconds": timeout_s, "timings": {}}
            return {"status": "success", "position_m": [1., 2.], "clock_bias_s": 0.,
                    "processing_seconds": .1, "timings": {}}

    monkeypatch.setattr(experiment, "PersistentWorker", FakeWorker)
    monkeypatch.setattr(experiment, "update_report", lambda *args: {})
    settings = {"noise_repeats": 1, "random_seed": 31, "warmup_per_strategy": warmups,
                "trial_timeout_s": .3, "timing_snapshots": True}
    experiment.benchmark_cohort(tmp_path, "pilot", settings, deepcopy(DEFAULT_CONFIG))
    return experiment.rows(tmp_path / "pilot" / "trials.jsonl"), calls, created


def test_failed_warmup_continues_frozen_plan_without_repeating_warmups(tmp_path, monkeypatch):
    records, calls, workers = _mock_benchmark(tmp_path, monkeypatch, fail_warmup=True, warmups=5)
    assert len(records) == 4
    assert sum(warm for _, _, warm in calls) == 1
    assert all(record["status"] == "success" for record in records)
    assert all(record["warmup_state"] == "disabled_after_failure" for record in records)
    assert all(record["completed_warmups"] == 0 for record in records)
    assert [record["execution_state"] for record in records] == ["cold", "cold", "reused", "reused"]
    assert all(record["pair_cache_comparable"] for record in records)
    assert len(workers) == 4 and all(not worker.alive for worker in workers)
    attempts = list((tmp_path / "pilot" / "benchmark_attempts").iterdir())
    assert experiment.rows(attempts[0] / "warmups.jsonl")[0]["status"] == "timeout"
    assert experiment.read_json(tmp_path / "pilot" / "cohort_completion.json")["recorded_requests"] == 4


def test_request_timeout_keeps_pair_and_restarts_both_strategies_cold(tmp_path, monkeypatch):
    records, calls, workers = _mock_benchmark(tmp_path, monkeypatch, fail_request=True, warmups=0)
    assert len(records) == 4 and len(calls) == 4
    assert [record["status"] for record in records].count("timeout") == 1
    assert [record["worker_generation"] for record in records] == [1, 1, 2, 2]
    assert all(record["execution_state"] == "cold" for record in records)
    assert all(record["pair_cache_comparable"] for record in records)
    assert len({(row["ue_id"], row["repeat_index"], row["strategy"]) for row in records}) == 4
    assert len(workers) == 4 and all(not worker.alive for worker in workers)


def test_two_strategy_workers_have_separate_equal_warmup_and_request_histories(tmp_path, monkeypatch):
    records, calls, workers = _mock_benchmark(tmp_path, monkeypatch)
    assert len(workers) == 2
    for worker in workers:
        worker_calls = [(strategy, warm) for pid, strategy, warm in calls if pid == worker.ready["pid"]]
        assert len({strategy for strategy, _ in worker_calls}) == 1
        assert [warm for _, warm in worker_calls] == [True, False, False]
    assert [record["execution_state"] for record in records] == ["prewarmed", "prewarmed", "reused", "reused"]
    assert all(record["warmup_state"] == "completed" for record in records)
    assert all(record["pair_cache_comparable"] for record in records)


def test_worker_startup_failure_is_recorded_and_remaining_pairs_continue(tmp_path, monkeypatch):
    records, calls, workers = _mock_benchmark(tmp_path, monkeypatch, fail_startup=True)
    assert len(records) == 4
    failed = [record for record in records if record["status"] == "localization_failed"]
    assert len(failed) == 1 and "startup_failed" in failed[0]["error"]
    assert failed[0]["execution_state"] == "unavailable"
    assert failed[0]["processing_seconds"] is None
    assert len(calls) == 3 and not any(warm for _, _, warm in calls)
    assert all(record["pair_cache_comparable"] is False for record in records[:2])
    assert all(record["pair_cache_comparable"] is True for record in records[2:])
    assert records[2]["execution_state"] == records[3]["execution_state"] == "cold"
    assert all(record["warmup_state"] != "pending" for record in records)
    assert len(workers) == 3 and all(not worker.alive for worker in workers)
