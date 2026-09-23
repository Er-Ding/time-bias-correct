"""检查补采口径、独立偏置与噪声、失败保留和真正经过 CSI 的可恢复小流程。"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from time_bias_localization.config import DEFAULT_CONFIG
from time_bias_localization.monte_carlo_experiment import (
    load_settings, prepare_samples, prepare_observations, solve_samples, summarize,
)


def small_config(tmp_path):
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["experiment"] = dict(sample_count=2, workers=2, max_proposals=1000, random_seed=126,
        bias_min_ns=-50., bias_max_ns=50., trial_timeout_s=60., channel_backend="synthetic_fixture", plots=True)
    cfg["scene"].update(max_reflections=2, max_diffractions=1, diffraction_position="last_from_bs")
    cfg["radio"].update(num_subcarriers=64, num_bs_antennas=6, bandwidth_hz=200e6)
    cfg["music"].update(spatial_subarray_size=4, frequency_subarray_size=16, angle_step_deg=4.,
        delay_min_s=-80e-9, delay_max_s=220e-9, delay_step_s=4e-9,
        subspace_selection={"mode": "eigenvalue_threshold", "noise_reference": "median", "threshold_ratio": 6.})
    cfg["localization"].update(solver_method="continuous", require_identifiable_solution=True,
        candidate_bias_mode="full_interval", continuous={"max_hypotheses":128,"max_enumerated_sequences":256,
            "max_starts":4,"max_seed_combinations":128,"max_iterations":8})
    cfg["output"]["root"] = str(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return load_settings(path)


def test_defaults_match_agreed_experiment():
    path = Path(__file__).parents[1] / "configs/monte_carlo_continuous_munich.yaml"
    settings, config = load_settings(path)
    assert settings["sample_count"] == 1000
    assert (settings["bias_min_ns"], settings["bias_max_ns"]) == (-50., 50.)
    assert config["radio"]["snr_db"] == 35.
    assert config["scene"]["diffraction_position"] == "last_from_bs"
    assert config["music"]["delay_min_s"] < 0
    gpu_settings, gpu_config = load_settings(path, compute_backend="cuda")
    assert gpu_settings == settings
    assert gpu_config["compute"]["backend"] == "cuda"
    gpu_config["compute"]["backend"] = config["compute"]["backend"]
    assert gpu_config == config  # 设备覆盖不改变观测、采样或求解参数。
    with pytest.raises(ValueError, match="compute.backend"):
        load_settings(path, compute_backend="typo")


def test_resample_only_illegal_or_zero_paths_and_keep_single_path(tmp_path, monkeypatch):
    from time_bias_localization.boundary_channel import CoverageProbe
    from time_bias_localization.provenance import artifact_record
    import time_bias_localization.boundary_channel as channel
    settings, config = small_config(tmp_path)
    class Provider:
        def __init__(self, config, setup, **kwargs):
            self.setup_root = setup
            setup.mkdir(parents=True)
            self.setup_json = setup / "channel_setup.json"
            self.setup_json.write_text('{}')
            self.calls = 0
        def probe(self, point, seed):
            self.calls += 1
            status = {1:"illegal",2:"no_signal"}.get(self.calls, "covered")
            n = 0 if status != "covered" else 1 if self.calls == 3 else 3
            meta = {"retained_mask":np.ones(n,bool), "interactions":np.array([[0]*n]),
                    "reflection_order":np.zeros(n,int),"diffraction_order":np.zeros(n,int)}
            return CoverageProbe(status,status,point,seed,.01,np.ones((6,64),complex),meta,
                channel_setup_sha256=artifact_record(self.setup_json)["sha256"])
        def close(self):
            pass
    monkeypatch.setattr(channel,"make_boundary_channel",Provider)
    root = tmp_path / "run"
    plan = prepare_samples(root, settings, config)
    assert plan["proposal_count"] == 4
    assert plan["proposal_status_counts"] == {"illegal":1,"no_signal":1,"covered":2}
    assert [s["retained_path_count"] for s in plan["samples"]] == [1,3]
    monkeypatch.setattr(channel,"make_boundary_channel",lambda *a,**k:pytest.fail("已固定采样不得重建 RT"))
    assert prepare_samples(root, settings, config) == plan


def test_independent_common_bias_noise_csi_solve_and_resume(tmp_path, monkeypatch):
    from time_bias_localization.signal import apply_common_delay_bias
    settings, config = small_config(tmp_path)
    root = tmp_path / "run"
    plan = prepare_samples(root, settings, config)
    observations = prepare_observations(root, plan, settings, config)
    assert len({s["clock_bias_ns"] for s in plan["samples"]}) == len(plan["samples"])
    assert all(-50 <= s["clock_bias_ns"] <= 50 for s in plan["samples"])
    for sample in plan["samples"]:
        artifacts = observations[sample["sample_id"]]["artifacts"]
        with np.load(artifacts["truth_npz"]["path"]) as t, np.load(artifacts["online_npz"]["path"]) as o:
            assert float(t["clock_bias_s"]) == sample["clock_bias_ns"] * 1e-9
            noisy = apply_common_delay_bias(t["csi_geometric"], o["subcarrier_frequencies_hz"], float(t["clock_bias_s"]),
                noise_std=float(t["injected_noise_std"]), seed=sample["noise_seed"])
            np.testing.assert_array_equal(noisy, o["csi_observed"])
            assert "ue_position_m" not in o.files and "clock_bias_s" not in o.files
            assert sum(sample["path_type_counts"]) == np.count_nonzero(t["retained_mask"])
    summary = solve_samples(root, plan, observations, settings, config)
    assert summary["all_samples_finished"]
    assert summary["completed_sample_count"] == 2
    assert not summary["status_counts"].get("localization_failed")
    for request in root.glob("samples/*/localization_attempts/*/online_request.json"):
        job = json.loads(request.read_text())
        assert set(job) == {"config","scene_json","online_input","generation_manifest","output_root","snapshot_path"}
        assert "simulation" not in job["config"] and "snr_db" not in job["config"]["radio"]
        assert "_config_path" not in job["config"]
    assert (root / "report/samples.csv").exists()
    assert (root / "report/error_cdf.png").exists()
    import time_bias_localization.monte_carlo_experiment as module
    monkeypatch.setattr(module,"PersistentWorker",lambda *a,**k:pytest.fail("已完成的样本不能重新求解"))
    assert solve_samples(root, plan, observations, settings, config) == summary
    # 模拟子进程结果已提交、父进程还未登记就中断；恢复时不能重新优化。
    first = root / "samples" / plan["samples"][0]["sample_id"]
    row = json.loads((first / "result.json").read_text())
    (first / "result.json").unlink()
    Path(row["solve_record"]["path"]).unlink()
    assert solve_samples(root, plan, observations, settings, config) == summary


def test_failure_denominator_and_six_counts_do_not_drop_single_paths():
    samples = [{"sample_id":"a","retained_path_count":1,"path_type_counts":[0,0,0,1,0,0]},
               {"sample_id":"b","retained_path_count":3,"path_type_counts":[0,1,1,0,1,0]},
               {"sample_id":"c","retained_path_count":2,"path_type_counts":[1,1,0,0,0,0]}]
    plan = {"samples":samples,"proposal_count":5,"proposal_status_counts":{"covered":3,"no_signal":2}}
    results = {"a":{"status":"unlocalizable"},"b":{"status":"success","position_error_m":.2,
              "clock_bias_error_ns":.1,"selected_path_type_counts":[0,1,1,0,0,0]}}
    summary = summarize(plan, results)
    assert summary["status_counts"] == {"unlocalizable":1,"success":1,"pending":1}
    assert summary["accuracy"]["all_samples"]["success_rate"] == 1/3
    assert summary["accuracy"]["at_least_two_rt_paths"]["success_rate"] == 1/2
    assert summary["rt_path_type_counts"] == [1,2,1,1,1,0]
    assert summary["samples_with_at_least_two_paths"] == 2
    assert summary["accuracy"]["single_rt_path"]["position_error_m"]["count"] == 0


def test_rt_error_retries_same_proposal_instead_of_replacing_it(tmp_path, monkeypatch):
    from time_bias_localization.boundary_channel import BoundaryChannel, CoverageProbe
    settings, config = small_config(tmp_path)
    original = BoundaryChannel.probe
    seen = []
    def fail(self, point, seed):
        seen.append((np.asarray(point).tolist(), seed))
        return CoverageProbe("unknown", "rt_error", np.asarray(point), seed, .01, error="temporary failure")
    monkeypatch.setattr(BoundaryChannel,"probe",fail)
    root = tmp_path / "run"
    with pytest.raises(RuntimeError, match="同一提案"):
        prepare_samples(root, settings, config)
    assert not list((root / "sampling/proposals").glob("*.json"))
    assert list((root / "sampling/errors").glob("*.json"))
    def retry(self, point, seed):
        seen.append((np.asarray(point).tolist(), seed))
        return original(self, point, seed)
    monkeypatch.setattr(BoundaryChannel,"probe",retry)
    plan = prepare_samples(root, settings, config)
    assert seen[0] == seen[1]
    assert len(plan["samples"]) == settings["sample_count"]


def test_timeout_remains_timeout_even_if_worker_wrote_late_success(tmp_path, monkeypatch):
    import time_bias_localization.monte_carlo_experiment as module
    settings, config = small_config(tmp_path)
    settings["plots"] = False
    root = tmp_path / "run"
    plan = prepare_samples(root, settings, config)
    observations = prepare_observations(root, plan, settings, config)
    class TimedOutWorker:
        def __init__(self, **kwargs):
            self._closed = False
        def run(self, job, timeout_s):
            module.write_json(Path(job["output_root"]) / "worker_result.json",
                {"status":"success","mu_m":[1.,1.],"clock_bias_s":0.,"processing_seconds":timeout_s+.01})
            self._closed = True
            return {"status":"timeout","processing_seconds":timeout_s}
        def close(self):
            self._closed = True
    monkeypatch.setattr(module,"PersistentWorker",TimedOutWorker)
    summary = solve_samples(root, plan, observations, settings, config)
    assert summary["status_counts"] == {"timeout":2}
    for path in root.glob("samples/*/result.json"):
        row = json.loads(path.read_text())
        assert json.loads(Path(row["solve_record"]["path"]).read_text())["status"] == "timeout"
        assert row["position_error_m"] is None
    first = root / "samples" / plan["samples"][0]["sample_id"] / "result.json"
    first.unlink()
    monkeypatch.setattr(module,"PersistentWorker",lambda *a,**k:pytest.fail("超时结果不能被选择性重跑"))
    assert solve_samples(root, plan, observations, settings, config) == summary
