from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from time_bias_localization.config import DEFAULT_CONFIG, load_config, validate_localization_config
from time_bias_localization.contracts import validate_generation_manifest_envelope
from time_bias_localization.experiment import make_noise_repeat, prepare_experiment, run_experiment, sample_positions
from time_bias_localization.pipeline import generate_data, prepare_scene
from time_bias_localization.provenance import file_sha256
from time_bias_localization.scene import Scene2D
from time_bias_localization.visualization import load_run, read_json, summarize


def test_batch_sionna_uses_public_munich_bounds_not_synthetic_defaults():
    from time_bias_localization.experiment import batch_localization_config
    generation = load_config(Path(__file__).resolve().parents[1] / "configs/deepmimo_sionna_munich.yaml")
    config = batch_localization_config(generation)
    validate_localization_config(config)
    assert config["scene"]["bounds_m"] == [-100., 160., -80., 180.]
    assert config["scene"]["bounds_m"] == config["scene"]["localization_bounds_m"]
    assert "simulation" not in config
    assert "snr_db" not in config["radio"]


@pytest.fixture
def generated(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    config["output"]["root"] = str(tmp_path / "source")
    scene = prepare_scene(config)
    data = generate_data(config)
    return config, scene, data


def test_noise_repeats_keep_channel_truth_and_online_boundary(generated, tmp_path):
    _, _, data = generated
    roots = [tmp_path / name for name in ("first", "same_seed", "other_seed")]
    manifests = [make_noise_repeat(Path(data["generation_manifest"]), root, seed=seed) for root, seed in zip(roots, (42, 42, 43))]
    for manifest in manifests:
        validate_generation_manifest_envelope(read_json(manifest))
    with np.load(roots[0] / "data/online/measurement.npz") as first, np.load(roots[1] / "data/online/measurement.npz") as same, np.load(roots[2] / "data/online/measurement.npz") as other:
        assert set(first.files) == {"csi_observed", "subcarrier_frequencies_hz", "carrier_frequency_hz", "antenna_spacing_m", "bs_position_m", "bs_boresight_rad"}
        np.testing.assert_array_equal(first["csi_observed"], same["csi_observed"])
        assert not np.array_equal(first["csi_observed"], other["csi_observed"])
    with np.load(roots[0] / "data/truth/ground_truth.npz") as first, np.load(roots[2] / "data/truth/ground_truth.npz") as other:
        for field in first.files:
            np.testing.assert_array_equal(first[field], other[field])
    digest = file_sha256(roots[0] / "data/online/measurement.npz")
    with pytest.raises(FileExistsError):
        make_noise_repeat(Path(data["generation_manifest"]), roots[0], seed=99)
    assert file_sha256(roots[0] / "data/online/measurement.npz") == digest


def test_sampling_reproducible_and_rejects_wall_region(generated):
    _, scene_files, _ = generated
    scene = Scene2D.load(scene_files["scene_json"])
    kwargs = dict(count=30, seed=42, bs=[2., 7.], wall_clearance=.5, bs_clearance=2.)
    first = sample_positions(scene, [10, 16, 3, 6], **kwargs)
    assert first == sample_positions(scene, [10, 16, 3, 6], **kwargs)
    assert np.asarray(first).shape == (30, 2)
    with pytest.raises(ValueError, match="墙段"):
        sample_positions(scene, [0, 5, 1, 6], **kwargs)


def test_failure_denominators_and_signed_bias():
    rows = [dict(ue_id="UE1", status="success", position_error_m=1., bias_error_ns=-2.),
            dict(ue_id="UE1", status="success", position_error_m=3., bias_error_ns=4.),
            dict(ue_id="UE2", status="localization_failed"), dict(ue_id="UE2", status="pending")]
    summary = summarize(rows)
    assert summary["within_1m_fraction_all"] == .25
    assert summary["position_median_m"] == 2.
    assert summary["bias_mean_signed_ns"] == 1.
    assert summary["bias_mae_ns"] == 3.
    assert summary["failed_count"] == summary["pending_count"] == 1
    assert summarize(rows[2:])["position_rmse_m"] is None


def test_batch_result_loader_and_tamper_rejection(generated, tmp_path, monkeypatch):
    config, scene_files, _ = generated
    generation_file = tmp_path / "generation.yaml"
    generation_file.write_text(yaml.safe_dump(config))
    spec_file = tmp_path / "experiment.yaml"
    spec_file.write_text(yaml.safe_dump(dict(generation_config=str(generation_file), scene_json=scene_files["scene_json"],
                         sampling_bounds_m=[10, 16, 3, 6], ue_count=1, noise_repeats=1, random_seed=20260907,
                         wall_clearance_m=.5, bs_clearance_m=2.)))
    output = tmp_path / "experiment"
    prepare_experiment(spec_file, output)
    run_experiment(output)
    root = output / "UE001/repeat_000"
    assert read_json(root / "attempt.json")["status"] == "success"
    run = load_run(root)
    assert len(run["paths"]) == 3
    assert run["metrics"]["localization_error_m"] >= 0
    peak_outputs = read_json(root / "localization/music_peaks.json")
    assert len(peak_outputs["nominal"]) == 3
    assert len(peak_outputs["observation_samples"]) > 3
    samples = read_json(root / "localization/spectrum_samples.json")
    assert samples["samples"] and samples["regions"]
    assert run["result"]["workflow"] == "music_spectrum_sampling_v1"
    assert (root / "localization/forward_check.json").exists()
    assert not (root / "localization/bootstrap_diagnostics.json").exists()
    # 使用定位器实际序列化的产物走完整报告，避免手写数据模型遗漏字段层级。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import time_bias_localization.step_visualization as step_module
    generated_figures = []
    def capture(plt, fig, directory, name):
        generated_figures.append((directory.name, name))
        plt.close(fig)
    monkeypatch.setattr(step_module, "_save", capture)
    report_root = tmp_path / "actual_result_steps"
    step_module.export_steps(plt, run, report_root, read_json(root / "attempt.json"))
    assert all(item["status"] == "available" for item in read_json(report_root / "step_index.json"))
    counts = read_json(report_root / "05_first_clustering/counts.json")
    assert counts["listed_member_count"] == counts["raw_count"] == len(run["raw"])
    assert counts["representative_count"] == len(run["clusters"])
    assert ("03_spectrum_sampling", "local_spectrum_000") in generated_figures
    assert ("05_first_clustering", "members_and_representatives") in generated_figures
    assert ("07_forward_check", "original_peak_residuals") in generated_figures
    status_digest = file_sha256(root / "attempt.json")
    run_experiment(output)
    assert file_sha256(root / "attempt.json") == status_digest
    path = root / "localization/clustered_candidates.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="指纹"):
        load_run(root)


def test_sample_medians_do_not_average_positions():
    from time_bias_localization.step_visualization import per_sample_statistics
    rows = [dict(ue_id="UE1", true_x_m=0, true_y_m=0, status="success", position_error_m=e) for e in (1., 9.)]
    rows += [dict(ue_id="UE2", true_x_m=1, true_y_m=1, status="localization_failed")]
    summary = per_sample_statistics(rows)
    assert summary[0]["median_error_m"] == 5.
    assert summary[0]["p90_error_m"] == 8.2
    assert summary[1]["median_error_m"] is None
    assert summary[1]["failed_count"] == 1


def test_missing_sample_preserves_all_steps(tmp_path):
    from time_bias_localization.step_visualization import export_steps, STEPS
    root = tmp_path / "sample/repeat_000"
    export_steps(None, None, root, dict(status="localization_failed", error="no solution"))
    assert len(read_json(root / "step_index.json")) == len(STEPS)
    for name, _, _ in STEPS:
        assert "no solution" in (root / name / "README.md").read_text()
        assert not list((root / name).glob("*.png"))


@pytest.mark.parametrize("workflow", [None, "music_spectrum_sampling_v1"])
@pytest.mark.parametrize("status", ["localization_failed", "pending"])
def test_missing_sample_uses_recorded_workflow_for_placeholders(tmp_path, workflow, status):
    from time_bias_localization.step_visualization import export_steps, STEPS, LEGACY_STEPS
    root = tmp_path / "steps"
    export_steps(None, None, root, dict(status=status, workflow=workflow, error="saved failure"))
    expected = LEGACY_STEPS if workflow is None else STEPS
    assert [item["step"] for item in read_json(root / "step_index.json")] == [item[0] for item in expected]
    assert all(item["status"] == "not_available" for item in read_json(root / "step_index.json"))
    assert ("旧流程（legacy）" in (root / "README.md").read_text()) == (workflow is None)
    assert (root / "03_spectrum_sampling").exists() == (workflow is not None)
    assert (root / "07_perturbation_solutions").exists() == (workflow is None)


def test_standard_cdf_and_global_quantiles(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import time_bias_localization.step_visualization as module
    rows = [dict(ue_id="UE1", true_x_m=0, true_y_m=0, status="success", position_error_m=e, bias_error_ns=0.) for e in (1., 9.)]
    rows += [dict(ue_id="UE2", true_x_m=1, true_y_m=1, status="localization_failed")]
    figures = {}
    def capture(plt, fig, directory, name):
        figures[name] = fig
    monkeypatch.setattr(module, "_save", capture)
    module.export_summary(plt, rows, summarize(rows), tmp_path)
    x, y = figures["position_error_cdf"].axes[0].lines[0].get_data()
    np.testing.assert_allclose(x, [0, 1, 9])
    np.testing.assert_allclose(y, [0, .5, 1])
    assert read_json(tmp_path / "summary.json")["position_p90_m"] == 8.2
    assert (tmp_path / "per_sample.csv").exists()
    for fig in figures.values():
        plt.close(fig)


def test_spectrum_report_uses_saved_samples_members_and_forward_checks(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import time_bias_localization.step_visualization as module
    from time_bias_localization.visualization import write_json

    source = tmp_path / "source"
    source.mkdir()
    scene = {"bounds_m": [-2., 4., -2., 4.], "walls": []}
    write_json(source / "scene.json", scene)
    write_json(source / "config.json", {})
    np.savez(source / "truth.npz", ue_position_m=[1., 1.])
    np.savez(source / "measurement.npz", csi_observed=np.ones((1, 2, 3), dtype=complex),
             subcarrier_frequencies_hz=[0., 1e6, 2e6])
    np.savez(source / "music.npz", spectrum=np.ones((3, 3)), aoa_grid_rad=[-.2, 0., .2],
             delay_grid_s=[1e-9, 2e-9, 3e-9])
    write_json(source / "peaks.json", {"nominal": [{"aoa_rad": 0., "delay_s": 2e-9, "spectrum_value": 1.}]})
    samples = [dict(observation_id="music_path_00", sample_id=f"s{i}", aoa_local_rad=angle,
                    aoa_global_rad=angle, delay_s=2e-9, spectrum_value=1., sampling_kind=kind)
               for i, (angle, kind) in enumerate([(0., "nominal"), (.03, "local_spectrum_mc")])]
    write_json(source / "samples.json", {"samples": samples, "regions": [{
        "observation_id": "music_path_00", "aoa_grid_rad": [-.1, 0., .1],
        "delay_grid_s": [1e-9, 2e-9, 3e-9], "spectrum": np.ones((3, 3)).tolist()}], "diagnostics": {}})
    raw = [dict(observation_id="music_path_00", sample_id=f"s{i}", topology_id="los",
                anchor_m=[2., 1. + i * .03], direction=[1., 0.], beta_min_m=0., beta_max_m=1.,
                reflection_wall_ids=[]) for i in range(2)]
    cluster = dict(candidate_id="c0", observation_id="music_path_00",
                   anchor_m=[2., 1.], direction=[1., 0.], beta_interval_m=[0., 1.],
                   metadata=dict(topology_id="los", raw_count=2, source_sample_ids=["s0", "s1"], representative_sample_id="s0",
                                 reflection_wall_ids=[]))
    write_json(source / "raw.json", raw)
    write_json(source / "clusters.json", [cluster])
    write_json(source / "forward.json", {"all_selected_paths_valid": True, "paths": [{
        "observation_id": "music_path_00", "candidate_id": "c0", "valid": True,
        "path_nodes_m": [[1., 1.], [0., 0.]], "prediction": {"aoa_global_deg": 45., "predicted_observed_delay_s": 2e-9},
        "original_peak_residuals": {"aoa_error_deg": .1, "delay_error_ns": .2},
        "sample_residuals": {"aoa_error_deg": .2, "delay_error_ns": .3}}]})
    central = {"mu_m": [1., 1.], "sigma_m2": [[.01, 0.], [0., .01]], "clock_bias_s": 0.}
    result = {**central, "workflow": module.WORKFLOW, "central_solution": central,
              "central_selected_candidates": [cluster], "central_residuals_m": [0.], "diagnostics": {}}
    write_json(source / "result.json", result)
    artifacts = {key: str(source / filename) for key, filename in {
        "music_spectrum": "music.npz", "music_peaks": "peaks.json", "spectrum_samples": "samples.json",
        "raw_reverse_candidates": "raw.json", "clustered_candidates": "clusters.json",
        "forward_check": "forward.json", "result": "result.json"}.items()}
    run = dict(scene=scene, bs=[0., 0.], boresight=0., true=[1., 1.], paths=[], result=result,
               artifacts=artifacts, metrics={"localization_error_m": 0.}, sources=[],
               input_paths={"scene_json": str(source / "scene.json"), "ground_truth": str(source / "truth.npz"),
                            "online_measurement": str(source / "measurement.npz")}, config_path=str(source / "config.json"))
    figures = {}
    def capture(plt, fig, directory, name):
        figures[(directory.name, name)] = fig
    monkeypatch.setattr(module, "_save", capture)
    output = tmp_path / "report"
    module.export_steps(plt, run, output, {"status": "success"})
    assert all(item["status"] == "available" for item in read_json(output / "step_index.json"))
    assert not (output / "03_peak_perturbations").exists()
    assert not (output / "07_perturbation_solutions").exists()
    assert read_json(output / "05_first_clustering/counts.json") == {
        "raw_count": 2, "representative_count": 1, "listed_member_count": 2}
    assert "s1" in (output / "05_first_clustering/cluster_members.csv").read_text()
    sample_figure = figures[("03_spectrum_sampling", "local_spectrum_000")]
    offsets = [collection.get_offsets() for collection in sample_figure.axes[0].collections]
    assert any(np.any(np.isclose(offset[:, 1], np.degrees(.03))) for offset in offsets)
    assert ("07_forward_check", "original_peak_residuals") in figures
    assert "original_peak_delay_error_ns" in (output / "07_forward_check/path_checks.csv").read_text()
    for fig in figures.values():
        plt.close(fig)


def test_old_results_are_dispatched_to_legacy_report(tmp_path, monkeypatch):
    import time_bias_localization.step_visualization as module
    called = []
    monkeypatch.setattr(module, "_export_legacy_steps", lambda *args: called.append(args))
    run = {"result": {"mu_m": [0., 0.]}}
    module.export_steps(None, run, tmp_path / "legacy", {"status": "success"})
    assert called[0][1] is run


def test_failed_run_loads_and_exports_completed_steps_with_hashes(generated, tmp_path, monkeypatch):
    from time_bias_localization.provenance import artifact_record, generation_bundle_id
    from time_bias_localization.visualization import load_failed_run, write_json
    import time_bias_localization.step_visualization as module
    _, _, data = generated
    generation_path = Path(data["generation_manifest"])
    generation = read_json(generation_path)
    root = tmp_path / "failed"
    failure = root / "localization_failures/run-failed"
    write_json(failure / "config.json", {})
    write_json(failure / "samples.json", {"samples": [], "regions": [], "diagnostics": {}})
    write_json(failure / "progress.json", {
        "workflow": module.WORKFLOW, "run_id": "run-failed", "completed_steps": ["01_csi_input", "03_spectrum_sampling"],
        "failed_step": "04_reverse_candidates", "error": "no candidates",
        "generation_bundle": {"bundle_id": generation_bundle_id(generation), "manifest": artifact_record(generation_path)},
        "config_snapshot": {"path": str(failure / "config.json"), "file_sha256": file_sha256(failure / "config.json")},
        "inputs": {"scene": generation["artifact_hashes"]["scene_json"],
                   "online_measurement": generation["artifact_hashes"]["online_measurement"]},
        "artifacts": {"spectrum_samples": artifact_record(failure / "samples.json")},
    })
    run = load_failed_run(root, failure / "progress.json")
    assert run["metrics"] == {}
    assert "mu_m" not in run["result"]
    # 截止边界前只导出已保存数据；这里不渲染场景/输入的重复图片。
    monkeypatch.setattr(module, "_export_scene", lambda *args: None)
    monkeypatch.setattr(module, "_export_csi", lambda *args: None)
    output = tmp_path / "failed_report"
    module.export_steps(None, run, output, {"status": "localization_failed", "error": "no candidates"})
    states = {item["step"]: item["status"] for item in read_json(output / "step_index.json")}
    assert states["03_spectrum_sampling"] == "available"
    assert states["04_reverse_candidates"] == states["08_final_evaluation"] == "not_available"
    (failure / "samples.json").write_text("{}")
    with pytest.raises(ValueError, match="指纹"):
        load_failed_run(root, failure / "progress.json")
