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


def test_batch_result_loader_and_tamper_rejection(generated, tmp_path):
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
    assert len(peak_outputs["nominal_observation_samples"]) == 3
    assert peak_outputs["perturbed_observation_samples"]
    perturbations = read_json(root / "localization/bootstrap_diagnostics.json")
    assert perturbations
    for item in perturbations:
        assert "observation_samples" in item
        if item["solved"]:
            assert item["raw_reverse_candidates"]
            assert item["clustered_candidates"]
            assert item["selected_candidates"]
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
