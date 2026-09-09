"""对照工具必须能检出峰和路径变化，不能只看最终位置是否接近。"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_gpu_localization.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_gpu_localization", _SCRIPT)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def _make_result(root: Path) -> None:
    folder = root / "localization"
    folder.mkdir(parents=True)
    candidate = {
        "observation_id": "path_0", "candidate_id": "path_0:wall:cluster_0",
        "metadata": {"topology_id": "wall", "reflection_wall_ids": ["wall"],
                     "raw_count": 1, "source_sample_ids": ["path_0:nominal"]},
    }
    files = {
        "localization_result.json": {
            "mu_m": [2.0, 3.0], "clock_bias_s": 2e-8,
            "central_solution": {"mu_m": [2.0, 3.0], "clock_bias_s": 2e-8},
            "central_selected_candidates": {"path_0": candidate},
        },
        "music_peaks.json": {
            "nominal": [{"aoa_index": 1, "delay_index": 2, "aoa_rad": 0.1,
                         "delay_s": 1e-8, "spectrum_value": 4.0}],
            "associations": [{"nominal_to_sample_index": {"0": 0},
                              "unmatched_sample_indices": [], "missed_nominal_indices": [],
                              "valid_sample_indices": [0]}],
            "perturbed_observation_samples": [[{"observation_id": "path_0", "delay_s": 1e-8}]],
        },
        "raw_reverse_candidates.json": [{"observation_id": "path_0", "sample_id": "path_0:nominal",
                                         "topology_id": "wall", "reflection_wall_ids": ["wall"]}],
        "clustered_candidates.json": [candidate],
        "bootstrap_diagnostics.json": [{"repetition": 0, "solved": True,
                                        "selected_topologies": {"path_0": "wall"}}],
    }
    for name, value in files.items():
        (folder / name).write_text(json.dumps(value))
    np.savez(folder / "music_spectrum.npz", spectrum=np.array([[1.0, 4.0], [2.0, 1.0]]))


@pytest.fixture
def roots(tmp_path):
    cpu_root, gpu_root = tmp_path / "cpu", tmp_path / "cuda"
    _make_result(cpu_root)
    _make_result(gpu_root)
    return cpu_root, gpu_root


def _compare(roots):
    return benchmark.compare_localizations(*roots, position_atol_m=1e-5, bias_atol_ns=1e-4)


def test_same_steps_and_position_pass(roots):
    assert _compare(roots)["passed"]


@pytest.mark.parametrize("filename, mutate, expected_check", [
    ("music_peaks.json", lambda value: value["nominal"][0].update(aoa_index=2), "nominal_peak_indices"),
    ("clustered_candidates.json", lambda value: value[0]["metadata"].update(reflection_wall_ids=["other"]), "first_clusters"),
    ("bootstrap_diagnostics.json", lambda value: value[0].update(solved=False), "perturbations"),
    ("localization_result.json", lambda value: value.update(mu_m=[2.1, 3.0]), "position_within_tolerance"),
])
def test_changed_steps_fail_even_if_other_values_match(roots, filename, mutate, expected_check):
    path = roots[1] / "localization" / filename
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))
    comparison = _compare(roots)
    assert not comparison["passed"]
    assert not comparison["checks"][expected_check]


def test_existing_output_is_never_reused(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        benchmark.main(["--input-root", str(tmp_path / "unused"), "--output-root", str(existing)])


def test_nan_tolerance_rejected_before_creating_output(tmp_path):
    output = tmp_path / "new"
    with pytest.raises(SystemExit):
        benchmark.main(["--input-root", str(tmp_path), "--output-root", str(output),
                        "--position-atol-m", "nan"])
    assert not output.exists()


def _convert_to_spectrum_sampling(root):
    folder = root / "localization"
    peaks = json.loads((folder / "music_peaks.json").read_text())
    peaks.pop("associations")
    peaks.pop("perturbed_observation_samples")
    peaks["workflow"] = "music_spectrum_sampling_v1"
    (folder / "music_peaks.json").write_text(json.dumps(peaks))
    (folder / "bootstrap_diagnostics.json").unlink()
    sampling = {
        "samples": [{"observation_id": "path_0", "sample_id": "path_0:mc_00000",
                     "sampling_kind": "local_spectrum_mc", "cell_aoa_index": 3,
                     "cell_delay_index": 4, "aoa_local_rad": 0.12345,
                     "delay_s": 10.12345e-9, "spectrum_value": 12.0}],
        "diagnostics": {"prepared_music": {"eigendecomposition_count": 1},
                        "added_csi_noise": False},
    }
    (folder / "spectrum_samples.json").write_text(json.dumps(sampling))


def test_new_workflow_comparison_never_requires_bootstrap_artifacts(roots):
    for root in roots:
        _convert_to_spectrum_sampling(root)
    assert _compare(roots)["passed"]
    changed = roots[1] / "localization" / "spectrum_samples.json"
    sampling = json.loads(changed.read_text())
    sampling["samples"][0]["aoa_local_rad"] += 1e-4
    changed.write_text(json.dumps(sampling))
    comparison = _compare(roots)
    assert not comparison["passed"]
    assert not comparison["checks"]["sample_aoa_local_rad_within_tolerance"]


def _convert_to_point_clustering(root):
    _convert_to_spectrum_sampling(root)
    folder = root / "localization"
    peaks = json.loads((folder / "music_peaks.json").read_text())
    peaks["workflow"] = "music_point_clustering_v2"
    (folder / "music_peaks.json").write_text(json.dumps(peaks))
    point = {"observation_id": "path_0", "sample_id": "path_0:nominal",
             "topology_id": "wall", "reflection_wall_ids": ["wall"],
             "position_m": [3., 4.], "reference_bias_s": 0.}
    representative = {"candidate_id": "path_0:wall:cluster_0", "point": point,
                      "members": [point], "metadata": {}}
    trajectory = json.loads((folder / "clustered_candidates.json").read_text())[0]
    trajectory.update(anchor_m=[3., 4.], direction=[1., 0.], beta_interval_m=[-2., 5.])
    files = {
        "initial_candidates.json": {"reference_bias_s": 0., "points": [point], "rejected_samples": []},
        "representative_points.json": {"reference_bias_s": 0., "representatives": [representative]},
        "representative_trajectories.json": [trajectory],
    }
    for name, value in files.items():
        (folder / name).write_text(json.dumps(value))
    (folder / "raw_reverse_candidates.json").unlink()
    (folder / "clustered_candidates.json").unlink()


@pytest.mark.parametrize("filename, mutate, expected_check", [
    ("initial_candidates.json", lambda data: data["points"][0].update(position_m=[4., 4.]),
     "initial_points_coordinates_within_tolerance"),
    ("representative_points.json", lambda data: data["representatives"][0]["point"].update(position_m=[4., 4.]),
     "representative_points_coordinates_within_tolerance"),
    ("representative_trajectories.json", lambda data: data[0].update(beta_interval_m=[-3., 5.]),
     "representative_trajectory_beta_interval_m_within_tolerance"),
    ("initial_candidates.json", lambda data: data["rejected_samples"].append({"sample_id": "missing", "reason": "outside"}),
     "initial_rejections"),
])
def test_point_workflow_checks_intermediates_even_with_same_final_solution(roots, filename, mutate, expected_check):
    for root in roots:
        _convert_to_point_clustering(root)
    assert _compare(roots)["passed"]
    path = roots[1] / "localization" / filename
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))
    comparison = _compare(roots)
    assert not comparison["passed"]
    assert not comparison["checks"][expected_check]


def _convert_to_fine_dbscan(root):
    from copy import deepcopy

    _convert_to_point_clustering(root)
    folder = root / "localization"
    peaks = json.loads((folder / "music_peaks.json").read_text())
    peaks.update(workflow="music_fine_spectrum_dbscan_v3", nominal_source_indices=[0],
                 coarse=deepcopy(peaks["nominal"]))
    (folder / "music_peaks.json").write_text(json.dumps(peaks))
    sampling = json.loads((folder / "spectrum_samples.json").read_text())
    sampling["diagnostics"].update(suppressed_refined_peaks=[], peak_and_proposal_share_fine_spectrum=True,
                                    exact_sample_spectrum_used_for_proposal=False)
    spectrum = np.array([[1., 2., 1.], [2., 3., 4.], [1., 2., 1.]])
    corners = (spectrum[:-1, :-1] + spectrum[1:, :-1] + spectrum[:-1, 1:] + spectrum[1:, 1:]) / 4
    sampling["regions"] = [{
        "observation_id": "path_0", "source_peak_index": 0,
        "refined_aoa_grid_index": 1, "refined_delay_grid_index": 2,
        "aoa_grid_rad": [.09, .1, .11], "delay_grid_s": [8e-9, 9e-9, 10e-9],
        "spectrum": spectrum.tolist(), "cell_spectrum": corners.tolist(),
        "cell_probabilities": (corners / corners.sum()).tolist(),
        "peak_on_window_edge": True, "refined_peak_search_boundary_axes": [],
        "unresolved_window_peak": True,
    }]
    (folder / "spectrum_samples.json").write_text(json.dumps(sampling))
    representatives = json.loads((folder / "representative_points.json").read_text())
    representative = representatives["representatives"][0]
    point = representative["point"]
    representative["metadata"].update(core_sample_ids=[point["sample_id"]], border_sample_ids=[])
    noise = {**point, "sample_id": "path_0:mc_99999", "position_m": [30., 40.]}
    representatives["noise_points"] = [noise]
    representatives["memberships"] = [
        {**point, "candidate_id": representative["candidate_id"], "role": "core",
         "neighbor_count": 4, "is_representative": True},
        {**noise, "candidate_id": None, "role": "noise", "neighbor_count": 1, "is_representative": False},
    ]
    (folder / "representative_points.json").write_text(json.dumps(representatives))
    initial = json.loads((folder / "initial_candidates.json").read_text())
    initial["points"].append(noise)
    (folder / "initial_candidates.json").write_text(json.dumps(initial))


@pytest.mark.parametrize("filename, mutate, expected_check", [
    ("music_peaks.json", lambda data: data["nominal"][0].update(aoa_rad=.15),
     "fine_peak_aoa_rad_within_tolerance"),
    ("music_peaks.json", lambda data: data["nominal"][0].update(delay_s=12e-9),
     "fine_peak_delay_s_within_tolerance"),
    ("music_peaks.json", lambda data: data.update(nominal_source_indices=[2]), "nominal_source_indices"),
    ("spectrum_samples.json", lambda data: data["regions"][0]["spectrum"][0].__setitem__(0, 1.5),
     "fine_region_spectrum_within_tolerance"),
    ("spectrum_samples.json", lambda data: data["regions"][0].update(cell_probabilities=[[.4, .1], [.4, .1]]),
     "fine_region_cell_probabilities_within_tolerance"),
    ("spectrum_samples.json", lambda data: data["regions"][0]["aoa_grid_rad"].__setitem__(0, .091),
     "fine_region_aoa_grid_rad_within_tolerance"),
    ("representative_points.json", lambda data: data["noise_points"].clear(), "dbscan_noise_points"),
    ("representative_points.json", lambda data: data["noise_points"][0].update(position_m=[30.5, 40.]),
     "dbscan_noise_points_coordinates_within_tolerance"),
    ("representative_points.json", lambda data: data["memberships"][0].update(role="border"),
     "dbscan_membership_roles"),
    ("representative_points.json", lambda data: data["memberships"][1].update(candidate_id="other"),
     "dbscan_membership_roles"),
    ("representative_points.json", lambda data: data["representatives"][0]["metadata"]["core_sample_ids"].clear(),
     "dbscan_core_border_members"),
])
def test_fine_dbscan_comparison_detects_changed_intermediates_with_unchanged_solution(
    roots, filename, mutate, expected_check,
):
    for root in roots:
        _convert_to_fine_dbscan(root)
    assert _compare(roots)["passed"]
    path = roots[1] / "localization" / filename
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))
    comparison = _compare(roots)
    assert not comparison["passed"]
    assert not comparison["checks"][expected_check]
    assert comparison["checks"]["position_within_tolerance"]
    assert comparison["checks"]["bias_within_tolerance"]


def test_kernel_uses_one_observation_without_legacy_noise_settings(tmp_path, monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace
    from time_bias_localization import compute, spectrum_sampling
    from time_bias_localization.config import DEFAULT_CONFIG
    from time_bias_localization.signal import synthesize_ula_csi

    config = deepcopy(DEFAULT_CONFIG)
    config["music"].update({
        "angle_min_deg": -60, "angle_max_deg": 60, "angle_step_deg": 3,
        "delay_min_s": 0, "delay_max_s": 160e-9, "delay_step_s": 4e-9,
        "num_paths": 2, "signal_subspace_rank": 2, "spatial_subarray_size": 3,
        "frequency_subarray_size": 6,
    })
    frequencies = (np.arange(12) - 5.5) * 2e6
    csi = synthesize_ula_csi(
        path_aoa_rad=np.deg2rad([12.4, -33.7]), path_delay_s=[63.7e-9, 116.3e-9],
        path_coefficients=[1 + 0.4j, 0.8 - 0.7j], num_bs_antennas=5,
        subcarrier_frequencies_hz=frequencies, carrier_frequency_hz=3.5e9,
    )
    rng = np.random.default_rng(43)
    csi += 0.03 * (rng.normal(size=csi.shape) + 1j * rng.normal(size=csi.shape))
    measurement = SimpleNamespace(csi_observed=csi, subcarrier_frequencies_hz=frequencies,
                                  carrier_frequency_hz=3.5e9, antenna_spacing_m=None,
                                  bs_boresight_rad=0)
    original_settings = compute.ComputeSettings
    monkeypatch.setattr(compute, "ComputeSettings", lambda **kwargs: original_settings(
        **{**kwargs, "backend": "numpy"}
    ))
    monkeypatch.setattr(benchmark, "_synchronize", lambda *args: None)
    sampled_arguments = []
    sample_music_spectrum = spectrum_sampling.sample_music_spectrum

    def capture_sampling(*args, **kwargs):
        sampled_arguments.append(kwargs)
        return sample_music_spectrum(*args, **kwargs)

    monkeypatch.setattr(spectrum_sampling, "sample_music_spectrum", capture_sampling)
    args = SimpleNamespace(device_id=0, batch_size=4, angle_chunk_size=16)
    result = benchmark._run_kernel(args, tmp_path, config, measurement)
    assert result["comparison"]["passed"]
    assert result["added_csi_noise"] is False
    assert result["spectrum_count"] == 1
    assert all(run["compute"]["completed_csi"] == 1 for run in result["runs"].values())
    assert all(run["compute"]["eigendecomposition_count"] == 1 for run in result["runs"].values())
    assert len(sampled_arguments) == 2
    assert all(args["minimum_angle_separation_rad"] == np.deg2rad(config["music"]["min_angle_separation_deg"])
               and args["minimum_delay_separation_s"] == config["music"]["min_delay_separation_s"]
               for args in sampled_arguments)
    assert all(result["comparison"]["fine_music_checks"].values())
    assert all("coarse_peaks" in run and "nominal_source_indices" in run for run in result["runs"].values())
