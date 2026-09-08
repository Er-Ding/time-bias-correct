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
