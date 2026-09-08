from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import time_bias_localization.cli as cli_module
from time_bias_localization.config import (
    DEFAULT_CONFIG,
    load_config,
    load_localization_config,
    localization_config_view,
    validate_config,
    validate_localization_config,
)


@pytest.mark.parametrize("field", [
    "uncertainty_repeats", "uncertainty_noise_scale", "uncertainty_extra_peaks",
    "uncertainty_min_relative_height", "association_max_normalized_distance",
    "false_peak_penalty", "missed_peak_penalty",
])
def test_removed_csi_perturbation_settings_fail_explicitly(field):
    config = deepcopy(DEFAULT_CONFIG)
    config["music"][field] = 1
    with pytest.raises(ValueError, match="CSI 重复加噪流程已移除"):
        validate_config(config)
    with pytest.raises(ValueError, match="CSI 重复加噪流程已移除"):
        localization_config_view(config)


def test_nested_sampling_config_rejects_unknown_fields_and_merges_partial_yaml(tmp_path):
    path = tmp_path / "sampling.yaml"
    path.write_text("radio:\n  bs_position_m: [2, 7]\nmusic:\n  spectrum_sampling:\n    samples_per_peak: 17\n")
    config = load_localization_config(path)
    assert config["music"]["spectrum_sampling"]["samples_per_peak"] == 17
    assert config["music"]["spectrum_sampling"]["uniform_mixture"] == 0.1
    config["music"]["spectrum_sampling"]["true_ue_position"] = [14, 4]
    with pytest.raises(ValueError, match="未定义"):
        validate_localization_config(config)


@pytest.mark.parametrize("field,value", [
    ("samples_per_peak", 0), ("samples_per_peak", True),
    ("local_grid_points_per_axis", 1), ("uniform_mixture", 1.1),
    ("aoa_half_width_grid_steps", 0), ("delay_half_width_grid_steps", float("nan")),
    ("spectrum_power", 0), ("include_nominal", "true"),
])
def test_invalid_spectrum_sampling_parameters(field, value):
    config = deepcopy(DEFAULT_CONFIG)
    config["music"]["spectrum_sampling"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_config(config)


def test_config_rejects_nonfinite_and_invalid_music_ranges() -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["music"]["delay_max_s"] = float("nan")
    with pytest.raises(ValueError, match="delay_max_s"):
        validate_config(config)

    config = deepcopy(DEFAULT_CONFIG)
    config["music"]["angle_max_deg"] = 100.0
    with pytest.raises(ValueError, match="角度范围"):
        validate_config(config)


def test_config_rejects_invalid_subarray_and_noninteger_counts() -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["music"]["frequency_subarray_size"] = config["radio"]["num_subcarriers"] + 1
    with pytest.raises(ValueError, match="frequency_subarray_size"):
        validate_config(config)

    config = deepcopy(DEFAULT_CONFIG)
    config["radio"]["num_bs_antennas"] = 12.5
    with pytest.raises(ValueError, match="num_bs_antennas"):
        validate_config(config)


def test_config_validates_signal_rank_and_front_facing_prior() -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["music"]["signal_subspace_rank"] = (
        config["music"]["spatial_subarray_size"]
        * config["music"]["frequency_subarray_size"]
    )
    with pytest.raises(ValueError, match="signal_subspace_rank"):
        validate_config(config)

    config = deepcopy(DEFAULT_CONFIG)
    config["radio"]["front_facing_only"] = False
    with pytest.raises(ValueError, match="front_facing_only=true"):
        validate_config(config)


def test_old_yaml_without_signal_rank_keeps_previous_num_paths_behavior(
    tmp_path,
) -> None:
    config_path = tmp_path / "old.yaml"
    config_path.write_text("music:\n  num_paths: 4\n", encoding="utf-8")

    config = load_config(config_path)

    assert config["music"]["num_paths"] == 4
    assert config["music"]["signal_subspace_rank"] == 4
    assert config["radio"]["front_facing_only"] is True


def test_localization_loader_rejects_simulation_and_returns_isolated_config(
    tmp_path,
) -> None:
    rejected_path = tmp_path / "contains_truth.yaml"
    rejected_path.write_text(
        "simulation:\n  ue_position_m: [14.0, 4.0]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="禁止包含 simulation"):
        load_localization_config(rejected_path)

    isolated_path = tmp_path / "localization.yaml"
    isolated_path.write_text(
        "radio:\n  bs_position_m: [2.0, 7.0]\n"
        "music:\n  num_paths: 4\noutput:\n  root: outputs/isolated\n",
        encoding="utf-8",
    )
    config = load_localization_config(isolated_path)

    assert "simulation" not in config
    assert config["music"]["num_paths"] == 4
    assert config["music"]["signal_subspace_rank"] == 4
    assert config["output"]["root"] == "outputs/isolated"
    assert config["radio"]["bs_position_m"] == [2.0, 7.0]
    assert "snr_db" not in config["radio"]
    assert config["_config_path"] == str(isolated_path.resolve())


def test_localization_config_view_strips_truth_without_mutating_full_config() -> None:
    full = deepcopy(DEFAULT_CONFIG)
    full["_config_path"] = "/tmp/generation.yaml"
    full["output"]["root"] = "outputs/custom"
    full["scene"]["vertical_path_tolerance_m"] = 0.1
    full["project"]["hidden_true_position_m"] = [14.0, 4.0]

    isolated = localization_config_view(full)

    assert "simulation" not in isolated
    assert isolated["radio"]["bs_position_m"] == [2.0, 7.0]
    assert "snr_db" not in isolated["radio"]
    assert isolated["_config_path"] == full["_config_path"]
    assert isolated["output"] == full["output"]
    assert "vertical_path_tolerance_m" not in isolated["scene"]
    assert "hidden_true_position_m" not in isolated["project"]
    isolated["output"]["root"] = "outputs/changed"
    assert full["output"]["root"] == "outputs/custom"


def test_public_localization_validator_rejects_unknown_and_truth_fields() -> None:
    config = localization_config_view(deepcopy(DEFAULT_CONFIG))
    config["hidden_truth"] = {"ue_position_m": [14.0, 4.0]}
    with pytest.raises(ValueError, match="不支持的顶层字段.*hidden_truth"):
        validate_localization_config(config)

    config = localization_config_view(deepcopy(DEFAULT_CONFIG))
    config["project"]["hidden_true_clock_bias_s"] = 25.0e-9
    with pytest.raises(
        ValueError, match="project 包含未定义字段.*hidden_true_clock_bias_s"
    ):
        validate_localization_config(config)

    config = localization_config_view(deepcopy(DEFAULT_CONFIG))
    config["simulation"] = {"ue_position_m": [14.0, 4.0]}
    with pytest.raises(ValueError, match="禁止包含 simulation"):
        validate_localization_config(config)


def test_localization_config_requires_public_bs_position_and_excludes_snr(
    tmp_path,
) -> None:
    missing_bs = tmp_path / "missing_bs.yaml"
    missing_bs.write_text("project:\n  random_seed: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="radio.bs_position_m"):
        load_localization_config(missing_bs)

    leaked_snr = tmp_path / "leaked_snr.yaml"
    leaked_snr.write_text(
        "radio:\n  bs_position_m: [2.0, 7.0]\n  snr_db: 35.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="radio 包含未定义字段.*snr_db"):
        load_localization_config(leaked_snr)


@pytest.mark.parametrize(
    "section_name,field_name",
    [
        ("scene", "source"),
        ("scene", "name"),
        ("output", "root"),
    ],
)
def test_localization_config_rejects_non_string_path_and_scene_labels(
    section_name: str, field_name: str
) -> None:
    config = localization_config_view(deepcopy(DEFAULT_CONFIG))
    config[section_name][field_name] = {
        "ue_position_m": [14.0, 4.0],
        "clock_bias_s": 25.0e-9,
    }

    with pytest.raises(ValueError, match=rf"{section_name}\.{field_name}.*非空字符串"):
        validate_localization_config(config)


@pytest.mark.parametrize("dead_field", ["candidate_angle_samples", "candidate_delay_samples"])
def test_localization_config_rejects_removed_unused_candidate_fields(
    dead_field: str,
) -> None:
    config = localization_config_view(deepcopy(DEFAULT_CONFIG))
    config["localization"][dead_field] = {"ue_position_m": [14.0, 4.0]}

    with pytest.raises(ValueError, match=rf"localization 包含未定义字段.*{dead_field}"):
        validate_localization_config(config)


@pytest.mark.parametrize(
    "section_name",
    ["project", "scene", "radio", "music", "localization", "output"],
)
def test_localization_loader_rejects_unknown_field_inside_each_section(
    tmp_path, section_name: str
) -> None:
    config_path = tmp_path / f"hidden_in_{section_name}.yaml"
    config_path.write_text(
        f"{section_name}:\n  hidden_true_ue_position_m: [14.0, 4.0]\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"{section_name} 包含未定义字段.*hidden_true_ue_position_m",
    ):
        load_localization_config(config_path)


def test_complete_loader_keeps_generation_only_fields(tmp_path) -> None:
    config_path = tmp_path / "generation.yaml"
    config_path.write_text(
        "scene:\n  vertical_path_tolerance_m: 0.1\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["scene"]["vertical_path_tolerance_m"] == 0.1


@pytest.mark.parametrize(
    "filename",
    [
        "offline_demo_localization.yaml",
        "deepmimo_sionna_smoke_localization.yaml",
        "deepmimo_sionna_munich_localization.yaml",
    ],
)
def test_repository_localization_configs_fit_strict_whitelist(filename: str) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / filename

    config = load_localization_config(config_path)

    assert "simulation" not in config


def test_cli_localize_uses_localization_only_loader(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "contains_truth.yaml"
    config_path.write_text(
        "simulation:\n  clock_bias_s: 25.0e-9\n",
        encoding="utf-8",
    )
    called = False

    def fake_localize(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("不应进入定位器")

    monkeypatch.setattr(cli_module, "localize", fake_localize)
    with pytest.raises(ValueError, match="禁止包含 simulation"):
        cli_module.main(
            [
                "localize",
                "--config",
                str(config_path),
                "--scene-json",
                str(tmp_path / "scene.json"),
                "--online-input",
                str(tmp_path / "measurement.npz"),
            ]
        )
    assert called is False


def test_cli_localize_forwards_receipt_and_prints_run_id(
    tmp_path, monkeypatch, capsys
) -> None:
    config = localization_config_view(deepcopy(DEFAULT_CONFIG))
    receipt_path = tmp_path / "receipts" / "run.json"
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli_module, "load_localization_config", lambda path: config)

    def fake_localize(received_config, **kwargs):
        captured["config"] = received_config
        captured.update(kwargs)
        return {
            "mu_m": [1.0, 2.0],
            "sigma_m2": [[1.0, 0.0], [0.0, 1.0]],
            "clock_bias_s": 25e-9,
            "localization_run_id": "receipt-bound-run",
        }

    monkeypatch.setattr(cli_module, "localize", fake_localize)

    assert (
        cli_module.main(
            [
                "localize",
                "--config",
                str(tmp_path / "localization.yaml"),
                "--scene-json",
                str(tmp_path / "scene.json"),
                "--online-input",
                str(tmp_path / "measurement.npz"),
                "--generation-manifest",
                str(tmp_path / "generation_manifest.json"),
                "--run-receipt",
                str(receipt_path),
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["localization_run_id"] == "receipt-bound-run"
    assert captured["run_receipt"] == str(receipt_path)
    assert captured["config"] is config


def test_cli_evaluate_requires_and_forwards_expected_run_id(
    tmp_path, monkeypatch, capsys
) -> None:
    base_arguments = [
        "evaluate",
        "--result-json",
        str(tmp_path / "result.json"),
        "--truth-npz",
        str(tmp_path / "truth.npz"),
        "--output-json",
        str(tmp_path / "metrics.json"),
    ]
    with pytest.raises(SystemExit):
        cli_module.main(base_arguments)
    assert "--expected-run-id" in capsys.readouterr().err

    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"localization_run_id": kwargs["expected_run_id"]}

    monkeypatch.setattr(cli_module, "evaluate", fake_evaluate)
    assert (
        cli_module.main(
            [*base_arguments, "--expected-run-id", "receipt-bound-run"]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["localization_run_id"] == "receipt-bound-run"
    assert captured["expected_run_id"] == "receipt-bound-run"
