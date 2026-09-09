from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
import math
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import subprocess
import sys

import numpy as np
import pytest

import time_bias_localization.pipeline as pipeline_module
from time_bias_localization.config import DEFAULT_CONFIG, localization_config_view
from time_bias_localization.data import (
    generate_synthetic_measurement,
    load_online_measurement,
)
from time_bias_localization.pipeline import (
    _distribution_statistics,
    _load_evaluation_truth,
    _write_json,
    _validate_unambiguous_delay_window,
    associate_perturbed_peaks,
    evaluate,
    generate_data,
    localize,
    prepare_scene,
    run_offline_demo,
)
from time_bias_localization.provenance import (
    artifact_record,
    canonical_json_sha256,
    exclusive_output_root_lock,
    file_sha256,
    generation_bundle_id,
)
from time_bias_localization.signal import MusicPeak2D, MusicPeakSamples


def _acquire_lock_in_child(output_root: str, messages) -> None:
    messages.put("waiting")
    with exclusive_output_root_lock(output_root):
        messages.put("acquired")


def _evaluate_in_child(
    result_json: str,
    truth_npz: str,
    output_json: str,
    expected_run_id: str,
    messages,
) -> None:
    messages.put("waiting")
    try:
        evaluate(
            result_json=result_json,
            truth_npz=truth_npz,
            output_json=output_json,
            expected_run_id=expected_run_id,
        )
    except BaseException as error:
        messages.put((type(error).__name__, str(error)))
    else:
        messages.put(("ok", ""))


def _prepare_scene_in_child(config, output_root: str, messages) -> None:
    messages.put("waiting")
    try:
        prepare_scene(config, output_root)
    except BaseException as error:
        messages.put((type(error).__name__, str(error)))
    else:
        messages.put(("ok", ""))


def _generate_data_in_child(
    config, scene_json: str, output_root: str, messages
) -> None:
    messages.put("waiting")
    try:
        generate_data(config, scene_json=scene_json, output_root=output_root)
    except BaseException as error:
        messages.put((type(error).__name__, str(error)))
    else:
        messages.put(("ok", ""))


def _current_run_id(result_path: str | Path) -> str:
    manifest_path = Path(result_path).resolve().parent / "localization_manifest.json"
    return str(json.loads(manifest_path.read_text(encoding="utf-8"))["run_id"])


def _truth_npz_bytes(ue_position_m, clock_bias_s) -> bytes:
    buffer = BytesIO()
    np.savez(
        buffer,
        ue_position_m=np.asarray(ue_position_m),
        clock_bias_s=np.asarray(clock_bias_s),
    )
    return buffer.getvalue()


def test_online_loader_rejects_truth_path(tmp_path) -> None:
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    path = truth_dir / "ground_truth.npz"
    np.savez(path, value=np.asarray(1))
    with pytest.raises(ValueError, match="拒绝读取真值"):
        load_online_measurement(path)


@pytest.mark.parametrize(
    "extra_field",
    ["ue_position_m", "clock_bias_s", "csi_geometric", "path_delays_s"],
)
def test_online_loader_rejects_every_extra_field(tmp_path, extra_field: str) -> None:
    path = tmp_path / f"measurement_with_{extra_field}.npz"
    fields = {
        "csi_observed": np.zeros((1, 2, 8), dtype=np.complex128),
        "subcarrier_frequencies_hz": np.arange(8, dtype=float),
        "carrier_frequency_hz": np.asarray(3.5e9),
        "antenna_spacing_m": np.asarray(0.04),
        "bs_position_m": np.asarray([2.0, 7.0]),
        "bs_boresight_rad": np.asarray(0.0),
        extra_field: np.asarray(0.0),
    }
    np.savez(path, **fields)

    with pytest.raises(ValueError, match=rf"不允许的额外字段.*{extra_field}"):
        load_online_measurement(path)


def test_online_loader_accepts_exact_contract(tmp_path) -> None:
    path = tmp_path / "measurement.npz"
    np.savez(
        path,
        csi_observed=np.zeros((1, 2, 8), dtype=np.complex128),
        subcarrier_frequencies_hz=np.arange(8, dtype=float),
        carrier_frequency_hz=np.asarray(3.5e9),
        antenna_spacing_m=np.asarray(0.04),
        bs_position_m=np.asarray([2.0, 7.0]),
        bs_boresight_rad=np.asarray(0.0),
    )

    measurement = load_online_measurement(path)

    assert measurement.csi_observed.shape == (1, 2, 8)


@pytest.mark.parametrize("occupied_kind", ["file", "dangling_symlink"])
def test_prepare_scene_preflights_every_fixed_target_before_writing(
    tmp_path, occupied_kind: str
) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    root = tmp_path / f"prepare_{occupied_kind}"
    scene_dir = root / "scene"
    scene_dir.mkdir(parents=True)
    occupied_path = scene_dir / "preprocess_manifest.json"
    if occupied_kind == "file":
        occupied_path.write_bytes(b"keep-existing-manifest")
    else:
        os.symlink(scene_dir / "missing-target", occupied_path)

    with pytest.raises(FileExistsError, match="preprocess_manifest.json"):
        prepare_scene(config, root)

    assert os.path.lexists(occupied_path)
    assert not (scene_dir / "scene_2d.json").exists()
    assert not (scene_dir / "scene_bev.png").exists()
    assert not (scene_dir / "scene_occupancy.npy").exists()


@pytest.mark.parametrize("occupied_kind", ["file", "dangling_symlink"])
def test_generate_data_preflights_generation_manifest_before_writing_bundle(
    tmp_path, occupied_kind: str
) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    scene_root = tmp_path / f"source_scene_{occupied_kind}"
    scene = prepare_scene(config, scene_root)
    root = tmp_path / f"generated_{occupied_kind}"
    generation_manifest_path = root / "data" / "generation_manifest.json"
    generation_manifest_path.parent.mkdir(parents=True)
    if occupied_kind == "file":
        generation_manifest_path.write_bytes(b"keep-existing-generation-manifest")
    else:
        os.symlink(root / "missing-generation-manifest", generation_manifest_path)

    with pytest.raises(FileExistsError, match="generation_manifest.json"):
        generate_data(config, scene_json=scene["scene_json"], output_root=root)

    assert os.path.lexists(generation_manifest_path)
    assert not (root / "data" / "online" / "measurement.npz").exists()
    assert not (root / "data" / "truth" / "ground_truth.npz").exists()


def test_public_scene_and_data_entries_refuse_second_publish_without_changes(
    tmp_path,
) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    root = tmp_path / "public_no_overwrite"
    scene = prepare_scene(config, root)
    scene_files_before = {
        path.relative_to(root): path.read_bytes()
        for path in (root / "scene").iterdir()
        if path.is_file()
    }

    with pytest.raises(FileExistsError, match="场景流水线目标已存在"):
        prepare_scene(config, root)
    assert {
        path.relative_to(root): path.read_bytes()
        for path in (root / "scene").iterdir()
        if path.is_file()
    } == scene_files_before

    generate_data(config, scene_json=scene["scene_json"], output_root=root)
    data_files_before = {
        path.relative_to(root): path.read_bytes()
        for path in (root / "data").rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError, match="生成数据流水线目标已存在"):
        generate_data(config, scene_json=scene["scene_json"], output_root=root)
    assert {
        path.relative_to(root): path.read_bytes()
        for path in (root / "data").rglob("*")
        if path.is_file()
    } == data_files_before


def test_generate_data_parses_and_hashes_one_captured_scene_copy(
    tmp_path, monkeypatch
) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    scene_root = tmp_path / "captured_scene_source"
    scene = prepare_scene(config, scene_root)
    scene_path = Path(scene["scene_json"]).resolve()
    original_bytes = scene_path.read_bytes()
    original_sha256 = file_sha256(scene_path)
    changed_scene = json.loads(original_bytes.decode("utf-8"))
    changed_scene["name"] = "changed-after-capture"
    changed_bytes = json.dumps(changed_scene, ensure_ascii=False, indent=2).encode(
        "utf-8"
    )
    real_capture_file = pipeline_module.capture_file
    scene_capture_count = 0

    def capture_then_change_scene(path):
        nonlocal scene_capture_count
        captured = real_capture_file(path)
        if Path(path).expanduser().resolve() == scene_path:
            scene_capture_count += 1
            scene_path.write_bytes(changed_bytes)
        return captured

    monkeypatch.setattr(pipeline_module, "capture_file", capture_then_change_scene)
    artifacts = generate_data(
        config,
        scene_json=scene_path,
        output_root=tmp_path / "captured_generation",
    )

    generation_manifest = json.loads(
        Path(artifacts["generation_manifest"]).read_text(encoding="utf-8")
    )
    assert scene_capture_count == 1
    assert generation_manifest["artifact_hashes"]["scene_json"] == {
        "path": str(scene_path),
        "sha256": original_sha256,
    }
    assert scene_path.read_bytes() == changed_bytes


@pytest.fixture
def localized_run(tmp_path) -> dict[str, object]:
    config = deepcopy(DEFAULT_CONFIG)
    root = tmp_path / "run"
    config["output"]["root"] = str(root)
    config["music"]["spectrum_sampling"].update(samples_per_peak=8, local_grid_points_per_axis=9)
    source_config = tmp_path / "source_config.yaml"
    source_config.write_text("# 仅用于测试来源路径\n", encoding="utf-8")
    config["_config_path"] = str(source_config)
    scene = prepare_scene(config, root)
    data = generate_data(config, scene_json=scene["scene_json"], output_root=root)
    result = localize(
        localization_config_view(config),
        scene_json=scene["scene_json"],
        online_input=data["online_npz"],
        output_root=root,
    )
    return {
        "config": config,
        "root": root,
        "scene": scene,
        "data": data,
        "result": result,
        "source_config": source_config,
    }


def test_successful_evaluation_is_bound_to_localization_run(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    result_path = root / "localization" / "localization_result.json"
    truth_path = Path(data["truth_npz"])
    metrics_path = root / "evaluation" / "metrics.json"

    metrics = evaluate(
        result_json=result_path,
        truth_npz=truth_path,
        output_json=metrics_path,
        expected_run_id=_current_run_id(result_path),
    )

    manifest = json.loads(
        (root / "localization" / "localization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    snapshot = json.loads(
        Path(manifest["config_snapshot"]["path"]).read_text(encoding="utf-8")
    )
    assert manifest["evaluation_pending"] is False
    assert manifest["schema_version"] == 6
    assert manifest["workflow"] == "music_fine_spectrum_dbscan_v3"
    expected_artifacts = {
        "result": "localization_result.json",
        "music_spectrum": "music_spectrum.npz",
        "spectrum_samples": "spectrum_samples.json",
        "music_peaks": "music_peaks.json",
        "initial_candidates": "initial_candidates.json",
        "representative_points": "representative_points.json",
        "representative_trajectories": "representative_trajectories.json",
        "forward_check": "forward_check.json",
    }
    assert set(manifest["artifacts"]) == set(expected_artifacts)
    for artifact_name, file_name in expected_artifacts.items():
        record = manifest["artifacts"][artifact_name]
        artifact_path = root / "localization" / file_name
        assert record == {
            "path": str(artifact_path.resolve()),
            "sha256": file_sha256(artifact_path),
        }
    assert metrics["localization_run_id"] == manifest["run_id"]
    assert metrics["source_result_sha256"] == file_sha256(result_path)
    assert metrics["source_truth_sha256"] == file_sha256(truth_path)
    assert manifest["evaluation"] == {
        "path": str(metrics_path.resolve()),
        "sha256": file_sha256(metrics_path),
    }
    assert metrics["generation_bundle_id"] == manifest["generation_bundle"][
        "bundle_id"
    ]
    assert metrics["source_generation_manifest_sha256"] == manifest[
        "generation_bundle"
    ]["manifest"]["sha256"]
    assert manifest["truth_access"] == {
        "truth_file_content_loaded": False,
        "generation_manifest_truth_metadata_visible": True,
        "note": "生成清单整体可见，但定位未打开真值文件或使用真值内容",
    }
    assert "_config_path" not in snapshot["resolved_config"]
    assert snapshot["source_config_path"] == str(
        Path(localized_run["source_config"]).resolve()
    )
    assert snapshot["canonical_sha256"] == canonical_json_sha256(
        snapshot["resolved_config"]
    )

    repeated_metrics = evaluate(
        result_json=result_path,
        truth_npz=truth_path,
        output_json=metrics_path,
        expected_run_id=_current_run_id(result_path),
    )
    assert repeated_metrics["localization_run_id"] == manifest["run_id"]


def test_evaluate_manifest_publish_failure_restores_old_metrics_and_manifest(
    localized_run: dict[str, object], monkeypatch
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    result_path = root / "localization" / "localization_result.json"
    truth_path = Path(data["truth_npz"])
    metrics_path = root / "evaluation" / "metrics.json"
    manifest_path = root / "localization" / "localization_manifest.json"
    run_id = _current_run_id(result_path)
    evaluate(
        result_json=result_path,
        truth_npz=truth_path,
        output_json=metrics_path,
        expected_run_id=run_id,
    )
    old_metrics_bytes = metrics_path.read_bytes()
    old_manifest_bytes = manifest_path.read_bytes()

    monkeypatch.setattr(
        pipeline_module,
        "_load_evaluation_truth",
        lambda data_bytes: (np.asarray([123.0, 456.0]), -2.0e-6),
    )
    real_replace = pipeline_module.os.replace
    failed = False

    def fail_new_manifest_publish(source, destination):
        nonlocal failed
        source_path = Path(source)
        destination_path = Path(destination).expanduser().resolve()
        if (
            not failed
            and destination_path == manifest_path.resolve()
            and source_path.name.endswith(".tmp")
        ):
            failed = True
            raise OSError("planned-evaluation-manifest-publish-failure")
        return real_replace(source, destination)

    monkeypatch.setattr(pipeline_module.os, "replace", fail_new_manifest_publish)

    with pytest.raises(
        OSError, match="planned-evaluation-manifest-publish-failure"
    ):
        evaluate(
            result_json=result_path,
            truth_npz=truth_path,
            output_json=metrics_path,
            expected_run_id=run_id,
        )

    assert metrics_path.read_bytes() == old_metrics_bytes
    assert manifest_path.read_bytes() == old_manifest_bytes
    assert not tuple(root.rglob(".*.evaluation-backup"))
    assert not tuple(root.rglob(".*.tmp"))


def test_first_evaluation_manifest_publish_failure_restores_pending_state(
    localized_run: dict[str, object], monkeypatch
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    result_path = root / "localization" / "localization_result.json"
    truth_path = Path(data["truth_npz"])
    metrics_path = root / "evaluation" / "metrics.json"
    manifest_path = root / "localization" / "localization_manifest.json"
    manifest_before = manifest_path.read_bytes()
    assert json.loads(manifest_before.decode("utf-8"))["evaluation_pending"] is True
    assert not metrics_path.exists()
    real_replace = pipeline_module.os.replace
    failed = False

    def fail_new_manifest_publish(source, destination):
        nonlocal failed
        source_path = Path(source)
        destination_path = Path(destination).expanduser().resolve()
        if (
            not failed
            and destination_path == manifest_path.resolve()
            and source_path.name.endswith(".tmp")
        ):
            failed = True
            raise OSError("planned-first-evaluation-manifest-publish-failure")
        return real_replace(source, destination)

    monkeypatch.setattr(pipeline_module.os, "replace", fail_new_manifest_publish)

    with pytest.raises(
        OSError, match="planned-first-evaluation-manifest-publish-failure"
    ):
        evaluate(
            result_json=result_path,
            truth_npz=truth_path,
            output_json=metrics_path,
            expected_run_id=_current_run_id(result_path),
        )

    assert not metrics_path.exists()
    assert manifest_path.read_bytes() == manifest_before
    restored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert restored_manifest["evaluation_pending"] is True
    assert "evaluation" not in restored_manifest
    assert not tuple(root.rglob(".*.evaluation-backup"))
    assert not tuple(root.rglob(".*.tmp"))


def test_evaluate_requires_nonempty_expected_run_id(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    arguments = {
        "result_json": root / "localization" / "localization_result.json",
        "truth_npz": data["truth_npz"],
        "output_json": root / "evaluation" / "metrics.json",
    }

    with pytest.raises(TypeError, match="expected_run_id"):
        evaluate(**arguments)
    for invalid_run_id in ("", "   "):
        with pytest.raises(ValueError, match="expected_run_id 必须是非空字符串"):
            evaluate(**arguments, expected_run_id=invalid_run_id)

    assert not Path(arguments["output_json"]).exists()


def test_evaluate_rejects_nonfixed_output_before_artifact_access(
    localized_run: dict[str, object], monkeypatch
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    result_path = root / "localization" / "localization_result.json"
    manifest_path = root / "localization" / "localization_manifest.json"
    metrics_path = root / "evaluation" / "metrics.json"
    wrong_output_path = root / "evaluation" / "manual_metrics.json"
    run_id = _current_run_id(result_path)
    evaluate(
        result_json=result_path,
        truth_npz=data["truth_npz"],
        output_json=metrics_path,
        expected_run_id=run_id,
    )
    original_bytes = {
        "result": result_path.read_bytes(),
        "manifest": manifest_path.read_bytes(),
        "metrics": metrics_path.read_bytes(),
    }

    def reject_lock_access(*args, **kwargs):
        raise AssertionError("错误输出路径不应进入评估锁")

    def reject_artifact_access(*args, **kwargs):
        raise AssertionError("错误输出路径不应读取任何评估产物")

    monkeypatch.setattr(
        pipeline_module, "exclusive_output_root_lock", reject_lock_access
    )
    monkeypatch.setattr(pipeline_module, "capture_file", reject_artifact_access)

    with pytest.raises(ValueError, match="评估输出路径必须固定为"):
        evaluate(
            result_json=result_path,
            truth_npz=data["truth_npz"],
            output_json=wrong_output_path,
            expected_run_id=run_id,
        )
    with pytest.raises(ValueError, match="评估输出路径必须固定为"):
        pipeline_module._evaluate_locked(
            result_path=result_path.resolve(),
            truth_path=Path(data["truth_npz"]).resolve(),
            output_path=wrong_output_path.resolve(),
            expected_run_id=run_id,
        )

    assert result_path.read_bytes() == original_bytes["result"]
    assert manifest_path.read_bytes() == original_bytes["manifest"]
    assert metrics_path.read_bytes() == original_bytes["metrics"]
    assert not wrong_output_path.exists()


def test_evaluate_rejects_expired_run_id_before_opening_truth(
    localized_run: dict[str, object], monkeypatch
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    result_path = root / "localization" / "localization_result.json"
    manifest_path = root / "localization" / "localization_manifest.json"
    metrics_path = root / "evaluation" / "metrics.json"
    truth_path = Path(data["truth_npz"]).resolve()
    manifest_before = manifest_path.read_bytes()
    real_capture_file = pipeline_module.capture_file

    def reject_truth_access(path):
        if Path(path).expanduser().resolve() == truth_path:
            raise AssertionError("过期运行编号不应触碰真值")
        return real_capture_file(path)

    monkeypatch.setattr(pipeline_module, "capture_file", reject_truth_access)

    with pytest.raises(ValueError, match="期望的定位 run_id 与当前定位清单不一致"):
        evaluate(
            result_json=result_path,
            truth_npz=truth_path,
            output_json=metrics_path,
            expected_run_id="expired-localization-run",
        )

    assert manifest_path.read_bytes() == manifest_before
    assert not metrics_path.exists()


def test_evaluate_rejects_result_run_id_before_opening_truth(
    localized_run: dict[str, object], monkeypatch
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    result_path = root / "localization" / "localization_result.json"
    manifest_path = root / "localization" / "localization_manifest.json"
    metrics_path = root / "evaluation" / "metrics.json"
    truth_path = Path(data["truth_npz"]).resolve()
    expected_run_id = _current_run_id(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["localization_run_id"] = "different-result-run"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["result"]["sha256"] = file_sha256(result_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_before = manifest_path.read_bytes()
    real_capture_file = pipeline_module.capture_file

    def reject_truth_access(path):
        if Path(path).expanduser().resolve() == truth_path:
            raise AssertionError("结果运行编号不匹配时不应触碰真值")
        return real_capture_file(path)

    monkeypatch.setattr(pipeline_module, "capture_file", reject_truth_access)

    with pytest.raises(ValueError, match="期望的定位 run_id 与定位结果不一致"):
        evaluate(
            result_json=result_path,
            truth_npz=truth_path,
            output_json=metrics_path,
            expected_run_id=expected_run_id,
        )

    assert manifest_path.read_bytes() == manifest_before
    assert not metrics_path.exists()


@pytest.mark.parametrize(
    ("ue_position_m", "clock_bias_s", "message"),
    [
        ([[1.0, 2.0]], 0.0, r"ue_position_m 必须严格为形状 \(2,\)"),
        ([1.0, 2.0], [0.0], r"clock_bias_s 必须严格为标量形状 \(\)"),
        ([np.nan, 2.0], 0.0, "ue_position_m 必须全部为有限数"),
        ([1.0, np.inf], 0.0, "ue_position_m 必须全部为有限数"),
        ([1.0, 2.0], np.nan, "clock_bias_s 必须为有限数"),
        ([1.0, 2.0], np.inf, "clock_bias_s 必须为有限数"),
    ],
    ids=(
        "position-wrong-shape",
        "bias-wrong-shape",
        "position-nan",
        "position-inf",
        "bias-nan",
        "bias-inf",
    ),
)
def test_evaluation_truth_loader_rejects_wrong_shape_and_nonfinite_values(
    ue_position_m, clock_bias_s, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _load_evaluation_truth(_truth_npz_bytes(ue_position_m, clock_bias_s))


def test_evaluation_truth_loader_accepts_only_vector_and_scalar() -> None:
    position, bias = _load_evaluation_truth(
        _truth_npz_bytes([1.25, -3.5], np.asarray(25e-9))
    )

    np.testing.assert_array_equal(position, np.asarray([1.25, -3.5]))
    assert bias == pytest.approx(25e-9)


def test_invalid_truth_is_rejected_before_any_evaluation_write(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    result_path = root / "localization" / "localization_result.json"
    manifest_path = root / "localization" / "localization_manifest.json"
    metrics_path = root / "evaluation" / "metrics.json"
    truth_path = Path(data["truth_npz"])
    generation_manifest_path = Path(data["generation_manifest"])
    expected_run_id = _current_run_id(result_path)
    evaluate(
        result_json=result_path,
        truth_npz=truth_path,
        output_json=metrics_path,
        expected_run_id=expected_run_id,
    )
    metrics_before = metrics_path.read_bytes()

    truth_path.write_bytes(_truth_npz_bytes([np.nan, 2.0], 25e-9))
    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    generation_manifest["artifact_hashes"]["ground_truth"] = artifact_record(
        truth_path
    )
    generation_manifest["bundle_id"] = generation_bundle_id(generation_manifest)
    generation_manifest_path.write_text(
        json.dumps(generation_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation_bundle"]["bundle_id"] = generation_manifest["bundle_id"]
    manifest["generation_bundle"]["manifest"]["sha256"] = file_sha256(
        generation_manifest_path
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="ue_position_m 必须全部为有限数"):
        evaluate(
            result_json=result_path,
            truth_npz=truth_path,
            output_json=metrics_path,
            expected_run_id=expected_run_id,
        )

    assert metrics_path.read_bytes() == metrics_before
    assert manifest_path.read_bytes() == manifest_before


def test_evaluate_rejects_protected_output_paths_without_overwriting_sources(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    result_path = root / "localization" / "localization_result.json"
    manifest_path = root / "localization" / "localization_manifest.json"
    truth_path = Path(data["truth_npz"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protected_paths = {
        "manifest": manifest_path,
        "truth": truth_path,
        "scene": Path(manifest["inputs"]["scene"]["path"]),
        "online": Path(manifest["inputs"]["online_measurement"]["path"]),
        "config_snapshot": Path(manifest["config_snapshot"]["path"]),
        "generation_manifest": Path(
            manifest["generation_bundle"]["manifest"]["path"]
        ),
    }
    protected_paths.update(
        {
            f"artifact_{name}": Path(record["path"])
            for name, record in manifest["artifacts"].items()
        }
    )
    original_contents = {
        name: path.read_bytes() for name, path in protected_paths.items()
    }

    for output_path in protected_paths.values():
        with pytest.raises(ValueError, match="评估输出路径必须固定为"):
            evaluate(
                result_json=result_path,
                truth_npz=truth_path,
                output_json=output_path,
                expected_run_id=_current_run_id(result_path),
            )
        assert {
            name: path.read_bytes() for name, path in protected_paths.items()
        } == original_contents

    assert not (root / "evaluation" / "metrics.json").exists()


def test_localize_rejects_undeclared_generation_manifest_extensions(
    tmp_path,
) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    root = tmp_path / "generation_protection"
    scene = prepare_scene(config, root)
    data = generate_data(config, scene_json=scene["scene_json"], output_root=root)
    declared_paths = {
        "snapshot": tmp_path / "generation_config_snapshot.json",
        "source": tmp_path / "generation_source.yaml",
        "source_config": tmp_path / "generation_source_config.yaml",
        "scene": tmp_path / "declared_scene_artifact.bin",
        "data": tmp_path / "declared_data_artifact.bin",
        "hash": tmp_path / "declared_hashed_artifact.bin",
    }
    for name, path in declared_paths.items():
        path.write_bytes(f"preserve-{name}".encode("utf-8"))

    generation_manifest_path = Path(data["generation_manifest"])
    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    generation_manifest["config_snapshot"] = {
        "path": str(declared_paths["snapshot"]),
        "source_path": str(declared_paths["source"]),
        "source_config_path": str(declared_paths["source_config"]),
    }
    generation_manifest["scene_artifacts"] = {
        "extra": str(declared_paths["scene"])
    }
    generation_manifest["data_artifacts"] = {
        "extra": str(declared_paths["data"])
    }
    generation_manifest["artifact_hashes"]["extra"] = artifact_record(
        declared_paths["hash"]
    )
    generation_manifest_path.write_text(
        json.dumps(generation_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    original_contents = {
        name: path.read_bytes() for name, path in declared_paths.items()
    }

    with pytest.raises(ValueError, match="生成清单顶层.*字段集合不符合契约"):
        localize(
            localization_config_view(config),
            scene_json=scene["scene_json"],
            online_input=data["online_npz"],
            generation_manifest=generation_manifest_path,
            output_root=root,
        )

    assert {
        name: path.read_bytes() for name, path in declared_paths.items()
    } == original_contents


def test_output_root_lock_blocks_another_linux_process(tmp_path) -> None:
    output_root = tmp_path / "locked_run"
    context = multiprocessing.get_context("fork")
    messages = context.Queue()
    process = context.Process(
        target=_acquire_lock_in_child,
        args=(str(output_root), messages),
    )

    with exclusive_output_root_lock(output_root):
        process.start()
        assert messages.get(timeout=2.0) == "waiting"
        with pytest.raises(Empty):
            messages.get(timeout=0.2)

    assert messages.get(timeout=2.0) == "acquired"
    process.join(timeout=2.0)
    assert process.exitcode == 0


def test_prepare_scene_public_entry_waits_for_output_root_lock(tmp_path) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    output_root = tmp_path / "locked_prepare_scene"
    context = multiprocessing.get_context("fork")
    messages = context.Queue()
    process = context.Process(
        target=_prepare_scene_in_child,
        args=(config, str(output_root), messages),
    )

    with exclusive_output_root_lock(output_root):
        process.start()
        assert messages.get(timeout=2.0) == "waiting"
        with pytest.raises(Empty):
            messages.get(timeout=0.2)

    status, message = messages.get(timeout=10.0)
    process.join(timeout=10.0)
    assert (status, message) == ("ok", "")
    assert process.exitcode == 0
    assert (output_root / "scene" / "preprocess_manifest.json").is_file()


def test_generate_data_public_entry_waits_for_output_root_lock(tmp_path) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    source_scene = prepare_scene(config, tmp_path / "generation_lock_source")
    output_root = tmp_path / "locked_generate_data"
    context = multiprocessing.get_context("fork")
    messages = context.Queue()
    process = context.Process(
        target=_generate_data_in_child,
        args=(config, source_scene["scene_json"], str(output_root), messages),
    )

    with exclusive_output_root_lock(output_root):
        process.start()
        assert messages.get(timeout=2.0) == "waiting"
        with pytest.raises(Empty):
            messages.get(timeout=0.2)

    status, message = messages.get(timeout=10.0)
    process.join(timeout=10.0)
    assert (status, message) == ("ok", "")
    assert process.exitcode == 0
    assert (output_root / "data" / "generation_manifest.json").is_file()


def test_waiting_evaluation_process_rejects_a_newer_localization_run(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    result_path = root / "localization" / "localization_result.json"
    manifest_path = root / "localization" / "localization_manifest.json"
    metrics_path = root / "evaluation" / "metrics.json"
    old_run_id = _current_run_id(result_path)
    context = multiprocessing.get_context("fork")
    messages = context.Queue()
    process = context.Process(
        target=_evaluate_in_child,
        args=(
            str(result_path),
            str(data["truth_npz"]),
            str(metrics_path),
            old_run_id,
            messages,
        ),
    )

    try:
        with exclusive_output_root_lock(root):
            process.start()
            assert messages.get(timeout=2.0) == "waiting"
            with pytest.raises(Empty):
                messages.get(timeout=0.2)
            newer_result = pipeline_module._localize_locked(
                localization_config_view(config),
                scene_json=scene["scene_json"],
                online_input=data["online_npz"],
                generation_manifest=data["generation_manifest"],
                output_root=root,
            )
            newer_manifest_before = manifest_path.read_bytes()
            assert newer_result["localization_run_id"] != old_run_id

        status, message = messages.get(timeout=10.0)
        assert status == "ValueError"
        assert "期望的定位 run_id 与当前定位清单不一致" in message
    finally:
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)

    assert process.exitcode == 0
    assert manifest_path.read_bytes() == newer_manifest_before
    assert not metrics_path.exists()


def test_localize_direct_api_rejects_hidden_fields_before_opening_inputs(
    tmp_path,
) -> None:
    config = localization_config_view(deepcopy(DEFAULT_CONFIG))
    config["music"]["hidden_true_delay_s"] = 25.0e-9
    output_root = tmp_path / "must_not_be_created"

    with pytest.raises(ValueError, match="music 包含未定义字段.*hidden_true_delay_s"):
        localize(
            config,
            scene_json=tmp_path / "missing_scene.json",
            online_input=tmp_path / "missing_online.npz",
            output_root=output_root,
        )

    assert not output_root.exists()


def test_evaluate_rechecks_manifest_before_writing_back(
    localized_run: dict[str, object], monkeypatch
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    manifest_path = root / "localization" / "localization_manifest.json"
    output_path = root / "evaluation" / "metrics.json"
    real_capture_file = pipeline_module.capture_file
    manifest_capture_count = 0

    def capture_and_replace_manifest(path):
        nonlocal manifest_capture_count
        resolved = Path(path).expanduser().resolve()
        if resolved == manifest_path.resolve():
            manifest_capture_count += 1
            if manifest_capture_count == 2:
                replacement = json.loads(manifest_path.read_text(encoding="utf-8"))
                replacement["run_id"] = "new-localization-run"
                manifest_path.write_text(
                    json.dumps(replacement, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        return real_capture_file(path)

    monkeypatch.setattr(
        pipeline_module, "capture_file", capture_and_replace_manifest
    )

    with pytest.raises(ValueError, match="run_id 在评估运行期间发生变化"):
        evaluate(
            result_json=root / "localization" / "localization_result.json",
            truth_npz=data["truth_npz"],
            output_json=output_path,
            expected_run_id=_current_run_id(
                root / "localization" / "localization_result.json"
            ),
        )

    assert not output_path.exists()


def test_write_json_failure_keeps_old_file_and_cleans_temporary_file(
    tmp_path, monkeypatch
) -> None:
    output_path = tmp_path / "atomic.json"
    output_path.write_text('{"old": true}', encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("replace-failed-for-test")

    monkeypatch.setattr(pipeline_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace-failed-for-test"):
        _write_json(output_path, {"new": True})

    assert output_path.read_text(encoding="utf-8") == '{"old": true}'
    assert list(tmp_path.glob(".atomic.json.*.tmp")) == []


def test_localize_parses_the_same_scene_and_online_bytes_that_were_hashed(
    localized_run: dict[str, object], monkeypatch
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene_artifacts = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene_artifacts, dict)
    assert isinstance(data, dict)
    scene_path = Path(scene_artifacts["scene_json"]).resolve()
    online_path = Path(data["online_npz"]).resolve()
    original_hashes = {
        scene_path: file_sha256(scene_path),
        online_path: file_sha256(online_path),
    }
    real_capture_file = pipeline_module.capture_file

    def capture_then_change_source(path):
        captured = real_capture_file(path)
        if captured.path == scene_path:
            captured.path.write_bytes(b"not-json-after-capture")
        elif captured.path == online_path:
            captured.path.write_bytes(b"not-npz-after-capture")
        return captured

    monkeypatch.setattr(pipeline_module, "capture_file", capture_then_change_source)

    result = localize(
        localization_config_view(config),
        scene_json=scene_path,
        online_input=online_path,
        generation_manifest=data["generation_manifest"],
        output_root=root,
    )

    manifest = json.loads(
        (root / "localization" / "localization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["localization_run_id"] == manifest["run_id"]
    assert manifest["inputs"]["scene"]["sha256"] == original_hashes[scene_path]
    assert (
        manifest["inputs"]["online_measurement"]["sha256"]
        == original_hashes[online_path]
    )


def test_evaluate_parses_the_same_result_and_truth_bytes_that_were_hashed(
    localized_run: dict[str, object], monkeypatch
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    result_path = root / "localization" / "localization_result.json"
    truth_path = Path(data["truth_npz"]).resolve()
    original_result_sha256 = file_sha256(result_path)
    original_truth_sha256 = file_sha256(truth_path)
    real_capture_file = pipeline_module.capture_file

    def capture_then_change_source(path):
        captured = real_capture_file(path)
        if captured.path == result_path.resolve():
            captured.path.write_bytes(b"not-json-after-capture")
        elif captured.path == truth_path:
            captured.path.write_bytes(b"not-npz-after-capture")
        return captured

    monkeypatch.setattr(pipeline_module, "capture_file", capture_then_change_source)

    metrics = evaluate(
        result_json=result_path,
        truth_npz=truth_path,
        output_json=root / "evaluation" / "metrics.json",
        expected_run_id=_current_run_id(result_path),
    )

    assert metrics["source_result_sha256"] == original_result_sha256
    assert metrics["source_truth_sha256"] == original_truth_sha256


def test_localize_rejects_full_config_before_creating_output(tmp_path) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    root = tmp_path / "must_not_be_created"

    with pytest.raises(ValueError, match="必须先生成定位专用配置"):
        localize(
            config,
            scene_json=tmp_path / "missing_scene.json",
            online_input=tmp_path / "missing_online.npz",
            output_root=root,
        )

    assert not root.exists()


def test_localize_rejects_scene_and_online_from_different_generation_batches(
    localized_run: dict[str, object], tmp_path
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    existing_result = root / "localization" / "localization_result.json"
    existing_result_bytes = existing_result.read_bytes()

    other_config = deepcopy(config)
    other_config["project"]["random_seed"] += 1
    other_root = tmp_path / "other_batch"
    other_scene = prepare_scene(other_config, other_root)
    other_data = generate_data(
        other_config,
        scene_json=other_scene["scene_json"],
        output_root=other_root,
    )

    with pytest.raises(ValueError, match="在线 CSI的路径与生成清单不一致"):
        localize(
            localization_config_view(config),
            scene_json=scene["scene_json"],
            online_input=other_data["online_npz"],
            generation_manifest=data["generation_manifest"],
            output_root=root,
        )

    assert existing_result.read_bytes() == existing_result_bytes


def test_evaluate_rejects_truth_from_different_generation_batch(
    localized_run: dict[str, object], tmp_path
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    assert isinstance(config, dict)
    other_config = deepcopy(config)
    other_config["project"]["random_seed"] += 1
    other_root = tmp_path / "other_truth_batch"
    other_scene = prepare_scene(other_config, other_root)
    other_data = generate_data(
        other_config,
        scene_json=other_scene["scene_json"],
        output_root=other_root,
    )
    manifest_path = root / "localization" / "localization_manifest.json"
    manifest_before = manifest_path.read_bytes()
    metrics_path = root / "evaluation" / "metrics.json"

    with pytest.raises(ValueError, match="评估真值的路径与生成清单不一致"):
        evaluate(
            result_json=root / "localization" / "localization_result.json",
            truth_npz=other_data["truth_npz"],
            output_json=metrics_path,
            expected_run_id=_current_run_id(
                root / "localization" / "localization_result.json"
            ),
        )

    assert manifest_path.read_bytes() == manifest_before
    assert not metrics_path.exists()


def test_evaluate_rejects_truth_changed_after_generation(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    truth_path = Path(data["truth_npz"])
    truth_path.write_bytes(truth_path.read_bytes() + b"changed-after-generation")
    manifest_path = root / "localization" / "localization_manifest.json"
    manifest_before = manifest_path.read_bytes()
    metrics_path = root / "evaluation" / "metrics.json"

    with pytest.raises(ValueError, match="评估真值 SHA-256"):
        evaluate(
            result_json=root / "localization" / "localization_result.json",
            truth_npz=truth_path,
            output_json=metrics_path,
            expected_run_id=_current_run_id(
                root / "localization" / "localization_result.json"
            ),
        )

    assert manifest_path.read_bytes() == manifest_before
    assert not metrics_path.exists()


def test_evaluate_rejects_generation_manifest_changed_after_localization(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    generation_manifest_path = Path(data["generation_manifest"])
    generation_manifest_path.write_text(
        generation_manifest_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    localization_manifest_path = root / "localization" / "localization_manifest.json"
    localization_manifest_before = localization_manifest_path.read_bytes()
    metrics_path = root / "evaluation" / "metrics.json"

    with pytest.raises(ValueError, match="生成清单 SHA-256"):
        evaluate(
            result_json=root / "localization" / "localization_result.json",
            truth_npz=data["truth_npz"],
            output_json=metrics_path,
            expected_run_id=_current_run_id(
                root / "localization" / "localization_result.json"
            ),
        )

    assert localization_manifest_path.read_bytes() == localization_manifest_before
    assert not metrics_path.exists()


def test_evaluate_rejects_result_modified_after_localization(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    result_path = root / "localization" / "localization_result.json"
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="SHA-256"):
        evaluate(
            result_json=result_path,
            truth_npz=data["truth_npz"],
            output_json=root / "evaluation" / "metrics.json",
            expected_run_id=_current_run_id(result_path),
        )


def test_localize_rerun_marks_existing_evaluation_pending(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    manifest_path = root / "localization" / "localization_manifest.json"
    metrics_path = root / "evaluation" / "metrics.json"
    evaluate(
        result_json=root / "localization" / "localization_result.json",
        truth_npz=data["truth_npz"],
        output_json=metrics_path,
        expected_run_id=_current_run_id(
            root / "localization" / "localization_result.json"
        ),
    )
    completed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_metrics = metrics_path.read_bytes()
    old_run_id = completed_manifest["run_id"]

    rerun_result = localize(
        localization_config_view(config),
        scene_json=scene["scene_json"],
        online_input=data["online_npz"],
        output_root=root,
    )

    rerun_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rerun_manifest["run_id"] != completed_manifest["run_id"]
    assert rerun_result["localization_run_id"] == rerun_manifest["run_id"]
    assert rerun_manifest["evaluation_pending"] is True
    assert "evaluation" not in rerun_manifest
    assert not metrics_path.exists()
    history_dir = root / "evaluation" / "history" / old_run_id
    history_metrics_path = history_dir / "metrics.json"
    history_manifest_path = history_dir / "localization_manifest.json"
    assert history_metrics_path.read_bytes() == old_metrics
    archived_manifest = json.loads(history_manifest_path.read_text(encoding="utf-8"))
    expected_archived_manifest = deepcopy(completed_manifest)
    expected_archived_manifest["evaluation"]["path"] = str(
        history_metrics_path.resolve()
    )
    expected_archived_manifest["result"] = str(
        (history_dir / "localization_result.json").resolve()
    )
    expected_archived_manifest["config_snapshot"]["path"] = str(
        (history_dir / "localization_config.json").resolve()
    )
    for artifact_name, record in expected_archived_manifest["artifacts"].items():
        record["path"] = str(
            (history_dir / Path(completed_manifest["artifacts"][artifact_name]["path"]).name).resolve()
        )
    assert archived_manifest == expected_archived_manifest
    for record in archived_manifest["artifacts"].values():
        assert file_sha256(record["path"]) == record["sha256"]
        assert history_dir.resolve() in Path(record["path"]).resolve().parents
    assert file_sha256(archived_manifest["config_snapshot"]["path"]) == (
        archived_manifest["config_snapshot"]["file_sha256"]
    )
    assert file_sha256(archived_manifest["evaluation"]["path"]) == (
        archived_manifest["evaluation"]["sha256"]
    )

    new_metrics = evaluate(
        result_json=root / "localization" / "localization_result.json",
        truth_npz=data["truth_npz"],
        output_json=metrics_path,
        expected_run_id=rerun_manifest["run_id"],
    )
    assert metrics_path.exists()
    assert new_metrics["localization_run_id"] == rerun_manifest["run_id"]
    assert history_metrics_path.read_bytes() == old_metrics


def test_localize_computation_failure_does_not_archive_old_evaluation(
    localized_run: dict[str, object], monkeypatch
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    manifest_path = root / "localization" / "localization_manifest.json"
    result_path = root / "localization" / "localization_result.json"
    metrics_path = root / "evaluation" / "metrics.json"
    evaluate(
        result_json=result_path,
        truth_npz=data["truth_npz"],
        output_json=metrics_path,
        expected_run_id=_current_run_id(result_path),
    )
    original_files = {
        "manifest": manifest_path.read_bytes(),
        "result": result_path.read_bytes(),
        "metrics": metrics_path.read_bytes(),
    }

    def fail_during_computation(*args, **kwargs):
        raise RuntimeError("planned-computation-failure")

    monkeypatch.setattr(
        pipeline_module, "get_music_computer", fail_during_computation
    )
    with pytest.raises(RuntimeError, match="planned-computation-failure"):
        localize(
            localization_config_view(config),
            scene_json=scene["scene_json"],
            online_input=data["online_npz"],
            output_root=root,
        )

    assert manifest_path.read_bytes() == original_files["manifest"]
    assert result_path.read_bytes() == original_files["result"]
    assert metrics_path.read_bytes() == original_files["metrics"]
    assert not (root / "evaluation" / "history").exists()


def test_staged_manifest_failure_keeps_old_active_run_self_consistent(
    localized_run: dict[str, object], monkeypatch
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    manifest_path = root / "localization" / "localization_manifest.json"
    result_path = root / "localization" / "localization_result.json"
    metrics_path = root / "evaluation" / "metrics.json"
    evaluate(
        result_json=result_path,
        truth_npz=data["truth_npz"],
        output_json=metrics_path,
        expected_run_id=_current_run_id(result_path),
    )
    old_files = {
        "manifest": manifest_path.read_bytes(),
        "result": result_path.read_bytes(),
        "metrics": metrics_path.read_bytes(),
    }
    real_write_json = pipeline_module._write_json

    def fail_staged_manifest(path, value):
        resolved = Path(path)
        if (
            resolved.name == "localization_manifest.json"
            and resolved.parent.name.startswith(".localization-staging-")
        ):
            raise OSError("planned-staged-manifest-failure")
        return real_write_json(resolved, value)

    monkeypatch.setattr(pipeline_module, "_write_json", fail_staged_manifest)
    with pytest.raises(OSError, match="planned-staged-manifest-failure"):
        localize(
            localization_config_view(config),
            scene_json=scene["scene_json"],
            online_input=data["online_npz"],
            output_root=root,
        )

    assert manifest_path.read_bytes() == old_files["manifest"]
    assert result_path.read_bytes() == old_files["result"]
    assert metrics_path.read_bytes() == old_files["metrics"]
    assert list(root.glob(".localization-staging-*")) == []


def test_directory_publish_failure_rolls_back_old_active_run(
    localized_run: dict[str, object], monkeypatch
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    manifest_path = root / "localization" / "localization_manifest.json"
    result_path = root / "localization" / "localization_result.json"
    metrics_path = root / "evaluation" / "metrics.json"
    evaluate(
        result_json=result_path,
        truth_npz=data["truth_npz"],
        output_json=metrics_path,
        expected_run_id=_current_run_id(result_path),
    )
    old_files = {
        "manifest": manifest_path.read_bytes(),
        "result": result_path.read_bytes(),
        "metrics": metrics_path.read_bytes(),
    }
    real_replace = pipeline_module.os.replace

    def fail_staging_directory_switch(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.name.startswith(".localization-staging-")
            and source_path.is_dir()
            and destination_path.resolve() == (root / "localization").resolve()
        ):
            raise OSError("planned-directory-switch-failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        pipeline_module.os, "replace", fail_staging_directory_switch
    )
    with pytest.raises(OSError, match="planned-directory-switch-failure"):
        localize(
            localization_config_view(config),
            scene_json=scene["scene_json"],
            online_input=data["online_npz"],
            output_root=root,
        )

    assert manifest_path.read_bytes() == old_files["manifest"]
    assert result_path.read_bytes() == old_files["result"]
    assert metrics_path.read_bytes() == old_files["metrics"]
    assert list(root.glob(".localization-staging-*")) == []
    assert list(root.glob(".localization-backup-*")) == []
    assert list(root.glob(".evaluation-metrics-backup-*")) == []


def test_localize_writes_exact_run_receipt_and_never_overwrites_it(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    receipt_path = root / "receipts" / "explicit-localization-run.json"

    result = localize(
        localization_config_view(config),
        scene_json=scene["scene_json"],
        online_input=data["online_npz"],
        output_root=root,
        run_receipt=receipt_path,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = root / "localization" / "localization_result.json"
    manifest_path = root / "localization" / "localization_manifest.json"
    assert set(receipt) == {"run_id", "result", "manifest"}
    assert receipt["run_id"] == result["localization_run_id"]
    assert receipt["result"] == artifact_record(result_path)
    assert receipt["manifest"] == artifact_record(manifest_path)
    assert list(receipt_path.parent.glob(f".{receipt_path.name}.*.tmp")) == []
    receipt_before = receipt_path.read_bytes()
    active_before = {
        path.name: path.read_bytes()
        for path in (root / "localization").iterdir()
        if path.is_file()
    }

    with pytest.raises(FileExistsError, match="运行回执已存在，拒绝覆盖"):
        localize(
            localization_config_view(config),
            scene_json=scene["scene_json"],
            online_input=data["online_npz"],
            output_root=root,
            run_receipt=receipt_path,
        )

    assert receipt_path.read_bytes() == receipt_before
    assert {
        path.name: path.read_bytes()
        for path in (root / "localization").iterdir()
        if path.is_file()
    } == active_before
    assert list(root.glob(".localization-staging-*")) == []
    assert list(root.glob(".localization-backup-*")) == []


def test_run_receipt_publish_failure_rolls_back_localization_directory(
    localized_run: dict[str, object], monkeypatch
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    receipt_path = root / "receipts" / "must-not-exist.json"
    result_path = root / "localization" / "localization_result.json"
    metrics_path = root / "evaluation" / "metrics.json"
    evaluate(
        result_json=result_path,
        truth_npz=data["truth_npz"],
        output_json=metrics_path,
        expected_run_id=_current_run_id(result_path),
    )
    metrics_before = metrics_path.read_bytes()
    active_before = {
        path.name: path.read_bytes()
        for path in (root / "localization").iterdir()
        if path.is_file()
    }

    real_write_new_json_atomic = pipeline_module._write_new_json_atomic

    def fail_receipt_publish(path, data):
        real_write_new_json_atomic(path, data)
        raise OSError("planned-receipt-publish-failure")

    monkeypatch.setattr(
        pipeline_module, "_write_new_json_atomic", fail_receipt_publish
    )
    with pytest.raises(OSError, match="planned-receipt-publish-failure"):
        localize(
            localization_config_view(config),
            scene_json=scene["scene_json"],
            online_input=data["online_npz"],
            output_root=root,
            run_receipt=receipt_path,
        )

    assert not receipt_path.exists()
    assert {
        path.name: path.read_bytes()
        for path in (root / "localization").iterdir()
        if path.is_file()
    } == active_before
    assert metrics_path.read_bytes() == metrics_before
    assert list(root.glob(".localization-staging-*")) == []
    assert list(root.glob(".localization-backup-*")) == []


def test_run_evaluate_script_refuses_to_guess_current_run_id(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output"
    result_path = output_root / "localization" / "localization_result.json"
    metrics_path = output_root / "evaluation" / "metrics.json"
    environment = os.environ.copy()
    environment.pop("RUN_RECEIPT_JSON", None)
    environment.pop("EXPECTED_RUN_ID", None)
    environment["PYTHON_BIN"] = sys.executable
    environment["RESULT_JSON"] = str(result_path)
    environment["TRUTH_NPZ"] = str(tmp_path / "truth.npz")
    environment["METRICS_JSON"] = str(metrics_path)

    completed = subprocess.run(
        ["bash", str(project_root / "run_evaluate.sh")],
        cwd=project_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "不会从当前定位清单猜测" in completed.stderr
    assert not metrics_path.exists()


def test_run_evaluate_script_rejects_nonfixed_metrics_path_without_changes(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    result_path = root / "localization" / "localization_result.json"
    manifest_path = root / "localization" / "localization_manifest.json"
    metrics_path = root / "evaluation" / "metrics.json"
    wrong_metrics_path = root / "evaluation" / "manual_metrics.json"
    run_id = _current_run_id(result_path)
    evaluate(
        result_json=result_path,
        truth_npz=data["truth_npz"],
        output_json=metrics_path,
        expected_run_id=run_id,
    )
    original_bytes = {
        "result": result_path.read_bytes(),
        "manifest": manifest_path.read_bytes(),
        "metrics": metrics_path.read_bytes(),
    }
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("RUN_RECEIPT_JSON", None)
    environment.update(
        {
            "PYTHON_BIN": sys.executable,
            "EXPECTED_RUN_ID": run_id,
            "RESULT_JSON": str(result_path),
            "TRUTH_NPZ": str(data["truth_npz"]),
            "METRICS_JSON": str(wrong_metrics_path),
        }
    )

    completed = subprocess.run(
        ["bash", str(project_root / "run_evaluate.sh")],
        cwd=project_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "METRICS_JSON 必须固定为" in completed.stderr
    assert result_path.read_bytes() == original_bytes["result"]
    assert manifest_path.read_bytes() == original_bytes["manifest"]
    assert metrics_path.read_bytes() == original_bytes["metrics"]
    assert not wrong_metrics_path.exists()


def test_run_evaluate_script_can_reuse_receipt_after_successful_evaluation(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    receipt_path = root / "receipts" / "shell-evaluation.json"
    result = localize(
        localization_config_view(config),
        scene_json=scene["scene_json"],
        online_input=data["online_npz"],
        output_root=root,
        run_receipt=receipt_path,
    )
    result_path = root / "localization" / "localization_result.json"
    metrics_path = root / "evaluation" / "metrics.json"
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("EXPECTED_RUN_ID", None)
    environment.update(
        {
            "PYTHON_BIN": sys.executable,
            "RUN_RECEIPT_JSON": str(receipt_path),
            "RESULT_JSON": str(result_path),
            "TRUTH_NPZ": str(data["truth_npz"]),
            "METRICS_JSON": str(metrics_path),
        }
    )

    first = subprocess.run(
        ["bash", str(project_root / "run_evaluate.sh")],
        cwd=project_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    second = subprocess.run(
        ["bash", str(project_root / "run_evaluate.sh")],
        cwd=project_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["localization_run_id"] == result["localization_run_id"]


def test_run_evaluate_script_rejects_non_evaluation_manifest_changes(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    receipt_path = root / "receipts" / "shell-tamper-check.json"
    localize(
        localization_config_view(config),
        scene_json=scene["scene_json"],
        online_input=data["online_npz"],
        output_root=root,
        run_receipt=receipt_path,
    )
    localization_dir = root / "localization"
    result_path = localization_dir / "localization_result.json"
    manifest_path = localization_dir / "localization_manifest.json"
    metrics_path = root / "evaluation" / "metrics.json"
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("EXPECTED_RUN_ID", None)
    environment.update(
        {
            "PYTHON_BIN": sys.executable,
            "RUN_RECEIPT_JSON": str(receipt_path),
            "RESULT_JSON": str(result_path),
            "TRUTH_NPZ": str(data["truth_npz"]),
            "METRICS_JSON": str(metrics_path),
        }
    )

    completed = subprocess.run(
        ["bash", str(project_root / "run_evaluate.sh")],
        cwd=project_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    evaluated_manifest_bytes = manifest_path.read_bytes()
    protected_artifacts = {
        path: path.read_bytes()
        for path in (*localization_dir.iterdir(), metrics_path)
        if path.is_file()
    }
    tamper_paths = (
        ("generation_bundle", "bundle_id"),
        ("inputs", "scene", "sha256"),
        ("config_snapshot", "canonical_sha256"),
        ("artifacts", "music_peaks", "sha256"),
    )

    for tamper_path in tamper_paths:
        manifest = json.loads(evaluated_manifest_bytes.decode("utf-8"))
        target = manifest
        for key in tamper_path[:-1]:
            target = target[key]
        target[tamper_path[-1]] = "tampered"
        tampered_manifest_bytes = json.dumps(
            manifest, ensure_ascii=False, indent=2
        ).encode("utf-8")
        manifest_path.write_bytes(tampered_manifest_bytes)

        rejected = subprocess.run(
            ["bash", str(project_root / "run_evaluate.sh")],
            cwd=project_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        assert rejected.returncode != 0
        assert "除评估写回外还发生了变化" in rejected.stderr
        assert manifest_path.read_bytes() == tampered_manifest_bytes
        for artifact_path, expected_bytes in protected_artifacts.items():
            if artifact_path != manifest_path:
                assert artifact_path.read_bytes() == expected_bytes
        manifest_path.write_bytes(evaluated_manifest_bytes)


def _write_legacy_archive_fixture(root: Path, schema_version: int) -> None:
    """构造旧版的文件集合和来源链；不能只改新版清单的版本号。"""
    localization = root / "localization"
    _write_json(localization / "raw_reverse_candidates.json", [])
    _write_json(localization / "clustered_candidates.json", [])
    result_path = localization / "localization_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("workflow", None)
    result.pop("forward_check", None)
    result["output_type"] = "gaussian_position_estimate"
    result["diagnostics"]["covariance_source"] = "nominal_fallback_no_solved_perturbation"
    _write_json(result_path, result)
    # 这里表示历史运行的扰动样本全部失败，保留其空解集合。
    np.savez(localization / "uncertainty_solutions.npz", positions_m=np.empty((0, 2)),
             betas_m=np.empty(0), sigmas_m2=np.empty((0, 2, 2)), weights=np.empty(0))
    _write_json(localization / "bootstrap_diagnostics.json", [])
    peaks_path = localization / "music_peaks.json"
    peaks = json.loads(peaks_path.read_text(encoding="utf-8"))
    peaks.pop("workflow", None)
    peaks["nominal_observation_samples"] = [sample for sample in peaks.pop("observation_samples", [])
                                               if sample["sample_id"].endswith(":nominal")]
    peaks["perturbed_observation_samples"] = []
    peaks["associations"] = []
    _write_json(peaks_path, peaks)
    metrics_path = root / "evaluation/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["schema_version"] = schema_version
    metrics["source_result_sha256"] = file_sha256(result_path)
    if schema_version == 2:
        metrics.pop("generation_bundle_id")
        metrics.pop("source_generation_manifest_sha256")
    _write_json(metrics_path, metrics)
    manifest_path = localization / "localization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = schema_version
    manifest.pop("workflow", None)
    manifest["artifacts"] = {key: artifact_record(localization / name) for key, name in {
        "result": "localization_result.json", "music_spectrum": "music_spectrum.npz",
        "music_peaks": "music_peaks.json", "raw_reverse_candidates": "raw_reverse_candidates.json",
        "clustered_candidates": "clustered_candidates.json", "uncertainty_solutions": "uncertainty_solutions.npz",
        "bootstrap_diagnostics": "bootstrap_diagnostics.json"}.items()}
    if schema_version == 2:
        manifest.pop("generation_bundle")
        manifest.pop("truth_access")
        # 第 2 版仅记录主结果摘要，归档时需要补录诊断文件摘要。
        manifest["artifacts"] = {"result": manifest["artifacts"]["result"]}
    manifest["evaluation"] = artifact_record(metrics_path)
    _write_json(manifest_path, manifest)


def test_localize_archives_legacy_schema2_evaluation_without_forging_sources(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    manifest_path = root / "localization" / "localization_manifest.json"
    metrics_path = root / "evaluation" / "metrics.json"
    evaluate(
        result_json=root / "localization" / "localization_result.json",
        truth_npz=data["truth_npz"],
        output_json=metrics_path,
        expected_run_id=_current_run_id(
            root / "localization" / "localization_result.json"
        ),
    )

    _write_legacy_archive_fixture(root, 2)
    legacy_metrics_bytes = metrics_path.read_bytes()
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_run_id = legacy_manifest["run_id"]

    rerun_config_path = root.parent / "schema2_rerun.yaml"
    rerun_config_path.write_text(
        json.dumps(
            {
                "music": {
                    "spectrum_sampling": {"samples_per_peak": 8, "local_grid_points_per_axis": 9},
                },
                "output": {"root": str(root)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    run_offline_demo(rerun_config_path)

    history_dir = root / "evaluation" / "history" / old_run_id
    assert (history_dir / "metrics.json").read_bytes() == legacy_metrics_bytes
    archived_manifest = json.loads(
        (history_dir / "localization_manifest.json").read_text(encoding="utf-8")
    )
    assert archived_manifest["schema_version"] == 2
    assert "generation_bundle" not in archived_manifest
    assert archived_manifest["evaluation"]["sha256"] == file_sha256(
        history_dir / "metrics.json"
    )
    assert archived_manifest["evaluation"]["path"] == str(
        (history_dir / "metrics.json").resolve()
    )
    assert archived_manifest["artifacts"]["result"]["sha256"] == (
        legacy_manifest["artifacts"]["result"]["sha256"]
    )
    assert Path(archived_manifest["artifacts"]["result"]["path"]).parent == (
        history_dir.resolve()
    )
    assert archived_manifest["archive_migration"] == {
        "source_schema_version": 2,
        "artifact_hashes_computed_during_archive": [
            "bootstrap_diagnostics",
            "clustered_candidates",
            "music_peaks",
            "music_spectrum",
            "raw_reverse_candidates",
            "uncertainty_solutions",
        ],
    }
    assert set(archived_manifest["artifacts"]) == {
        "result",
        "music_spectrum",
        "uncertainty_solutions",
        "music_peaks",
        "raw_reverse_candidates",
        "clustered_candidates",
        "bootstrap_diagnostics",
    }
    for record in archived_manifest["artifacts"].values():
        assert file_sha256(record["path"]) == record["sha256"]
    assert set(archived_manifest["archived_sources"]) == {
        "scene",
        "online_measurement",
        "ground_truth",
    }
    for record in archived_manifest["archived_sources"].values():
        assert file_sha256(record["path"]) == record["sha256"]


@pytest.mark.parametrize("schema_version", [3, 4, 5, 6])
def test_archive_uses_frozen_records_after_generation_manifest_updates(
    localized_run: dict[str, object], schema_version,
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    manifest_path = root / "localization" / "localization_manifest.json"
    metrics_path = root / "evaluation" / "metrics.json"
    evaluate(
        result_json=root / "localization" / "localization_result.json",
        truth_npz=data["truth_npz"],
        output_json=metrics_path,
        expected_run_id=_current_run_id(
            root / "localization" / "localization_result.json"
        ),
    )
    if schema_version == 3:
        _write_legacy_archive_fixture(root, 3)
    elif schema_version == 4:
        # 构造谱面轨迹聚类版本的完整产物集合，验证升级时仍可归档。
        folder = root / "localization"
        _write_json(folder / "raw_reverse_candidates.json", [])
        _write_json(folder / "clustered_candidates.json", [])
        manifest = json.loads(manifest_path.read_text())
        manifest.update(schema_version=4, workflow="music_spectrum_sampling_v1")
        manifest["artifacts"] = {
            name: artifact_record(folder / filename)
            for name, filename in pipeline_module._SPECTRUM_TRAJECTORY_ARTIFACT_FILENAMES.items()
        }
        _write_json(manifest_path, manifest)
    if schema_version == 5:
        manifest = json.loads(manifest_path.read_text())
        manifest.update(schema_version=5, workflow="music_point_clustering_v2")
        _write_json(manifest_path, manifest)
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_run_id = old_manifest["run_id"]
    old_generation_sha256 = old_manifest["generation_bundle"]["manifest"][
        "sha256"
    ]

    generation_manifest_path = Path(data["generation_manifest"])
    regenerated_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    generation_manifest_path.write_text(
        json.dumps(regenerated_manifest, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    assert file_sha256(generation_manifest_path) != old_generation_sha256

    localize(
        localization_config_view(config),
        scene_json=scene["scene_json"],
        online_input=data["online_npz"],
        generation_manifest=generation_manifest_path,
        output_root=root,
    )

    archived_manifest = json.loads(
        (
            root
            / "evaluation"
            / "history"
            / old_run_id
            / "localization_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert archived_manifest["generation_bundle"]["manifest"]["sha256"] == (
        old_generation_sha256
    )
    assert archived_manifest["schema_version"] == schema_version
    new_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert new_manifest["generation_bundle"]["manifest"]["sha256"] == file_sha256(
        generation_manifest_path
    )


def test_run_offline_demo_twice_archives_first_completed_run(tmp_path) -> None:
    output_root = tmp_path / "offline_twice"
    config_path = tmp_path / "offline_twice.yaml"
    config_path.write_text(
        json.dumps(
            {
                "music": {
                    "spectrum_sampling": {"samples_per_peak": 8, "local_grid_points_per_axis": 9},
                },
                "output": {"root": str(output_root)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    first = run_offline_demo(config_path)
    first_manifest = json.loads(
        (output_root / "localization" / "localization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    first_metrics = (output_root / "evaluation" / "metrics.json").read_bytes()
    second = run_offline_demo(config_path)

    assert first["result"]["localization_run_id"] == first_manifest["run_id"]
    assert second["result"]["localization_run_id"] != first_manifest["run_id"]
    history_dir = output_root / "evaluation" / "history" / first_manifest["run_id"]
    assert (history_dir / "metrics.json").read_bytes() == first_metrics
    assert (history_dir / "localization_manifest.json").is_file()
    assert (output_root / "evaluation" / "metrics.json").is_file()


def test_run_offline_demo_generation_failure_preserves_old_sources_in_history(
    tmp_path, monkeypatch
) -> None:
    output_root = tmp_path / "offline_generation_failure"
    config_path = tmp_path / "offline_generation_failure.yaml"
    config_path.write_text(
        json.dumps(
            {
                "music": {
                    "spectrum_sampling": {"samples_per_peak": 8, "local_grid_points_per_axis": 9},
                },
                "output": {"root": str(output_root)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    run_offline_demo(config_path)
    old_manifest = json.loads(
        (output_root / "localization" / "localization_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    old_run_id = old_manifest["run_id"]
    fixed_metrics_path = output_root / "evaluation" / "metrics.json"
    old_metrics = fixed_metrics_path.read_bytes()

    def destroy_fixed_scene_then_fail(config, root, *, allow_overwrite):
        scene_path = Path(root) / "scene" / "scene_2d.json"
        scene_path.write_bytes(b"destroyed-during-new-generation")
        raise RuntimeError("planned-generation-failure")

    monkeypatch.setattr(
        pipeline_module, "_prepare_scene_locked", destroy_fixed_scene_then_fail
    )
    with pytest.raises(RuntimeError, match="planned-generation-failure"):
        run_offline_demo(config_path)

    history_manifest_path = (
        output_root
        / "evaluation"
        / "history"
        / old_run_id
        / "localization_manifest.json"
    )
    archived_manifest = json.loads(
        history_manifest_path.read_text(encoding="utf-8")
    )
    for record in archived_manifest["archived_sources"].values():
        assert file_sha256(record["path"]) == record["sha256"]
    for record in archived_manifest["inputs"].values():
        assert file_sha256(record["path"]) == record["sha256"]
    assert fixed_metrics_path.read_bytes() == old_metrics


@pytest.mark.parametrize(
    "conflicting_name",
    ["metrics.json", "localization_manifest.json", "music_spectrum.npz"],
)
def test_localize_rejects_conflicting_history_without_overwriting_old_run(
    localized_run: dict[str, object], conflicting_name: str
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    manifest_path = root / "localization" / "localization_manifest.json"
    result_path = root / "localization" / "localization_result.json"
    metrics_path = root / "evaluation" / "metrics.json"
    evaluate(
        result_json=result_path,
        truth_npz=data["truth_npz"],
        output_json=metrics_path,
        expected_run_id=_current_run_id(result_path),
    )
    completed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_files = {
        "manifest": manifest_path.read_bytes(),
        "result": result_path.read_bytes(),
        "metrics": metrics_path.read_bytes(),
    }
    history_dir = root / "evaluation" / "history" / completed_manifest["run_id"]
    history_dir.mkdir(parents=True)
    conflicting_path = history_dir / conflicting_name
    conflicting_path.write_bytes(b"conflicting-history")

    with pytest.raises(ValueError, match="历史.*内容冲突"):
        localize(
            localization_config_view(config),
            scene_json=scene["scene_json"],
            online_input=data["online_npz"],
            output_root=root,
        )

    assert conflicting_path.read_bytes() == b"conflicting-history"
    assert manifest_path.read_bytes() == original_files["manifest"]
    assert result_path.read_bytes() == original_files["result"]
    assert metrics_path.read_bytes() == original_files["metrics"]


def test_localize_rejects_damaged_old_diagnostic_before_staging_new_run(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    manifest_path = root / "localization" / "localization_manifest.json"
    result_path = root / "localization" / "localization_result.json"
    metrics_path = root / "evaluation" / "metrics.json"
    evaluate(
        result_json=result_path,
        truth_npz=data["truth_npz"],
        output_json=metrics_path,
        expected_run_id=_current_run_id(result_path),
    )
    old_manifest = manifest_path.read_bytes()
    old_result = result_path.read_bytes()
    old_metrics = metrics_path.read_bytes()
    diagnostic_path = root / "localization" / "music_peaks.json"
    diagnostic_path.write_bytes(diagnostic_path.read_bytes() + b"damaged")
    damaged_diagnostic = diagnostic_path.read_bytes()

    with pytest.raises(ValueError, match="music_peaks.*路径或哈希不一致"):
        localize(
            localization_config_view(config),
            scene_json=scene["scene_json"],
            online_input=data["online_npz"],
            output_root=root,
        )

    assert manifest_path.read_bytes() == old_manifest
    assert result_path.read_bytes() == old_result
    assert metrics_path.read_bytes() == old_metrics
    assert diagnostic_path.read_bytes() == damaged_diagnostic
    assert list(root.glob(".localization-staging-*")) == []


def test_localize_rejects_stale_fixed_metrics_for_pending_run(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    config = localized_run["config"]
    scene = localized_run["scene"]
    data = localized_run["data"]
    assert isinstance(config, dict)
    assert isinstance(scene, dict)
    assert isinstance(data, dict)
    manifest_path = root / "localization" / "localization_manifest.json"
    result_path = root / "localization" / "localization_result.json"
    metrics_path = root / "evaluation" / "metrics.json"
    stale_metrics = b'{"misleading": true}'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_bytes(stale_metrics)
    old_manifest = manifest_path.read_bytes()
    old_result = result_path.read_bytes()

    with pytest.raises(ValueError, match="仍标记为待评估.*metrics.json 已存在"):
        localize(
            localization_config_view(config),
            scene_json=scene["scene_json"],
            online_input=data["online_npz"],
            output_root=root,
        )

    assert manifest_path.read_bytes() == old_manifest
    assert result_path.read_bytes() == old_result
    assert metrics_path.read_bytes() == stale_metrics


def test_evaluate_rejects_output_inside_history(
    localized_run: dict[str, object],
) -> None:
    root = Path(localized_run["root"])
    data = localized_run["data"]
    assert isinstance(data, dict)
    output_path = root / "evaluation" / "history" / "manual" / "metrics.json"

    with pytest.raises(ValueError, match="评估输出路径必须固定为"):
        evaluate(
            result_json=root / "localization" / "localization_result.json",
            truth_npz=data["truth_npz"],
            output_json=output_path,
            expected_run_id=_current_run_id(
                root / "localization" / "localization_result.json"
            ),
        )

    assert not output_path.exists()


def test_offline_generated_path_count_does_not_follow_signal_rank() -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["music"]["num_paths"] = 3
    config["music"]["signal_subspace_rank"] = 6
    scene = pipeline_module.make_synthetic_room()

    _, truth = generate_synthetic_measurement(scene, config)

    assert len(truth["paths"]) == 3
    assert truth["path_selection"]["requested_generated_path_count"] == 3
    assert truth["path_selection"]["retained_path_count_after_front_filter"] == 3
    assert truth["path_selection"]["front_facing_only"] is True


def test_localize_routes_signal_rank_to_single_music_preparation(
    tmp_path, monkeypatch
) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["music"]["num_paths"] = 3
    config["music"]["signal_subspace_rank"] = 6
    root = tmp_path / "rank_routing"
    scene_artifacts = prepare_scene(config, root)
    frequencies = np.arange(96, dtype=float) * (400.0e6 / 96.0)
    measurement_path = tmp_path / "measurement.npz"
    np.savez_compressed(
        measurement_path,
        csi_observed=np.zeros((1, 12, 96), dtype=np.complex128),
        subcarrier_frequencies_hz=frequencies,
        carrier_frequency_hz=np.asarray(3.5e9),
        antenna_spacing_m=np.asarray(299792458.0 / 3.5e9 * 0.5),
        bs_position_m=np.asarray([2.0, 7.0]),
        bs_boresight_rad=np.asarray(0.0),
    )
    truth_path = tmp_path / "ground_truth.npz"
    np.savez_compressed(
        truth_path,
        ue_position_m=np.asarray([0.0, 0.0]),
        clock_bias_s=np.asarray(0.0),
    )
    generation_manifest = {
        "schema_version": 2,
        "stage": "synthetic_csi_generation",
        "scene_json": str(Path(scene_artifacts["scene_json"]).resolve()),
        "online_input": str(measurement_path.resolve()),
        "truth_input": str(truth_path.resolve()),
        "separation_rule": "localization 只允许读取 online 目录",
        "link_direction": "uplink_ue_to_bs",
        "absolute_delay_normalization": False,
        "delay_convention": (
            "observed_delay=geometric_delay+common_bias+noise"
        ),
        "path_selection": {
            "rule": "planar_height_and_order_then_front_facing_local_angle_window",
            "front_facing_only": True,
            "bs_boresight_rad": 0.0,
            "local_angle_min_rad": math.radians(-89.0),
            "local_angle_max_rad": math.radians(89.0),
            "planar_path_count_before_front_filter": 3,
            "front_facing_angle_path_count": 3,
            "retained_path_count_after_front_filter": 3,
            "requested_generated_path_count": 3,
        },
        "rt_model": {
            "los": True,
            "specular_reflection": True,
            "max_reflections": 2,
            "diffraction": False,
            "diffuse_reflection": False,
            "transmission": False,
            "front_facing_only": True,
        },
        "artifact_hashes": {
            "scene_json": artifact_record(scene_artifacts["scene_json"]),
            "online_measurement": artifact_record(measurement_path),
            "ground_truth": artifact_record(truth_path),
        },
    }
    generation_manifest["bundle_id"] = generation_bundle_id(generation_manifest)
    generation_manifest_path = tmp_path / "generation_manifest.json"
    generation_manifest_path.write_text(
        json.dumps(generation_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    calls: dict[str, int] = {}

    def stop_after_prepare(self, csi, *, num_sources, **kwargs):
        calls["prepare"] = num_sources
        assert csi.shape == (1, 12, 96)
        np.testing.assert_array_equal(csi, np.zeros_like(csi))
        raise RuntimeError("rank-routing-complete")

    from time_bias_localization.compute import MusicComputer
    monkeypatch.setattr(MusicComputer, "prepare", stop_after_prepare)

    with pytest.raises(RuntimeError, match="rank-routing-complete"):
        localize(
            localization_config_view(config),
            scene_json=scene_artifacts["scene_json"],
            online_input=measurement_path,
            generation_manifest=generation_manifest_path,
            output_root=root,
        )

    assert calls == {"prepare": 6}


def test_initial_points_are_clustered_before_any_trajectory_is_constructed(tmp_path, monkeypatch) -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["output"]["root"] = str(tmp_path / "run")
    config["music"]["spectrum_sampling"].update(samples_per_peak=8, local_grid_points_per_axis=9)
    root = tmp_path / "run"
    scene = prepare_scene(config, root)
    data = generate_data(config, scene_json=scene["scene_json"], output_root=root)
    # 每条旧入口一旦被调用立即失败，验证主流程没有重新加噪或汇总扰动解。
    def reject_legacy_call(*args, **kwargs):
        raise AssertionError("点聚类主流程不应进入 CSI 扰动或先轨迹聚类支路")
    for name in ("estimate_noise_std_from_observed_csi", "estimate_music_peak_samples",
                 "_bootstrap_joint_solutions", "_distribution_statistics",
                 "generate_reverse_candidates", "cluster_reverse_candidates"):
        monkeypatch.setattr(pipeline_module, name, reject_legacy_call)
    observed = load_online_measurement(data["online_npz"]).csi_observed.copy()
    from time_bias_localization.compute import MusicComputer
    real_prepare = MusicComputer.prepare
    real_reverse = pipeline_module.generate_initial_candidate_points
    real_cluster = pipeline_module.cluster_initial_candidate_points
    real_build = pipeline_module.build_representative_trajectories
    real_solve = pipeline_module.solve_position_and_bias
    counts = {"prepare": 0, "solve": 0}
    recorded = {}
    stages = []
    def capture_prepare(self, csi, **kwargs):
        counts["prepare"] += 1
        np.testing.assert_array_equal(csi, observed)
        return real_prepare(self, csi, **kwargs)
    def capture_reverse(scene, bs, samples, **kwargs):
        stages.append("initial_points")
        recorded["samples"] = list(samples)
        output = real_reverse(scene, bs, recorded["samples"], **kwargs)
        recorded["points"] = output.points
        assert all(not hasattr(point, "anchor_m") and not hasattr(point, "beta_min_m")
                   for point in output.points)
        return output
    def capture_cluster(points, **kwargs):
        stages.append("point_clustering")
        assert points is recorded["points"]
        assert "direction_radius_deg" not in kwargs
        output = real_cluster(points, **kwargs)
        recorded["representatives"] = output.representatives
        recorded["clustering"] = output
        return output
    def capture_build(representatives, **kwargs):
        stages.append("representative_trajectories")
        assert representatives is recorded["representatives"]
        output = real_build(representatives, **kwargs)
        recorded["trajectories"] = output
        return output
    def capture_solve(trajectories, *args, **kwargs):
        stages.append("joint_solution")
        counts["solve"] += 1
        assert trajectories is recorded["trajectories"]
        return real_solve(trajectories, *args, **kwargs)
    monkeypatch.setattr(MusicComputer, "prepare", capture_prepare)
    monkeypatch.setattr(pipeline_module, "generate_initial_candidate_points", capture_reverse)
    monkeypatch.setattr(pipeline_module, "cluster_initial_candidate_points", capture_cluster)
    monkeypatch.setattr(pipeline_module, "build_representative_trajectories", capture_build)
    monkeypatch.setattr(pipeline_module, "solve_position_and_bias", capture_solve)
    result = localize(
        localization_config_view(config),
        scene_json=scene["scene_json"],
        online_input=data["online_npz"],
        output_root=root,
    )
    mu = np.asarray(result["mu_m"])
    sigma = np.asarray(result["sigma_m2"])
    assert np.linalg.norm(mu - np.asarray(config["simulation"]["ue_position_m"])) < 0.5
    assert sigma.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(sigma) >= -1e-12)
    serialized = json.loads(
        (root / "localization" / "localization_result.json").read_text(encoding="utf-8")
    )
    assert "accept" not in serialized
    assert "reject" not in serialized
    assert counts == {"prepare": 1, "solve": 1}
    assert len(recorded["samples"]) == 3 * (8 + 1)
    assert len({sample.sample_id for sample in recorded["samples"]}) == 27
    assert stages == ["initial_points", "point_clustering", "representative_trajectories", "joint_solution"]
    assert len(recorded["points"]) > len(recorded["representatives"])
    assert len(recorded["trajectories"]) == len(recorded["representatives"])
    assert any(len(candidate.members) > 1 for candidate in recorded["representatives"])
    np.testing.assert_array_equal(serialized["mu_m"], serialized["central_solution"]["mu_m"])
    assert serialized["clock_bias_s"] == serialized["central_solution"]["clock_bias_s"]
    assert (
        serialized["diagnostics"]["covariance_source"]
        == "selected_candidate_geometric_residual_approximation"
    )
    manifest = json.loads(
        (root / "localization" / "localization_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["truth_was_loaded"] is False
    generation_manifest = json.loads(
        (root / "data" / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert generation_manifest["path_selection"][
        "planar_path_count_before_front_filter"
    ] >= generation_manifest["path_selection"]["front_facing_angle_path_count"]
    assert generation_manifest["path_selection"][
        "retained_path_count_after_front_filter"
    ] == config["music"]["num_paths"]


def _music_peak(angle_rad: float, delay_s: float = 0.0) -> MusicPeak2D:
    return MusicPeak2D(
        aoa_rad=angle_rad,
        delay_s=delay_s,
        spectrum_value=1.0,
        aoa_index=0,
        delay_index=0,
    )


def test_missing_peak_assignment_uses_global_minimum_not_input_order() -> None:
    nominal = [_music_peak(0.0), _music_peak(0.1), _music_peak(1.0)]
    samples = MusicPeakSamples(
        aoa_rad=np.asarray([[0.04, 0.0]]),
        delay_s=np.asarray([[0.0, 0.0]]),
        spectrum_value=np.ones((1, 2)),
    )

    [association] = associate_perturbed_peaks(
        nominal,
        samples,
        angle_scale_rad=0.1,
        delay_scale_s=1.0,
    )

    # 逐峰贪心会先把 0.04 配给 nominal 0；整体最优应把精确的 0 留给它。
    assert association.matches[0][0] == 0.0
    assert association.matches[1][0] == 0.04
    assert 2 not in association.matches
    assert association.missed_nominal_indices == (2,)


def test_distant_false_peak_can_remain_unmatched() -> None:
    nominal = [_music_peak(0.0)]
    samples = MusicPeakSamples(
        aoa_rad=np.asarray([[1.0]]),
        delay_s=np.asarray([[0.0]]),
        spectrum_value=np.ones((1, 1)),
    )

    [association] = associate_perturbed_peaks(
        nominal,
        samples,
        angle_scale_rad=0.1,
        delay_scale_s=1.0,
        max_normalized_distance=2.0,
    )

    assert association.matches == {}
    assert association.unmatched_sample_indices == (0,)
    assert association.missed_nominal_indices == (0,)


def test_extra_false_peak_is_kept_as_unmatched_diagnostic() -> None:
    nominal = [_music_peak(0.0)]
    samples = MusicPeakSamples(
        aoa_rad=np.asarray([[0.0, 0.2]]),
        delay_s=np.asarray([[0.0, 0.0]]),
        spectrum_value=np.asarray([[10.0, 1.0]]),
    )

    [association] = associate_perturbed_peaks(
        nominal,
        samples,
        angle_scale_rad=0.1,
        delay_scale_s=1.0,
    )

    assert association.matches == {0: (0.0, 0.0)}
    assert association.unmatched_sample_indices == (1,)
    assert association.total_cost == 4.0


def test_distribution_statistics_do_not_count_nominal_or_add_it_twice() -> None:
    mu, sigma, beta_m, source = _distribution_statistics(
        central_mu=np.asarray([100.0, 100.0]),
        central_sigma=np.eye(2) * 10.0,
        central_beta_m=100.0,
        sampled_positions=np.asarray([[0.0, 0.0], [2.0, 0.0]]),
        sampled_betas_m=np.asarray([4.0, 6.0]),
        sampled_sigmas_m2=np.zeros((2, 2, 2)),
        sample_weights=np.asarray([1.0, 1.0]),
        requested_count=2,
    )

    assert np.allclose(mu, [1.0, 0.0])
    assert np.isclose(beta_m, 5.0)
    assert np.isclose(sigma[0, 0], 2.0)
    assert sigma[1, 1] < 1e-6
    assert source == "total_variance_of_quality_weighted_perturbation_solutions"


def test_single_success_falls_back_consistently_and_inflates_covariance() -> None:
    mu, sigma, beta_m, source = _distribution_statistics(
        central_mu=np.asarray([1.0, 2.0]),
        central_sigma=np.diag([2.0, 3.0]),
        central_beta_m=4.0,
        sampled_positions=np.asarray([[100.0, 100.0]]),
        sampled_betas_m=np.asarray([100.0]),
        sampled_sigmas_m2=np.asarray([np.eye(2)]),
        sample_weights=np.asarray([0.1]),
        requested_count=5,
    )

    assert np.allclose(mu, [1.0, 2.0])
    assert np.allclose(sigma, np.diag([10.0, 15.0]))
    assert beta_m == 4.0
    assert source == "nominal_fallback_single_solved_perturbation"


def test_distribution_uses_per_repeat_conditional_covariance_not_central_twice() -> None:
    _, sigma, _, _ = _distribution_statistics(
        central_mu=np.asarray([0.0, 0.0]),
        central_sigma=np.eye(2) * 100.0,
        central_beta_m=0.0,
        sampled_positions=np.asarray([[0.0, 0.0], [0.0, 0.0]]),
        sampled_betas_m=np.asarray([0.0, 0.0]),
        sampled_sigmas_m2=np.asarray([np.eye(2) * 2.0, np.eye(2) * 4.0]),
        sample_weights=np.asarray([1.0, 1.0]),
        requested_count=2,
    )

    assert np.allclose(sigma, np.eye(2) * 3.0)


def test_delay_search_rejects_periodic_music_aliases() -> None:
    music = deepcopy(DEFAULT_CONFIG["music"])
    frequencies = np.arange(96, dtype=float) * (400.0e6 / 96.0)
    music["delay_min_s"] = 0.0
    music["delay_max_s"] = 500.0e-9

    with pytest.raises(ValueError, match="无混叠周期"):
        _validate_unambiguous_delay_window(music, frequencies)

    music["delay_max_s"] = 230.0e-9
    period = _validate_unambiguous_delay_window(music, frequencies)
    assert np.isclose(period, 240.0e-9)
