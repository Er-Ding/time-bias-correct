from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest

import time_bias_localization.data as data_module
from time_bias_localization.data import (
    OnlineMeasurement,
    load_online_measurement_bytes,
    save_measurement_bundle,
)


_TARGETS = (
    "online/measurement.npz",
    "online/manifest.json",
    "truth/ground_truth.npz",
    "truth/ground_truth.json",
)


def _bundle_inputs(version: int = 1) -> tuple[OnlineMeasurement, dict[str, object]]:
    online = OnlineMeasurement(
        csi_observed=np.full((version, 2, 8), version, dtype=np.complex128),
        subcarrier_frequencies_hz=np.arange(8, dtype=float) * 1.0e6,
        carrier_frequency_hz=3.5e9,
        antenna_spacing_m=0.04,
        bs_position_m=np.asarray([2.0, 7.0 + version]),
        bs_boresight_rad=0.0,
    )
    path = SimpleNamespace(
        path_id=f"los-v{version}",
        reflection_order=0,
        interaction_wall_ids=(),
        interaction_points_m=(),
        length_m=10.0 + version,
        delay_s=(10.0 + version) / 299792458.0,
        arrival_aoa_deg=0.0,
    )
    path_selection = {
        "rule": f"test-v{version}",
        "front_facing_only": True,
        "bs_boresight_rad": 0.0,
        "local_angle_min_rad": -1.0,
        "local_angle_max_rad": 1.0,
    }
    truth: dict[str, object] = {
        "ue_position_m": np.asarray([12.0 + version, 7.0]),
        "clock_bias_s": 25.0e-9,
        "distance_bias_m": 25.0e-9 * 299792458.0,
        "csi_geometric": np.full(
            (version, 2, 8), version, dtype=np.complex128
        ),
        "injected_noise_std": 0.01,
        "path_coefficients": np.full(
            (version, 1), version, dtype=np.complex128
        ),
        "paths": [path],
        "path_selection": path_selection,
    }
    return online, truth


def _formal_paths(root):
    return tuple(root / relative for relative in _TARGETS)


def _online_npz_bytes(**overrides) -> bytes:
    fields = {
        "csi_observed": np.ones((1, 2, 8), dtype=np.complex128),
        "subcarrier_frequencies_hz": np.arange(8, dtype=float) * 1.0e6,
        "carrier_frequency_hz": np.asarray(3.5e9),
        "antenna_spacing_m": np.asarray(0.04),
        "bs_position_m": np.asarray([2.0, 7.0]),
        "bs_boresight_rad": np.asarray(0.0),
    }
    fields.update(overrides)
    stream = BytesIO()
    np.savez_compressed(stream, **fields)
    return stream.getvalue()


def test_measurement_bundle_success_writes_readable_complete_files(tmp_path) -> None:
    online, truth = _bundle_inputs()

    artifacts = save_measurement_bundle(tmp_path, online, truth)

    assert {path for path in artifacts.values()} == {
        str(path) for path in _formal_paths(tmp_path)
    }
    with np.load(tmp_path / "online" / "measurement.npz") as saved_online:
        assert saved_online["csi_observed"].shape == (1, 2, 8)
    with np.load(tmp_path / "truth" / "ground_truth.npz") as saved_truth:
        assert saved_truth["ue_position_m"].shape == (2,)
    assert (tmp_path / "online" / "manifest.json").read_text(encoding="utf-8")
    assert (tmp_path / "truth" / "ground_truth.json").read_text(encoding="utf-8")
    assert not tuple(tmp_path.rglob(".*.tmp.*"))


@pytest.mark.parametrize("conflicting_relative", _TARGETS)
def test_measurement_bundle_single_conflict_rejects_before_any_write(
    tmp_path, conflicting_relative: str
) -> None:
    online, truth = _bundle_inputs()
    conflicting_path = tmp_path / conflicting_relative
    conflicting_path.parent.mkdir(parents=True, exist_ok=True)
    old_bytes = b"preserve-existing-file-byte-for-byte"
    conflicting_path.write_bytes(old_bytes)

    with pytest.raises(FileExistsError, match="拒绝覆盖整组文件"):
        save_measurement_bundle(tmp_path, online, truth)

    assert conflicting_path.read_bytes() == old_bytes
    assert all(
        path == conflicting_path or not path.exists()
        for path in _formal_paths(tmp_path)
    )


def test_measurement_bundle_rerun_keeps_every_old_file_unchanged(tmp_path) -> None:
    online, truth = _bundle_inputs()
    save_measurement_bundle(tmp_path, online, truth)
    before = {path: path.read_bytes() for path in _formal_paths(tmp_path)}

    with pytest.raises(FileExistsError, match="拒绝覆盖整组文件"):
        save_measurement_bundle(tmp_path, online, truth)

    assert {path: path.read_bytes() for path in before} == before


def test_measurement_bundle_treats_broken_symlink_as_existing_target(tmp_path) -> None:
    online, truth = _bundle_inputs()
    target = tmp_path / "online" / "measurement.npz"
    target.parent.mkdir(parents=True)
    target.symlink_to(tmp_path / "missing-source.npz")

    with pytest.raises(FileExistsError, match="拒绝覆盖整组文件"):
        save_measurement_bundle(tmp_path, online, truth)

    assert target.is_symlink()
    assert all(path == target or not path.exists() for path in _formal_paths(tmp_path))


def test_measurement_bundle_explicit_overwrite_replaces_complete_group(tmp_path) -> None:
    first_online, first_truth = _bundle_inputs(version=1)
    second_online, second_truth = _bundle_inputs(version=2)
    save_measurement_bundle(tmp_path, first_online, first_truth)
    before = {path: path.read_bytes() for path in _formal_paths(tmp_path)}

    save_measurement_bundle(
        tmp_path,
        second_online,
        second_truth,
        allow_overwrite=True,
    )

    after = {path: path.read_bytes() for path in _formal_paths(tmp_path)}
    assert all(after[path] != before[path] for path in before)
    with np.load(tmp_path / "online" / "measurement.npz") as saved_online:
        assert saved_online["csi_observed"].shape == (2, 2, 8)
        assert np.all(saved_online["csi_observed"] == 2)
    with np.load(tmp_path / "truth" / "ground_truth.npz") as saved_truth:
        assert np.allclose(saved_truth["ue_position_m"], [14.0, 7.0])
    assert not tuple(tmp_path.rglob(".*.tmp.*"))
    assert not tuple(tmp_path.rglob(".*.backup.*"))


def test_measurement_bundle_overwrite_publish_failure_restores_old_group(
    tmp_path, monkeypatch
) -> None:
    first_online, first_truth = _bundle_inputs(version=1)
    second_online, second_truth = _bundle_inputs(version=2)
    save_measurement_bundle(tmp_path, first_online, first_truth)
    formal_paths = set(_formal_paths(tmp_path))
    before = {path: path.read_bytes() for path in formal_paths}
    real_replace = data_module.os.replace
    publish_count = 0

    def fail_second_new_publish(source, destination):
        nonlocal publish_count
        source_path = data_module.Path(source)
        destination_path = data_module.Path(destination)
        if destination_path in formal_paths and ".tmp" in source_path.name:
            publish_count += 1
            if publish_count == 2:
                raise OSError("planned-overwrite-publish-failure")
        return real_replace(source, destination)

    monkeypatch.setattr(data_module.os, "replace", fail_second_new_publish)

    with pytest.raises(OSError, match="planned-overwrite-publish-failure"):
        save_measurement_bundle(
            tmp_path,
            second_online,
            second_truth,
            allow_overwrite=True,
        )

    assert {path: path.read_bytes() for path in formal_paths} == before
    assert tuple(tmp_path.rglob(".*.tmp.*")), "失败临时文件应保留用于排查"
    assert not tuple(tmp_path.rglob(".*.backup.*"))


def test_measurement_bundle_stage_failure_leaves_no_formal_target(
    tmp_path, monkeypatch
) -> None:
    online, truth = _bundle_inputs()
    real_savez = data_module.np.savez_compressed
    call_count = 0

    def fail_second_npz(handle, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            handle.write(b"partial-truth-npz")
            raise OSError("planned-npz-write-failure")
        return real_savez(handle, *args, **kwargs)

    monkeypatch.setattr(data_module.np, "savez_compressed", fail_second_npz)

    with pytest.raises(OSError, match="planned-npz-write-failure"):
        save_measurement_bundle(tmp_path, online, truth)

    assert all(not path.exists() for path in _formal_paths(tmp_path))
    assert tuple(tmp_path.rglob(".*.tmp.*")), "失败临时文件应保留用于排查"


def test_measurement_bundle_json_stage_failure_leaves_no_formal_target(
    tmp_path, monkeypatch
) -> None:
    online, truth = _bundle_inputs()
    real_stage = data_module._stage_binary_file

    def fail_online_manifest(path, writer):
        if path.name == "manifest.json":
            def write_partial_json_then_fail(handle):
                handle.write(b'{"partial":')
                raise OSError("planned-json-write-failure")

            return real_stage(path, write_partial_json_then_fail)
        return real_stage(path, writer)

    monkeypatch.setattr(data_module, "_stage_binary_file", fail_online_manifest)

    with pytest.raises(OSError, match="planned-json-write-failure"):
        save_measurement_bundle(tmp_path, online, truth)

    assert all(not path.exists() for path in _formal_paths(tmp_path))
    assert tuple(tmp_path.rglob(".*.tmp.*")), "失败临时文件应保留用于排查"


def test_measurement_bundle_publish_failure_rolls_back_formal_targets(
    tmp_path, monkeypatch
) -> None:
    online, truth = _bundle_inputs()
    real_replace = data_module.os.replace
    publish_count = 0
    formal_paths = set(_formal_paths(tmp_path))

    def fail_second_publish(source, destination):
        nonlocal publish_count
        destination_path = data_module.Path(destination)
        if destination_path in formal_paths:
            publish_count += 1
            if publish_count == 2:
                raise OSError("planned-publish-failure")
        return real_replace(source, destination)

    monkeypatch.setattr(data_module.os, "replace", fail_second_publish)

    with pytest.raises(OSError, match="planned-publish-failure"):
        save_measurement_bundle(tmp_path, online, truth)

    assert all(not path.exists() for path in formal_paths)
    assert tuple(tmp_path.rglob(".*.tmp.*")), "失败临时文件应保留用于排查"


def test_load_online_measurement_bytes_accepts_exact_shapes_and_finite_values(
    tmp_path,
) -> None:
    measurement = load_online_measurement_bytes(
        _online_npz_bytes(),
        source_path=tmp_path / "online" / "measurement.npz",
    )

    assert measurement.csi_observed.shape == (1, 2, 8)
    assert measurement.subcarrier_frequencies_hz.shape == (8,)
    assert measurement.bs_position_m.shape == (2,)
    assert measurement.carrier_frequency_hz == 3.5e9


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("csi_observed", np.ones((2, 8), dtype=np.complex128)),
        ("csi_observed", np.ones((1, 2, 8, 1), dtype=np.complex128)),
        ("subcarrier_frequencies_hz", np.ones((1, 8), dtype=float)),
        ("carrier_frequency_hz", np.asarray([3.5e9])),
        ("antenna_spacing_m", np.asarray([0.04])),
        ("bs_position_m", np.asarray([[2.0, 7.0]])),
        ("bs_boresight_rad", np.asarray([0.0])),
    ),
)
def test_load_online_measurement_bytes_rejects_wrong_raw_shape(
    tmp_path, field_name: str, bad_value
) -> None:
    with pytest.raises(ValueError, match=field_name):
        load_online_measurement_bytes(
            _online_npz_bytes(**{field_name: bad_value}),
            source_path=tmp_path / "online" / "measurement.npz",
        )


_CSI_NONFINITE_REAL = np.ones((1, 2, 8), dtype=np.complex128)
_CSI_NONFINITE_REAL[0, 0, 0] = complex(np.nan, 0.0)
_CSI_NONFINITE_IMAG = np.ones((1, 2, 8), dtype=np.complex128)
_CSI_NONFINITE_IMAG[0, 0, 0] = complex(1.0, np.inf)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("csi_observed", _CSI_NONFINITE_REAL),
        ("csi_observed", _CSI_NONFINITE_IMAG),
        (
            "subcarrier_frequencies_hz",
            np.asarray([0.0, 1.0e6, np.nan]),
        ),
        ("carrier_frequency_hz", np.asarray(np.inf)),
        ("antenna_spacing_m", np.asarray(np.nan)),
        ("bs_position_m", np.asarray([2.0, np.inf])),
        ("bs_boresight_rad", np.asarray(np.nan)),
    ),
)
def test_load_online_measurement_bytes_rejects_nonfinite_values(
    tmp_path, field_name: str, bad_value
) -> None:
    with pytest.raises(ValueError, match=field_name):
        load_online_measurement_bytes(
            _online_npz_bytes(**{field_name: bad_value}),
            source_path=tmp_path / "online" / "measurement.npz",
        )
