from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from time_bias_localization.adapters import (
    AdapterContractError,
    DeepMIMOAdapterConfig,
    OptionalDependencyError,
    SionnaPathExportConfig,
    absolute_delays,
    absolute_delays_from_path_lengths,
    convert_sionna_paths_to_deepmimo,
    downlink_to_uplink_csi,
    export_sionna_paths,
    extract_deepmimo_dataset,
    frequency_response_from_paths,
    load_deepmimo_module,
    load_sionna_rt_module,
    sionna_paths_cfr,
    sionna_paths_cir,
)


def _fake_dataset() -> tuple[SimpleNamespace, object]:
    scene = object()
    paths = SimpleNamespace(
        delays=np.array([[2.0e-9, 5.0e-9], [3.0e-9, np.nan]]),
        aoa_az=np.array([[0.1, 0.2], [0.3, np.nan]]),
        aod_az=np.array([[1.1, 1.2], [1.3, np.nan]]),
        interactions=np.array(
            [
                [[0, -1], [1, 2]],
                [[0, -1], [-1, -1]],
            ]
        ),
        interaction_positions=np.array(
            [
                [
                    [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                ],
                [
                    [[7.0, 8.0, 9.0], [np.nan, np.nan, np.nan]],
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ],
                ],
            ]
        ),
    )
    dataset = SimpleNamespace(
        receiver_positions=np.array([[10.0, 11.0, 1.5], [12.0, 13.0, 1.5]]),
        bs_pos=np.array([0.0, 0.0, 1.5]),
        paths=paths,
        metadata={"scenario": scene},
    )
    return dataset, scene


def test_extract_dataset_aliases_shapes_scene_and_link_semantics() -> None:
    dataset, scene = _fake_dataset()

    result = extract_deepmimo_dataset(
        dataset,
        DeepMIMOAdapterConfig(
            link_direction="uplink",
            angle_unit="rad",
            azimuth_convention="global +x counter-clockwise",
        ),
    )

    assert result.rx_pos.shape == (2, 3)
    assert result.tx_pos.shape == (2, 3)
    np.testing.assert_array_equal(result.tx_pos[0], [0.0, 0.0, 1.5])
    np.testing.assert_array_equal(result.tx_pos[1], [0.0, 0.0, 1.5])
    assert result.delay.shape == (2, 2)
    assert result.inter.shape == (2, 2, 2)
    assert result.inter_pos.shape == (2, 2, 2, 3)
    assert result.scene is scene
    assert result.link_direction == "uplink"
    assert result.transmitter_role == "ue"
    assert result.receiver_role == "bs"
    assert result.aoa_az_semantics == "arrival_at_receiver"
    assert result.aod_az_semantics == "departure_from_transmitter"
    assert result.delays_are_absolute is True
    np.testing.assert_array_equal(
        result.valid_path_mask,
        np.array([[True, True], [True, False]]),
    )


def test_custom_alias_and_single_interaction_axes_are_normalized() -> None:
    scene = object()
    dataset = {
        "receivers": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "transmitter": np.array([[0.0, 0.0]]),
        "tau": np.array([[1.0e-9], [2.0e-9]]),
        "arrival": np.array([[0.1], [0.2]]),
        "departure": np.array([[0.3], [0.4]]),
        "bounce_type": np.array([[0], [1]]),
        "bounce_position": np.array([[[0.0, 0.0]], [[1.0, 1.0]]]),
        "world": scene,
    }
    config = DeepMIMOAdapterConfig(
        field_aliases={
            "rx_pos": ("receivers",),
            "tx_pos": ("transmitter",),
            "delay": ("tau",),
            "aoa_az": ("arrival",),
            "aod_az": ("departure",),
            "inter": ("bounce_type",),
            "inter_pos": ("bounce_position",),
            "scene": ("world",),
        }
    )

    result = extract_deepmimo_dataset(dataset, config)

    assert result.inter.shape == (2, 1, 1)
    assert result.inter_pos.shape == (2, 1, 1, 2)
    assert result.scene is scene


def test_shape_validation_rejects_mismatched_path_arrays() -> None:
    dataset, _ = _fake_dataset()
    dataset.paths.aoa_az = np.zeros((2, 3))

    with pytest.raises(AdapterContractError, match="必须与 delay 形状一致"):
        extract_deepmimo_dataset(dataset)


def test_multibase_station_dimension_is_not_guessed() -> None:
    dataset, _ = _fake_dataset()
    dataset.bs_pos = np.zeros((2, 2, 3))

    with pytest.raises(AdapterContractError, match="tx_pos 必须"):
        extract_deepmimo_dataset(dataset)


def test_absolute_delay_is_never_shifted_to_first_path() -> None:
    dataset, _ = _fake_dataset()
    result = extract_deepmimo_dataset(dataset)

    copied = absolute_delays(result)

    np.testing.assert_allclose(copied[0], [2.0e-9, 5.0e-9])
    assert copied[0, 0] != 0.0
    copied[0, 0] = 123.0
    assert result.delay[0, 0] == pytest.approx(2.0e-9)


def test_absolute_delay_generation_and_frequency_response_keep_common_delay() -> None:
    speed = 2.0
    delays = absolute_delays_from_path_lengths([[2.0, 4.0]], speed)
    gains = np.array([[1.0 + 0.0j, 0.5 + 0.0j]])
    frequencies = np.array([0.0, 0.25])

    response = frequency_response_from_paths(gains, delays, frequencies)
    expected = np.sum(
        gains[..., np.newaxis]
        * np.exp(-2j * np.pi * delays[..., np.newaxis] * frequencies),
        axis=1,
    )

    np.testing.assert_allclose(delays, [[1.0, 2.0]])
    np.testing.assert_allclose(response, expected)


class _FakeSionnaPaths:
    def __init__(self) -> None:
        self.cir_kwargs: dict[str, object] | None = None
        self.cfr_kwargs: dict[str, object] | None = None

    def cir(self, **kwargs: object) -> str:
        self.cir_kwargs = dict(kwargs)
        return "cir-result"

    def cfr(self, **kwargs: object) -> str:
        self.cfr_kwargs = dict(kwargs)
        return "cfr-result"


def test_sionna_cir_and_cfr_force_absolute_delays() -> None:
    paths = _FakeSionnaPaths()

    assert sionna_paths_cir(paths, out_type="numpy") == "cir-result"
    assert paths.cir_kwargs == {"out_type": "numpy", "normalize_delays": False}

    frequencies = np.array([-1.0e6, 0.0, 1.0e6])
    assert sionna_paths_cfr(paths, frequencies, out_type="numpy") == "cfr-result"
    assert paths.cfr_kwargs is not None
    assert paths.cfr_kwargs["normalize_delays"] is False
    np.testing.assert_array_equal(paths.cfr_kwargs["frequencies"], frequencies)


def test_sionna_wrapper_rejects_delay_normalization() -> None:
    paths = _FakeSionnaPaths()

    with pytest.raises(AdapterContractError, match="禁止 normalize_delays=True"):
        sionna_paths_cir(paths, normalize_delays=True)

    with pytest.raises(AdapterContractError, match="必须提供 frequencies_hz"):
        export_sionna_paths(
            paths,
            SionnaPathExportConfig(representation="cfr"),
        )


def test_converter_can_be_injected_without_deepmimo_installation() -> None:
    paths = object()
    scene = object()
    calls: list[tuple[object, object, int]] = []

    def converter(input_paths: object, *, scene: object, max_paths: int) -> str:
        calls.append((input_paths, scene, max_paths))
        return "deepmimo-dataset"

    result = convert_sionna_paths_to_deepmimo(
        paths,
        scene=scene,
        converter=converter,
        max_paths=8,
    )

    assert result == "deepmimo-dataset"
    assert calls == [(paths, scene, 8)]


def test_optional_dependencies_are_imported_lazily_with_chinese_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def missing(name: str, package: str | None = None) -> object:
        if name in {"deepmimo", "sionna.rt"}:
            raise ModuleNotFoundError(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(OptionalDependencyError, match="缺少 DeepMIMO V4"):
        load_deepmimo_module()
    with pytest.raises(OptionalDependencyError, match="缺少 Sionna RT"):
        load_sionna_rt_module()


def test_downlink_to_uplink_is_conjugate_transpose() -> None:
    downlink = np.array(
        [
            [1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j],
            [7.0 + 8.0j, 9.0 + 10.0j, 11.0 + 12.0j],
        ]
    )

    uplink = downlink_to_uplink_csi(downlink)

    assert uplink.shape == (3, 2)
    np.testing.assert_array_equal(uplink, downlink.conj().T)


def test_downlink_to_uplink_preserves_non_antenna_axes() -> None:
    downlink = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(complex)
    downlink += 1j * (downlink + 1.0)

    uplink = downlink_to_uplink_csi(
        downlink,
        receive_axis=1,
        transmit_axis=2,
    )

    assert uplink.shape == (2, 4, 3, 5)
    np.testing.assert_array_equal(uplink, np.swapaxes(downlink.conj(), 1, 2))
