from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from time_bias_localization.boundary_channel import (
    LegalRegion, load_probe, make_boundary_channel, save_probe, write_observation_bundle,
)
from time_bias_localization.config import DEFAULT_CONFIG, localization_config_view
from time_bias_localization.contracts import validate_generation_manifest_envelope, validate_localization_input_contract
from time_bias_localization.data import load_online_measurement
from time_bias_localization.scene import make_synthetic_room
from time_bias_localization.provenance import artifact_record
from time_bias_localization.sionna_generation import extract_planar_uplink_csi


def config():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["scene"].update(max_reflections=0, max_diffractions=0)
    cfg["radio"].update(num_subcarriers=16, num_bs_antennas=4)
    cfg["simulation"]["bs_position_m"] = [2.0, 7.0]
    # Deliberately unrelated to the one physically available LOS path.
    cfg["music"]["num_paths"] = 3
    cfg["music"].update(spatial_subarray_size=3, frequency_subarray_size=8)
    return cfg


def test_single_signal_retained_and_noise_repeats_share_exact_channel(tmp_path):
    cfg = config()
    provider = make_boundary_channel(cfg, tmp_path / "setup", backend="synthetic_fixture")
    probe = provider.probe([10.0, 7.0], seed=7)
    assert probe.status == "covered"
    assert probe.summary()["retained_path_count"] == 1
    assert probe.summary()["channel_category"] == "los"
    save_probe(probe, tmp_path / "frozen")
    reloaded = load_probe(tmp_path / "frozen")
    np.testing.assert_array_equal(reloaded.csi_geometric, probe.csi_geometric)
    a = provider.write_observation_bundle(reloaded, output_root=tmp_path / "a", noise_seed=101)
    provider.close()
    b = write_observation_bundle(reloaded, cfg, setup_root=tmp_path / "setup", output_root=tmp_path / "b", noise_seed=102)
    c = write_observation_bundle(reloaded, cfg, setup_root=tmp_path / "setup", output_root=tmp_path / "c", noise_seed=101)
    np.testing.assert_array_equal(load_online_measurement(a["online_npz"]).csi_observed,
                                  load_online_measurement(c["online_npz"]).csi_observed)
    assert not np.array_equal(load_online_measurement(a["online_npz"]).csi_observed,
                              load_online_measurement(b["online_npz"]).csi_observed)
    with np.load(a["truth_npz"]) as at, np.load(b["truth_npz"]) as bt:
        np.testing.assert_array_equal(at["csi_geometric"], bt["csi_geometric"])
    manifest = json.loads(open(a["generation_manifest"]).read())
    stage, _ = validate_generation_manifest_envelope(manifest)
    validate_localization_input_contract(manifest, stage, localization_config_view(cfg), provider.scene_2d,
                                         load_online_measurement(a["online_npz"]))
    with pytest.raises(FileExistsError):
        provider.write_observation_bundle(probe, output_root=tmp_path / "a", noise_seed=103)


def test_coverage_does_not_use_reverse_geometry_success(tmp_path, monkeypatch):
    import time_bias_localization.boundary_channel as module
    provider = make_boundary_channel(config(), tmp_path / "setup", backend="synthetic_fixture")
    def mismatch(*args, **kwargs):
        raise RuntimeError("deliberate map mismatch")
    monkeypatch.setattr(module, "_validate_reverse_scene_consistency", mismatch)
    probe = provider.probe([10.0, 7.0], 7)
    assert probe.status == "covered"
    assert probe.geometry_diagnostic["passed"] is False
    provider.write_observation_bundle(probe, output_root=tmp_path / "accepted", noise_seed=11)


def test_unknown_is_distinct_from_no_signal_and_illegal(tmp_path, monkeypatch):
    provider = make_boundary_channel(config(), tmp_path / "setup", backend="synthetic_fixture")
    assert provider.probe([1.0, 7.0], 7).status == "illegal"  # BS margin
    assert provider.probe([0.6, 12.0], 7).status == "no_signal"  # behind array
    def fail(*args, **kwargs):
        raise MemoryError("RT allocation failed")
    monkeypatch.setattr(provider, "_compute", fail)
    probe = provider.probe([10.0, 7.0], 7)
    assert probe.status == "unknown"
    assert "MemoryError" in probe.error


def test_frozen_channel_binding_rejects_changed_radio_or_another_setup(tmp_path):
    cfg = config()
    a = make_boundary_channel(cfg, tmp_path / "setup_a", backend="synthetic_fixture")
    b = make_boundary_channel(cfg, tmp_path / "setup_b", backend="synthetic_fixture")
    probe = a.probe([10.0, 7.0], 7)
    with pytest.raises(ValueError, match="同一公共信道"):
        b.write_observation_bundle(probe, output_root=tmp_path / "bad", noise_seed=8)
    changed = deepcopy(cfg)
    changed["radio"]["carrier_hz"] *= 2
    with pytest.raises(ValueError, match="无线配置"):
        a.write_observation_bundle(probe, config=changed, output_root=tmp_path / "bad2", noise_seed=8)


def test_legal_region_uses_actual_overhead_triangle_not_convex_hull():
    scene = make_synthetic_room(bounds_m=(0, 20, 0, 14), fixed_height_m=1.5, bev_resolution_m=.1)
    roof = np.asarray([[[5, 5, 5], [10, 5, 5], [5, 10, 5]]], dtype=float)
    region = LegalRegion(scene, [2, 7], overhead_triangles_m=roof)
    assert region.classify([6, 6]) == (False, "inside_public_overhead_footprint")
    assert region.classify([9, 9]) == (True, "legal")  # inside roof bbox but outside triangle
    below = roof.copy()
    below[:, :, 2] = 0
    assert LegalRegion(scene, [2, 7], overhead_triangles_m=below).classify([6, 6])[0]
    region = LegalRegion(scene, [2, 7], include_polygons=[[[3, 1], [8, 1], [8, 8], [3, 8]]],
                         exclude_polygons=[[[5, 4], [7, 4], [7, 6], [5, 6]]])
    assert region.classify([4, 2])[0]
    assert region.classify([6, 5])[1] == "inside_excluded_polygon"
    assert region.classify([10, 10])[1] == "outside_declared_polygons"


class FakePaths:
    def __init__(self, amplitudes):
        n = len(amplitudes)
        self.vertices = np.zeros((0, 1, 1, n, 3))
        self.interactions = np.zeros((0, 1, 1, n), dtype=int)
        self.phi_r = np.zeros((1, 1, n))
        self.tau = np.full((1, 1, n), 1e-7)
        self.amplitudes = np.asarray(amplitudes, dtype=complex)
    def cir(self, *, normalize_delays=False, **kwargs):
        assert normalize_delays is False
        return self.amplitudes.reshape(1, 1, 1, 1, -1, 1), self.tau


def test_sionna_boundary_filter_accepts_one_path_and_excludes_zero_nonfinite():
    frequencies = np.arange(8) * 1e6
    csi, meta = extract_planar_uplink_csi(FakePaths([1, 0, np.nan, np.inf]), frequencies,
                                         fixed_height_m=1.5, vertical_tolerance_m=.1, max_reflections=0,
                                         minimum_path_count=0, coefficient_zero_threshold=0.0)
    assert meta["retained_mask"].tolist() == [True, False, False, False]
    assert np.all(np.isfinite(csi))
    csi, meta = extract_planar_uplink_csi(FakePaths([]), frequencies,
                                         fixed_height_m=1.5, vertical_tolerance_m=.1, max_reflections=0,
                                         minimum_path_count=0, coefficient_zero_threshold=0.0)
    assert meta["retained_mask"].size == 0
    assert not np.any(csi)
    with pytest.raises(RuntimeError, match="少于两条"):
        extract_planar_uplink_csi(FakePaths([1]), frequencies,
                                  fixed_height_m=1.5, vertical_tolerance_m=.1, max_reflections=0)


@pytest.mark.parametrize("mutation,match", [
    (lambda setup: setup.update(backend="sionna"), "backend"),
    (lambda setup: setup["scene_fingerprint"].update(sha256="0" * 64), "场景指纹"),
    (lambda setup: setup["propagation"].update(max_reflections=1), "反射上限"),
    (lambda setup: setup.update(ue_position_m=[10, 7]), "禁止包含"),
])
def test_contract_rejects_changed_setup_even_with_recomputed_file_hash(tmp_path, mutation, match):
    provider = make_boundary_channel(config(), tmp_path / "setup", backend="synthetic_fixture")
    bundle = provider.write_observation_bundle(provider.probe([10, 7], 1), output_root=tmp_path / "bundle", noise_seed=2)
    manifest = json.loads(open(bundle["generation_manifest"]).read())
    setup_path = tmp_path / "changed_setup.json"
    setup = deepcopy(provider.provenance)
    mutation(setup)
    setup_path.write_text(json.dumps(setup))
    manifest["channel_setup"] = artifact_record(setup_path)
    with pytest.raises(ValueError, match=match):
        validate_generation_manifest_envelope(manifest)


@pytest.mark.parametrize("field,value", [("samples_per_src", 0), ("max_num_paths_per_src", -1), ("seed", True), ("seed", -1)])
def test_boundary_sionna_contract_rejects_invalid_rt_budget(tmp_path, field, value):
    provider = make_boundary_channel(config(), tmp_path / "setup", backend="synthetic_fixture")
    bundle = provider.write_observation_bundle(provider.probe([10, 7], 1), output_root=tmp_path / "bundle", noise_seed=2)
    manifest = json.loads(open(bundle["generation_manifest"]).read())
    manifest["stage"] = "sionna_rt_boundary_v1"
    manifest["rt_params"] = {"max_depth": 0, "los": True, "specular_reflection": True,
                              "diffuse_reflection": False, "diffraction": False, "refraction": False,
                              "samples_per_src": 10, "max_num_paths_per_src": 100,
                              "synthetic_array": True, "seed": 1}
    manifest["rt_params"][field] = value
    setup = deepcopy(provider.provenance)
    setup.update(backend="sionna", rt_parameters=manifest["rt_params"])
    setup_path = tmp_path / "sionna_setup.json"
    setup_path.write_text(json.dumps(setup))
    manifest["channel_setup"] = artifact_record(setup_path)
    with pytest.raises(ValueError, match="非负整数|正整数"):
        validate_generation_manifest_envelope(manifest)


@pytest.mark.parametrize("field,value", [("bs_position_m", [5, 7]), ("carrier_hz", 2e9)])
def test_contract_binds_setup_radio_and_bs_to_public_online_input(tmp_path, field, value):
    cfg = config()
    provider = make_boundary_channel(cfg, tmp_path / "setup", backend="synthetic_fixture")
    bundle = provider.write_observation_bundle(provider.probe([10, 7], 1), output_root=tmp_path / "bundle", noise_seed=2)
    manifest = json.loads(open(bundle["generation_manifest"]).read())
    setup = deepcopy(provider.provenance)
    if field == "bs_position_m":
        setup[field] = value
    else:
        setup["radio"][field] = value
    setup_path = tmp_path / "changed_setup.json"
    setup_path.write_text(json.dumps(setup))
    manifest["channel_setup"] = artifact_record(setup_path)
    with pytest.raises(ValueError, match="公开信道设置 BS|公开设置无线参数"):
        validate_localization_input_contract(manifest, manifest["stage"], localization_config_view(cfg), provider.scene_2d,
                                             load_online_measurement(bundle["online_npz"]))
