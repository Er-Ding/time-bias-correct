"""验证新主流程真正绕过旧空间候选链，以及观测/结果来源隔离。"""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from time_bias_localization.config import DEFAULT_CONFIG, localization_config_view
from time_bias_localization.continuous_config import continuous_settings
from time_bias_localization.continuous_pipeline import localize_saved_music
from time_bias_localization.provenance import artifact_record
from time_bias_localization.spectrum_sampling import sample_music_spectrum, refine_music_peaks
from time_bias_localization.signal import MusicPeak2D


class FineSpectrum:
    def spectrum(self, *, aoa_grid_rad, delay_grid_s):
        return 1 + np.exp(-((np.asarray(aoa_grid_rad)[:, None]-.12)/.07)**2
                          - ((np.asarray(delay_grid_s)[None, :]-9e-8)/1e-8)**2)

    def values(self, *, aoa_rad, delay_s):
        return np.ones_like(aoa_rad)

    def metadata(self):
        return {}


def test_refinement_preserves_peak_but_never_generates_random_samples(monkeypatch):
    kwargs = dict(aoa_grid_rad=np.linspace(-.5,.5,21), delay_grid_s=np.linspace(0,2e-7,21),
                  bs_boresight_rad=0., settings={"samples_per_peak":8,"local_grid_points_per_axis":9}, seed=3)
    peaks=[MusicPeak2D(.1,9e-8,1.,12,9)]
    old=sample_music_spectrum(FineSpectrum(),peaks,**kwargs)
    monkeypatch.setattr(np.random,"default_rng",lambda *a,**k:pytest.fail("连续峰位细化不能生成随机样本"))
    new=refine_music_peaks(FineSpectrum(),peaks,**kwargs)
    assert new.refined_peaks==old.refined_peaks
    assert new.refined_peak_source_indices==old.refined_peak_source_indices
    assert not new.samples and not new.records and not new.diagnostics["sampling_performed"]
    changed=refine_music_peaks(FineSpectrum(),peaks,**{**kwargs,"settings":{
        "samples_per_peak":8192,"uniform_mixture":.99,"local_grid_points_per_axis":9},"seed":777})
    assert changed.refined_peaks==new.refined_peaks


@pytest.mark.parametrize("settings", [{"truth_bias_s":0}, {"angle_scale_deg":0},
    {"max_starts":True}, {"max_hypotheses":1.5}, {"length_scale_m":float("nan")}])
def test_continuous_settings_reject_unknown_or_invalid_inputs(settings):
    with pytest.raises(ValueError):
        continuous_settings(settings)


def small_config(root):
    config=deepcopy(DEFAULT_CONFIG)
    config["scene"].update(max_reflections=1,max_diffractions=0)
    config["simulation"]["path_amplitudes"]=[1.,.8,.6]
    config["radio"]["snr_db"]=80.
    config["music"]["spectrum_sampling"].update(samples_per_peak=4,local_grid_points_per_axis=9)
    config["localization"].update(solver_method="continuous",require_identifiable_solution=True,
        continuous={"max_hypotheses":64,"max_enumerated_sequences":128,
                    "max_starts":16,"max_seed_combinations":128,"max_iterations":40,
                    "ambiguity_cost_tolerance":.001})
    config["output"]["root"]=str(root)
    return config


def test_csi_pipeline_never_calls_legacy_spatial_stages_and_publishes_new_manifest(tmp_path,monkeypatch):
    import time_bias_localization.pipeline as pipeline
    import time_bias_localization.bias_interval_candidates as interval
    config=small_config(tmp_path)
    scene=pipeline.prepare_scene(config,tmp_path)
    bundle=pipeline.generate_data(config,scene_json=scene["scene_json"],output_root=tmp_path)
    # 在线入口即使所有真值文件都不存在，也必须独立工作。
    from pathlib import Path
    Path(bundle["truth_npz"]).unlink()
    Path(bundle["truth_json"]).unlink()
    def forbidden(*args,**kwargs):
        pytest.fail("连续主流程调用了旧采样/候选点/聚类/代表/RANSAC")
    for name in ("sample_music_spectrum","generate_initial_candidate_points",
                 "cluster_initial_candidate_points","build_representative_trajectories",
                 "solve_position_and_bias"):
        monkeypatch.setattr(pipeline,name,forbidden)
    monkeypatch.setattr(interval,"generate_bias_interval_points",forbidden)
    result=pipeline.localize(localization_config_view(config),scene_json=scene["scene_json"],
                             online_input=bundle["online_npz"],output_root=tmp_path)
    assert result["status"]=="success",result
    assert result["workflow"]=="music_continuous_propagation_v1"
    assert result["forward_check"]["all_selected_paths_valid"]
    manifest=json.loads((tmp_path/'localization/localization_manifest.json').read_text())
    assert manifest["schema_version"]==8 and not manifest["truth_was_loaded"]
    assert {"continuous_observations","propagation_hypotheses","continuous_search"} <= set(manifest["artifacts"])
    assert not {"initial_candidates","representative_points","representative_trajectories","spectrum_samples"} & set(manifest["artifacts"])
    assert result["scientific_validation_status"]=="not_validated"


def test_unavailable_csi_run_keeps_continuous_workflow_and_diagnostics(tmp_path, monkeypatch):
    import time_bias_localization.pipeline as pipeline
    import time_bias_localization.continuous_pipeline as continuous
    config = small_config(tmp_path)
    scene = pipeline.prepare_scene(config, tmp_path)
    bundle = pipeline.generate_data(config, scene_json=scene["scene_json"], output_root=tmp_path)

    def ambiguous(*args, **kwargs):
        result = {"workflow": continuous.WORKFLOW, "status": "ambiguous", "reason": "ambiguous",
                  "mu_m": None, "diagnostics": {"test_marker": "two_legal_solutions"},
                  "alternatives": [{"position_m": [3., 4.]}, {"position_m": [5., 6.]}]}
        return result, {"continuous_search": result}, None, None

    monkeypatch.setattr(continuous, "run_continuous_from_peaks", ambiguous)
    result = pipeline.localize(localization_config_view(config), scene_json=scene["scene_json"],
                               online_input=bundle["online_npz"], output_root=tmp_path)
    assert result["status"] == "ambiguous"
    assert result["workflow"] == continuous.WORKFLOW
    assert result["mu_m"] is None
    assert len(result["alternatives"]) == 2
    assert len(result["candidate_solution_for_diagnostics_only"]["alternatives"]) == 2
    from pathlib import Path
    progress = json.loads(Path(result["progress_path"]).read_text())
    saved = json.loads(Path(progress["artifacts"]["result"]["path"]).read_text())
    assert saved["workflow"] == continuous.WORKFLOW
    assert "continuous_search" in progress["artifacts"]
    assert not (tmp_path / "localization/localization_result.json").exists()


def test_saved_music_provenance_checked_and_old_output_never_overwritten(tmp_path, monkeypatch):
    from time_bias_localization.scene import make_synthetic_room
    from time_bias_localization.raytrace2d import enumerate_specular_paths
    from time_bias_localization.constants import SPEED_OF_LIGHT_M_S as C
    scene=make_synthetic_room()
    bs=np.array([2.,7.]); position=np.array([13.3,4.7]); beta=3.4
    paths=enumerate_specular_paths(scene,position,bs,max_reflections=1)
    assert len(paths)>2
    source=tmp_path/'source';source.mkdir()
    scene_path=source/'scene.json';scene_path.write_text(json.dumps(scene.to_dict()))
    peaks_path=source/'music_peaks.json'
    peaks_path.write_text(json.dumps({"nominal":[{"aoa_rad":np.deg2rad(p.arrival_aoa_deg),
        "delay_s":(p.length_m+beta)/C} for p in paths],"nominal_source_indices":list(range(len(paths)))}))
    manifest_path=source/'localization_manifest.json'
    manifest_path.write_text(json.dumps({"artifacts":{"music_peaks":artifact_record(peaks_path)},
                                       "inputs":{"scene":artifact_record(scene_path)}}))
    config=localization_config_view(small_config(tmp_path/'new'))
    result=localize_saved_music(config,scene_json=scene_path,music_peaks_json=peaks_path,
                                output_root=tmp_path/'new',source_manifest=manifest_path)
    assert result["status"]=="success",result
    np.testing.assert_allclose(result["mu_m"],position,atol=1e-5)
    assert result["distance_bias_m"]==pytest.approx(beta,abs=1e-5)
    with pytest.raises(FileExistsError):
        localize_saved_music(config,scene_json=scene_path,music_peaks_json=peaks_path,
                              output_root=tmp_path/'new',source_manifest=manifest_path)
    original_peaks = peaks_path.read_text()
    peaks_path.write_text(original_peaks+' ')
    with pytest.raises(ValueError,match="哈希"):
        localize_saved_music(config,scene_json=scene_path,music_peaks_json=peaks_path,
                              output_root=tmp_path/'tampered',source_manifest=manifest_path)
    peaks_path.write_text(original_peaks)
    import time_bias_localization.continuous_solver as solver
    def stop_after_model(*args, **kwargs):
        raise RuntimeError("test_solver_interruption")
    monkeypatch.setattr(solver, "solve_continuous_position_and_bias", stop_after_model)
    with pytest.raises(RuntimeError, match="test_solver_interruption"):
        localize_saved_music(config, scene_json=scene_path, music_peaks_json=peaks_path,
                              output_root=tmp_path/'interrupted', source_manifest=manifest_path)
    partial = tmp_path / 'interrupted/localization'
    assert (partial / 'continuous_observations.json').is_file()
    assert (partial / 'propagation_hypotheses.json').is_file()
    assert not (partial / 'localization_result.json').exists()
