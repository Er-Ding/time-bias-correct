"""端射边界伪峰剔除：判据、余量补回、以及不该误删的情形。"""
import math

import numpy as np
import pytest

from time_bias_localization.signal import MusicPeak2D
from time_bias_localization.spurious_peaks import (
    DEFAULT_SPURIOUS_PEAK_FILTER, filter_boundary_mirror_peaks, spurious_peak_filter_settings,
)


GRID = np.deg2rad(np.arange(-89.0, 89.5, 1.0))


def peak(aoa_deg, delay_ns, value, index=0):
    return MusicPeak2D(math.radians(aoa_deg), delay_ns * 1e-9, value, index, index)


def run(peaks, *, keep_count=None, **settings):
    return filter_boundary_mirror_peaks(
        peaks, list(range(len(peaks))), aoa_grid_rad=GRID,
        settings=spurious_peak_filter_settings({"enabled": True, **settings}),
        keep_count=keep_count)


def test_disabled_filter_returns_input_unchanged():
    peaks = [peak(89.0, 331.0, 1e3), peak(-86.6, 331.0, 1e5)]
    result = filter_boundary_mirror_peaks(peaks, [0, 1], aoa_grid_rad=GRID,
                                         settings=spurious_peak_filter_settings(None))
    assert result.peaks == tuple(peaks) and result.report["enabled"] is False
    assert result.report["removed_count"] == 0


def test_boundary_peak_weaker_at_same_delay_is_removed():
    peaks = [peak(89.0, 331.0, 1e3), peak(-86.6, 331.0, 1e5)]
    result = run(peaks)
    assert [p.aoa_rad for p in result.peaks] == [math.radians(-86.6)]
    removed, = result.report["removed"]
    assert removed["removed_aoa_deg"] == pytest.approx(89.0)
    assert removed["explaining_aoa_deg"] == pytest.approx(-86.6)
    assert removed["spectrum_ratio"] == pytest.approx(0.01)
    assert result.report["kept_count"] == 1


def test_boundary_peak_is_kept_when_no_stronger_peak_shares_its_delay():
    # 端射附近确实存在的强峰：没有同延的更强峰来解释它，不能删。
    peaks = [peak(89.0, 331.0, 1e5), peak(10.0, 700.0, 2e4)]
    result = run(peaks)
    assert len(result.peaks) == 2 and result.report["removed_count"] == 0


def test_boundary_peak_is_kept_when_it_is_the_stronger_one():
    peaks = [peak(89.0, 331.0, 1e5), peak(-86.6, 331.0, 1e3)]
    result = run(peaks)
    assert len(result.peaks) == 2 and result.report["removed_count"] == 0


def test_delay_mismatch_protects_a_genuine_boundary_peak():
    peaks = [peak(89.0, 331.0, 1e3), peak(-86.6, 500.0, 1e5)]
    result = run(peaks, delay_tolerance_ns=1.0)
    assert len(result.peaks) == 2 and result.report["removed_count"] == 0


def test_two_boundary_peaks_never_remove_each_other():
    peaks = [peak(89.0, 331.0, 1e5), peak(-89.0, 331.0, 1e4)]
    result = run(peaks)
    assert len(result.peaks) == 2 and result.report["removed_count"] == 0


def test_spectrum_ratio_threshold_protects_a_sizable_second_peak():
    peaks = [peak(89.0, 331.0, 8e4), peak(-86.6, 331.0, 1e5)]
    assert len(run(peaks).peaks) == 2
    assert len(run(peaks, maximum_spectrum_ratio=0.9).peaks) == 1


def test_keep_count_truncates_by_spectrum_value_and_keeps_index_alignment():
    peaks = [peak(89.0, 331.0, 1e3), peak(-86.6, 331.0, 1e5),
             peak(10.0, 700.0, 5e4), peak(-40.0, 200.0, 2e4)]
    result = filter_boundary_mirror_peaks(
        peaks, [7, 8, 9, 10], aoa_grid_rad=GRID,
        settings=spurious_peak_filter_settings({"enabled": True}), keep_count=2)
    assert [p.delay_s for p in result.peaks] == [331.0 * 1e-9, 700.0 * 1e-9]
    assert list(result.source_indices) == [8, 9]
    assert result.report["truncated_count"] == 1
    assert result.report["truncated"][0]["aoa_deg"] == pytest.approx(-40.0)


def test_margin_settings_validation():
    assert DEFAULT_SPURIOUS_PEAK_FILTER["peak_margin"] == 2
    assert spurious_peak_filter_settings({"peak_margin": 0})["peak_margin"] == 0
    for bad in ({"peak_margin": -1}, {"peak_margin": 1.5}, {"unknown": 1},
                {"enabled": "yes"}, {"maximum_spectrum_ratio": 0},
                {"edge_tolerance_deg": float("nan")}):
        with pytest.raises(ValueError):
            spurious_peak_filter_settings(bad)


def test_peak_and_source_index_lengths_must_agree():
    with pytest.raises(ValueError):
        filter_boundary_mirror_peaks([peak(0.0, 1.0, 1.0)], [0, 1], aoa_grid_rad=GRID,
                                     settings=spurious_peak_filter_settings({"enabled": True}))


def test_removed_peak_samples_never_reach_reverse_rt_or_saved_records(tmp_path, monkeypatch):
    from copy import deepcopy
    from dataclasses import replace
    import json
    from pathlib import Path
    import time_bias_localization.pipeline as pipeline
    from time_bias_localization.config import DEFAULT_CONFIG, localization_config_view

    config = deepcopy(DEFAULT_CONFIG)
    config["music"]["spurious_peak_filter"]["enabled"] = True
    config["music"]["spectrum_sampling"].update(samples_per_peak=2, local_grid_points_per_axis=9)
    scene = pipeline.prepare_scene(config, tmp_path)
    bundle = pipeline.generate_data(config, scene_json=scene["scene_json"], output_root=tmp_path)
    original = pipeline.sample_music_spectrum
    retained_ids = set()

    def inject(*args, **kwargs):
        sampled = original(*args, **kwargs)
        indices = sampled.refined_peak_source_indices[:3]
        assert len(indices) == 3
        retained_ids.update(f"music_path_{index:02d}" for index in indices[1:])
        return replace(sampled, refined_peaks=[peak(89., 80., 1.), peak(-40., 80., 100.),
                                               peak(20., 120., 50.)],
                       refined_peak_source_indices=indices)

    def stop_at_reverse(scene, bs, samples, **kwargs):
        assert {sample.observation_id for sample in samples} == retained_ids
        raise RuntimeError("checked_filtered_reverse_input")

    monkeypatch.setattr(pipeline, "sample_music_spectrum", inject)
    monkeypatch.setattr(pipeline, "generate_initial_candidate_points", stop_at_reverse)
    with pytest.raises(RuntimeError, match="checked_filtered_reverse_input") as error:
        pipeline.localize(localization_config_view(config), scene_json=scene["scene_json"],
                          online_input=bundle["online_npz"], output_root=tmp_path)
    progress = json.loads(Path(error.value.failure_progress).read_text())
    saved = json.loads(Path(progress["artifacts"]["spectrum_samples"]["path"]).read_text())
    assert {row["observation_id"] for row in saved["samples"]} == retained_ids
    assert {row["observation_id"] for row in saved["regions"]} == retained_ids
    assert saved["diagnostics"]["num_peaks"] == 2
    assert saved["diagnostics"]["total_samples"] == len(saved["samples"])
