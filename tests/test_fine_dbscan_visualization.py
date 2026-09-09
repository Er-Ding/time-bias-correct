"""报告必须如实区分粗搜索、细谱正式峰、点簇和离群点。"""

import csv

import numpy as np
import pytest

from time_bias_localization.visualization import read_json, write_json
import time_bias_localization.step_visualization as report


@pytest.fixture
def plots(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = {}
    monkeypatch.setattr(report, "_save", lambda _, fig, folder, name: figures.__setitem__((folder.name, name), fig))
    yield plt, figures
    for figure in figures.values():
        plt.close(figure)


def _rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_formal_fine_peak_is_visible_without_nominal_sample_and_keeps_source_index(tmp_path, plots):
    plt, figures = plots
    spectrum = np.array([[1., 2., 1.], [2., 20., 2.], [1., 2., 1.]])
    region = dict(observation_id="music_path_02", peak_index=2, spectrum=spectrum.tolist(),
                  aoa_grid_rad=[-.1, 0., .1], delay_grid_s=[1e-9, 2e-9, 3e-9],
                  nominal_aoa_local_rad=0., nominal_delay_s=2e-9,
                  coarse_aoa_local_rad=.05, coarse_delay_s=2.5e-9,
                  cell_probabilities=np.full((2, 2), .25).tolist())
    write_json(tmp_path / "samples.json", {"regions": [region], "samples": [
        dict(observation_id="music_path_02", sample_id="mc0", sampling_kind="local_spectrum_mc",
             aoa_local_rad=.03, aoa_global_rad=.03, delay_s=2.2e-9, spectrum_value=3.)]})
    np.savez(tmp_path / "spectrum.npz", spectrum=np.ones((3, 3)), aoa_grid_rad=[-.2, 0., .2],
             delay_grid_s=[1e-9, 2e-9, 3e-9])
    peak = dict(aoa_rad=0., delay_s=2e-9, spectrum_value=20.)
    write_json(tmp_path / "peaks.json", {"nominal": [peak], "nominal_source_indices": [2], "coarse": [
        {**peak, "aoa_rad": .05, "delay_s": 2.5e-9, "spectrum_value": 2.} for _ in range(3)]})
    run = dict(result={"workflow": report.FINE_WORKFLOW}, artifacts={
        "spectrum_samples": str(tmp_path / "samples.json"), "music_peaks": str(tmp_path / "peaks.json"),
        "music_spectrum": str(tmp_path / "spectrum.npz")})
    folder = tmp_path / "03_spectrum_sampling"
    folder.mkdir()
    report._export_spectrum_samples(plt, run, folder)
    fig = figures[(folder.name, "local_spectrum_002")]
    assert "观测 3" in fig.axes[0].get_title()
    by_label = {collection.get_label(): collection for collection in fig.axes[0].collections}
    np.testing.assert_allclose(by_label["正式峰（本张细谱的最大值）"].get_offsets(), [[2., 0.]])
    np.testing.assert_allclose(by_label["粗搜索点（仅确定区域）"].get_offsets(), [[2.5, np.degrees(.05)]])
    probability_figure = figures[(folder.name, "sampling_probabilities_002")]
    np.testing.assert_allclose(probability_figure.axes[0].collections[0].get_array().ravel(), [.25] * 4)
    music_folder = tmp_path / "02_music"
    music_folder.mkdir()
    report._export_music(plt, run, music_folder)
    assert _rows(music_folder / "nominal_peaks.csv")[0]["observation_id"] == "music_path_02"
    music_fig = figures[(music_folder.name, "music_spectrum")]
    assert "粗搜索" in music_fig.axes[0].get_title()
    assert [text.get_text() for text in music_fig.axes[0].texts] == ["P3"]
    assert "底图只用于粗搜索" in music_fig._suptitle.get_text()


@pytest.mark.parametrize("all_noise", [False, True])
def test_dbscan_noise_is_separate_and_failure_still_exports_all_saved_points(tmp_path, plots, all_noise):
    plt, figures = plots
    points = [dict(observation_id="music_path_00", sample_id=f"s{i}", topology_id="los", reference_bias_s=0.,
                   position_m=[2. + i, 1.], reflection_wall_ids=[], observed_aoa_global_rad=0., observed_delay_s=2e-9)
              for i in range(3)]
    members = [] if all_noise else points[:2]
    noise = points if all_noise else points[2:]
    representatives = [] if all_noise else [dict(candidate_id="c0", point=points[0], members=members, metadata={})]
    roles = [dict(observation_id=p["observation_id"], sample_id=p["sample_id"], candidate_id=None if p in noise else "c0",
                  role="noise" if p in noise else "core" if i == 0 else "border") for i, p in enumerate(points)]
    write_json(tmp_path / "points.json", dict(reference_bias_s=0., points=points))
    write_json(tmp_path / "representatives.json", dict(reference_bias_s=0., representatives=representatives,
               noise_points=noise, memberships=roles, diagnostics=dict(algorithm="dbscan", eps_m=1.5, min_samples=5)))
    run = dict(result={"workflow": report.FINE_WORKFLOW}, artifacts={"initial_candidates": str(tmp_path / "points.json"),
               "representative_points": str(tmp_path / "representatives.json")},
               scene={"bounds_m": [-2., 6., -2., 4.], "walls": []}, bs=[0., 0.], true=[1., 1.], boresight=0., metrics={})
    folder = tmp_path / "report"
    report.export_steps(plt, run, folder, dict(status="localization_failed", error="no legal joint solution"))
    cluster_folder = folder / "05_point_clustering"
    counts = read_json(cluster_folder / "counts.json")
    assert counts["listed_member_count"] + counts["noise_count"] == counts["raw_count"] == 3
    assert counts["noise_count"] == len(noise)
    assert counts["representative_count"] == len(representatives)
    assert len(_rows(cluster_folder / "noise_points.csv")) == len(noise)
    table = _rows(cluster_folder / "point_memberships.csv")
    assert len(table) == 3
    assert all(row["candidate_id"] == "" and row["is_representative"] == "False" for row in table if row["role"] == "noise")
    states = {item["step"]: item["status"] for item in read_json(folder / "step_index.json")}
    assert states["05_point_clustering"] == "available"
    assert states["06_representative_trajectories"] == "not_available"
    figure = figures[("05_point_clustering", "members_and_representatives")]
    noise_collection = next(c for c in figure.axes[0].collections if c.get_label().startswith("离群点"))
    assert len(noise_collection.get_offsets()) == len(noise)
    assert "不合并成一个簇" in (cluster_folder / "README.md").read_text()
    # 把离群点偷偷标为一个簇的成员，必须被拒绝，不能画成貌似有效的结果。
    bad = read_json(tmp_path / "representatives.json")
    bad["memberships"][-1]["candidate_id"] = "fake_cluster"
    write_json(tmp_path / "representatives.json", bad)
    invalid_folder = tmp_path / "invalid"
    invalid_folder.mkdir()
    with pytest.raises(ValueError, match="离群点"):
        report._export_point_clusters(plt, run, invalid_folder)

