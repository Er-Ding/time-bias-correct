"""有限墙段连续角域剪枝不得丢合法路线，有限预算应覆盖不同观测。"""
import math

import numpy as np

from time_bias_localization.diffraction import enumerate_paths
from time_bias_localization.propagation_hypotheses import (
    _receiver_window_catalogs, _segment_angle_intervals, build_hypothesis_bank,
)
from time_bias_localization.propagation_model import ContinuousObservation
from time_bias_localization.scene import Scene2D, WallSegment, make_synthetic_room


def test_angle_windows_handle_wrap_and_receiver_on_segment_conservatively():
    starts = np.array([[-10., -1.], [-1., 0.], [10., -1.]])
    ends = np.array([[-10., 1.], [1., 0.], [10., 1.]])
    lower, upper = _segment_angle_intervals(starts, ends, np.zeros(2), math.pi-.01, -.02, .02)
    assert lower[0] <= 0 <= upper[0]
    assert lower[1] == -.02 and upper[1] == .02
    assert lower[2] > upper[2]


def test_finite_last_wall_prunes_routes_whose_infinite_mirror_bounds_overlap():
    scene = Scene2D("broad_map", (-1000., 1000., -1000., 1000.), (
        WallSegment("east", (10., -1.), (10., 1.)),
        WallSegment("north", (-1., 10.), (1., 10.)),
    ), 1.5, .1, "test")
    obs = ContinuousObservation("east_peak", 0., 100., .01, .5)
    bank = build_hypothesis_bank(scene, (0., 0.), [obs], max_reflections=1,
                                 max_diffractions=0, aoa_gate_rad=.02)
    routes = {h.interactions for h in bank.hypotheses}
    assert (("reflection", "east"),) in routes
    assert (("reflection", "north"),) not in routes
    report = bank.search_report
    assert report["complete_within_configured_orders"]
    assert report["unsearched_sequences"] == 0
    assert report["total_sequences"] == report["enumerated_sequences"] + report["excluded_by_receiver_angle_domain"]
    assert report["receiver_angle_preselection"]["finite_segment_interval_checks"] == 2


def test_unfolded_windows_keep_every_valid_two_reflection_path_at_multiple_positions():
    scene, bs = make_synthetic_room(), (2.1, 3.2)
    for ue in ((7.3, 5.1), (3.4, 8.2), (8.7, 1.4)):
        paths = enumerate_paths(scene, ue, bs, max_reflections=2)
        observations = [ContinuousObservation(str(i), math.radians(p.arrival_aoa_deg),
                                               p.length_m+2., .01, .1) for i, p in enumerate(paths)]
        bank = build_hypothesis_bank(scene, bs, observations, max_reflections=2,
                                     max_diffractions=0, aoa_gate_rad=1e-6,
                                     beta_interval_m=(-3., 4.), max_hypotheses=10000)
        assert bank.search_report["complete_within_configured_orders"]
        for path, indices in zip(paths, bank.observation_hypothesis_indices):
            route = tuple(("reflection", key) for key in path.interaction_wall_ids)
            assert route in {bank.hypotheses[i].interactions for i in indices}


def test_previous_unfolded_wall_must_share_the_same_angle_window():
    # 从 BS 沿 +x 到 last；previous 的镜像仍位于 y=8，不在同一窄角窗内。
    scene = Scene2D("two_walls", (-20., 20., -20., 20.), (
        WallSegment("last", (10., -1.), (10., 1.)),
        WallSegment("previous", (5., 8.), (8., 8.)),
    ), 1.5, .1, "test")
    walls = {wall.wall_id: wall for wall in scene.walls}
    catalog, _, _, report = _receiver_window_catalogs(walls, {}, np.zeros(2),
        [ContinuousObservation("peak", 0., 30., .01, .5)], 2, .01)
    assert ("previous", "last") not in catalog[2]
    assert report["finite_segment_interval_rejections"] > 0


def test_observations_receive_alternating_wall_sequences_instead_of_global_min_angle_sort():
    walls = {f"east{i}": WallSegment(f"east{i}", (10.+i, -.1), (10.+i, .1)) for i in range(20)}
    walls["north"] = WallSegment("north", (-.1, 5.), (.1, 5.))
    observations = [ContinuousObservation("east", 0., 30., .01, 1.),
                    ContinuousObservation("north", math.pi/2, 10., .01, 1.)]
    catalog, _, _, _ = _receiver_window_catalogs(walls, {}, np.zeros(2), observations, 1, .02)
    assert catalog[1][0][0].startswith("east")
    assert catalog[1][1] == ("north",)
