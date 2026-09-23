"""BS 端的提前排除必须有与 UE 无关的证据，并保留全部合法路径。"""
import math

import numpy as np
import pytest

from time_bias_localization.diffraction import enumerate_paths
from time_bias_localization.propagation_hypotheses import (
    _fully_hidden_walls_from_receiver, _receiver_geometry_failure, build_hypothesis_bank,
)
from time_bias_localization.propagation_model import ContinuousObservation, evaluate_hypothesis
from time_bias_localization.scene import Scene2D, WallSegment


def _scene(walls):
    return Scene2D("receiver_visibility", (-20., 20., -20., 20.), tuple(walls), 1.5, .1, "test")


def _walls(*walls):
    return {w.wall_id: w for w in walls}


@pytest.mark.parametrize("angle", [0., .7, 2.4])
@pytest.mark.parametrize("reverse_endpoints", [False, True])
def test_wall_wholly_behind_another_wall_is_removed_before_bank_admission(angle, reverse_endpoints):
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    walls = []
    for key, points in (("front", ((2., -2.), (2., 2.))), ("hidden", ((4., -1.), (4., 1.)))):
        endpoints = np.asarray(points) @ rotation.T
        if reverse_endpoints:
            endpoints = endpoints[::-1]
        walls.append(WallSegment(key, tuple(endpoints[0]), tuple(endpoints[1])))
    receiver = np.zeros(2)
    hidden = _fully_hidden_walls_from_receiver(_walls(*walls), receiver)
    assert hidden == {"hidden": "front"}
    bank = build_hypothesis_bank(_scene(walls), receiver,
        [ContinuousObservation("peak", 0., 20., .1, 1.)], max_reflections=1, max_diffractions=0)
    assert (("reflection", "hidden"),) not in {h.interactions for h in bank.hypotheses}
    assert (("reflection", "front"),) in {h.interactions for h in bank.hypotheses}
    report = bank.search_report
    assert report["receiver_geometry_preselection"]["rejected_sequence_counts"] == {"last_wall_fully_hidden_from_bs": 1}
    assert report["excluded_by_fixed_geometry"] == 1
    assert report["complete_within_configured_orders"]


def test_partly_blocked_wall_is_kept_and_its_visible_reflection_survives():
    walls = (WallSegment("front", (2., -1.), (2., 1.)),
             WallSegment("partial", (4., -10.), (4., 10.)))
    hidden = _fully_hidden_walls_from_receiver(_walls(*walls), np.zeros(2))
    assert "partial" not in hidden
    bank = build_hypothesis_bank(_scene(walls), (0., 0.),
        [ContinuousObservation("peak", math.pi/4, 12., .1, 1.)], max_reflections=1, max_diffractions=0)
    candidate = next(h for h in bank.hypotheses if h.interactions == (("reflection", "partial"),))
    assert evaluate_hypothesis(bank.scene, candidate, (0., 8.)).valid


@pytest.mark.parametrize("target", [((-4., -1.), (-4., 1.)), ((4., -2.), (4., 2.)),
                                     ((1., -1.), (1., 1.))])
def test_wall_behind_bs_grazing_shadow_or_in_front_is_not_certified_hidden(target):
    walls = _walls(WallSegment("blocker", (2., -1.), (2., 1.)), WallSegment("target", *target))
    assert "target" not in _fully_hidden_walls_from_receiver(walls, np.zeros(2))


def test_opposite_side_reflection_sequence_has_an_explicit_rejection_reason():
    walls = _walls(WallSegment("previous", (6., -2.), (6., 2.)),
                   WallSegment("last", (4., -5.), (4., 5.)))
    sequence = (("reflection", "previous"), ("reflection", "last"))
    assert _receiver_geometry_failure(sequence, walls, np.zeros(2), {}) == "previous_wall_on_opposite_side_of_last_wall"
    assert _receiver_geometry_failure(sequence, walls, np.array([8., 0.]), {}) is None
    assert _receiver_geometry_failure(sequence, walls, np.array([4., 0.]), {}) is None


@pytest.mark.parametrize("with_diffraction", [False, True])
def test_early_checks_preserve_every_reference_path_at_several_positions(with_diffraction):
    scene = _scene((WallSegment("screen", (0., -6.), (0., 0.)),
                    WallSegment("right", (8., -9.), (8., 9.)),
                    WallSegment("behind_right", (10., -3.), (10., 3.))))
    bs = (-4., -2.)
    positions = [(-3., 4.), (1., 7.), (4., 4.), (5., 6.)]
    if with_diffraction:
        # 这两个位置被 screen 挡住，只有开启绕射才有参考路径。
        positions += [(4., -2.), (5., -3.)]
    for ue in positions:
        paths = enumerate_paths(scene, ue, bs, max_reflections=2, max_diffractions=int(with_diffraction))
        assert paths
        observations = [ContinuousObservation(str(i), math.radians(p.arrival_aoa_deg),
                        p.length_m + 2., .01, .1) for i, p in enumerate(paths)]
        bank = build_hypothesis_bank(scene, bs, observations, max_reflections=2,
            max_diffractions=int(with_diffraction), max_hypotheses=10000, max_enumerated_sequences=10000,
            aoa_gate_rad=.001, beta_interval_m=(-3., 4.))
        routes = {h.interactions for h in bank.hypotheses}
        for path in paths:
            expected = path.propagation_interactions or tuple(("reflection", key) for key in path.interaction_wall_ids)
            assert expected in routes
        assert bank.search_report["complete_within_configured_orders"]


def test_empty_map_and_no_reflections_remain_supported():
    assert _fully_hidden_walls_from_receiver({}, np.zeros(2)) == {}
    bank = build_hypothesis_bank(_scene(()), (0., 0.),
        [ContinuousObservation("peak", 0., 5., .1, 1.)], max_reflections=0, max_diffractions=0)
    assert len(bank.hypotheses) == 1
    assert bank.search_report["receiver_geometry_preselection"]["fully_hidden_wall_count"] == 0
