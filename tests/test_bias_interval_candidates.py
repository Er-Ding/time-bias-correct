"""合法分支覆盖必须跨越反射墙、绕射边缘和参考偏差。"""
from dataclasses import replace
import math
import numpy as np
import pytest
from time_bias_localization.bias_interval_candidates import generate_bias_interval_points
from time_bias_localization.candidates import PathObservationSample
from time_bias_localization.constants import SPEED_OF_LIGHT_M_S as C
from time_bias_localization.initial_candidates import (
    generate_initial_candidate_points, cluster_initial_candidate_points, build_representative_trajectories,
)
from time_bias_localization.scene import Scene2D, WallSegment


def room():
    return Scene2D("corridor", (-10.,10.,-10.,10.),
                   (WallSegment("right", (6.,-8.), (6.,8.)),
                    WallSegment("left", (-2.,-8.), (-2.,8.))), 1.5,.1,"test")


def test_all_reflection_segments_exist_even_when_reference_rejects_endpoint():
    sample = PathObservationSample("obs","s0",0.,23./C)
    old = generate_initial_candidate_points(room(),(0.,0.),[sample])
    assert not old.points
    result = generate_bias_interval_points(room(),(0.,0.),[sample],bias_interval_s=(-1/C,23/C))
    assert {p.reflection_wall_ids for p in result.points} == {(),("right",),("right","left")}
    clusters = cluster_initial_candidate_points(result.points, min_samples=1,
        allow_mixed_references=True, beta_interval_m=(-1,23),return_diagnostics=True)
    trajectories = build_representative_trajectories(clusters.representatives,(-1,23))
    for length in (2.,8.,18.):
        legacy = generate_initial_candidate_points(room(),(0.,0.),[sample],reference_bias_s=(23-length)/C).points[0]
        choices = [t for t in trajectories if t.is_valid(23-length)]
        assert any(np.allclose(t.point(23-length),legacy.position_m) for t in choices)
    assert all(t.weight == 1. for t in trajectories)


def test_full_interval_generation_matches_independent_reference_ray_tracing_at_many_biases():
    samples = [PathObservationSample("obs",f"s{i}",angle,20/C) for i,angle in enumerate((-.42,.07,.41))]
    result = generate_bias_interval_points(room(),(0.,0.),samples,bias_interval_s=(-3/C,19/C))
    assert len({p.sample_id for p in result.points}) == len(result.points)
    for beta in np.linspace(-3,19,113):
        expected = generate_initial_candidate_points(room(),(0.,0.),samples,reference_bias_s=beta/C).points
        for point in expected:
            corresponding = [p for p in result.points if p.parent_sample_id == point.sample_id and p.interactions == point.interactions]
            assert len(corresponding) == 1
            p = corresponding[0]
            actual = np.asarray(p.endpoint_origin_m) + (C*p.observed_delay_s-beta-p.prefix_length_m)*np.asarray(p.endpoint_direction)
            np.testing.assert_allclose(actual,point.position_m,atol=1e-10)


def test_diffraction_prefixes_and_post_diffraction_reflections_survive_reference_pruning():
    scene = Scene2D("screen",(-10.,10.,-10.,10.),
                    (WallSegment("screen",(0.,-6.),(0.,0.)),WallSegment("right",(8.,-9.),(8.,9.))),1.5,.1,"test")
    bs = (-4.,-2.)
    samples = [PathObservationSample("obs",f"s{i:03d}",math.atan2(2,4),40/C) for i in range(24)]
    full = generate_bias_interval_points(scene,bs,samples,bias_interval_s=(0.,35/C),max_reflections=1,
                                         max_diffractions=1,diffraction_directions_per_sample=8)
    assert any(p.has_diffraction for p in full.points)
    assert any(p.has_diffraction and p.interactions[-1][0] == "reflection" for p in full.points)
    for beta in (20.,25.,30.):
        old = generate_initial_candidate_points(scene,bs,samples,reference_bias_s=beta/C,max_reflections=1,
                                               max_diffractions=1,diffraction_directions_per_sample=8)
        for p in old.points:
            matching = [q for q in full.points if q.sample_id.rsplit(":segment_",1)[0] == p.sample_id and q.interactions == p.interactions]
            assert len(matching) == 1
            q = matching[0]
            actual = np.asarray(q.endpoint_origin_m)+(C*q.observed_delay_s-beta-q.prefix_length_m)*np.asarray(q.endpoint_direction)
            np.testing.assert_allclose(actual,p.position_m,atol=1e-9)


def test_no_illegal_reference_positions_and_mixed_references_require_explicit_mode():
    sample = PathObservationSample("obs","s0",0.,23/C)
    generated = generate_bias_interval_points(room(),(0.,0.),[sample],bias_interval_s=(0.,23/C))
    assert len({p.reference_bias_s for p in generated.points}) > 1
    for p in generated.points:
        remaining = C*(p.observed_delay_s-p.reference_bias_s)-p.prefix_length_m
        assert 0 < remaining < p.endpoint_free_distance_m
        assert room().contains(p.position_m)
    with pytest.raises(ValueError,match="相同"):
        cluster_initial_candidate_points(generated.points,min_samples=1)
