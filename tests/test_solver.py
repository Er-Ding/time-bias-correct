from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from time_bias_localization.solver import (  # noqa: E402
    CandidateTrajectory,
    SolverConfig,
    SolverError,
    _fit_assignment,
    solve_position_and_bias,
)


def _unit(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.array((np.cos(angle), np.sin(angle)))


def _candidate(
    observation_id: str,
    candidate_id: str,
    mu: np.ndarray,
    beta: float,
    direction: np.ndarray,
    *,
    error: tuple[float, float] = (0.0, 0.0),
    beta_interval: tuple[float, float] = (-np.inf, np.inf),
) -> CandidateTrajectory:
    anchor = mu + beta * direction + np.asarray(error)
    return CandidateTrajectory(
        observation_id=observation_id,
        candidate_id=candidate_id,
        anchor_m=anchor,
        direction=direction,
        beta_min_m=beta_interval[0],
        beta_max_m=beta_interval[1],
    )


def test_two_observations_recover_exact_position_and_bias() -> None:
    true_mu = np.array((4.0, -3.0))
    true_beta = 7.5
    candidates = [
        _candidate("path-0", "los", true_mu, true_beta, _unit(0.0)),
        _candidate("path-1", "wall-a", true_mu, true_beta, _unit(90.0)),
    ]

    result = solve_position_and_bias(candidates)

    np.testing.assert_allclose(result.mu, true_mu, atol=1e-9)
    assert result.beta == pytest.approx(true_beta, abs=1e-9)
    assert result.selected_candidate_ids == {
        "path-0": "los",
        "path-1": "wall-a",
    }
    assert max(result.residuals.values()) < 1e-9


def test_multiple_observations_use_one_shared_solution() -> None:
    true_mu = np.array((12.0, 5.0))
    true_beta = 3.25
    directions = [_unit(angle) for angle in (-70.0, -10.0, 35.0, 85.0, 145.0)]
    errors = [
        (0.02, -0.01),
        (-0.01, 0.015),
        (0.005, -0.02),
        (-0.015, 0.005),
        (0.01, 0.015),
    ]
    candidates = [
        _candidate(f"path-{index}", "only", true_mu, true_beta, direction, error=error)
        for index, (direction, error) in enumerate(zip(directions, errors))
    ]

    result = solve_position_and_bias(candidates, SolverConfig(huber_delta=0.2))

    np.testing.assert_allclose(result.mu, true_mu, atol=0.03)
    assert result.beta == pytest.approx(true_beta, abs=0.03)
    assert result.diagnostics.selected_observation_count == len(candidates)


def test_candidates_from_same_observation_are_mutually_exclusive() -> None:
    true_mu = np.array((-2.0, 8.0))
    true_beta = 4.0
    correct = _candidate("path-0", "correct", true_mu, true_beta, _unit(-20.0))
    wrong = CandidateTrajectory(
        observation_id="path-0",
        candidate_id="wrong-reflector",
        anchor_m=np.array((30.0, -25.0)),
        direction=_unit(150.0),
    )
    candidates = [
        wrong,
        correct,
        _candidate("path-1", "only", true_mu, true_beta, _unit(55.0)),
        _candidate("path-2", "only", true_mu, true_beta, _unit(125.0)),
    ]

    result = solve_position_and_bias(candidates)

    assert result.selected_candidate_ids["path-0"] == "correct"
    assert len(result.selected_candidates) == 3
    np.testing.assert_allclose(result.mu, true_mu, atol=1e-8)
    assert result.beta == pytest.approx(true_beta, abs=1e-8)


def test_huber_weighting_limits_one_outlier() -> None:
    true_mu = np.array((6.0, 2.0))
    true_beta = 5.0
    inlier_directions = [_unit(angle) for angle in (-80.0, -35.0, 10.0, 55.0, 105.0, 155.0)]
    candidates = [
        _candidate(
            f"inlier-{index}",
            "only",
            true_mu,
            true_beta,
            direction,
            error=(0.01 * (-1) ** index, -0.008 * (-1) ** index),
        )
        for index, direction in enumerate(inlier_directions)
    ]
    outlier_direction = _unit(25.0)
    candidates.append(
        _candidate(
            "outlier",
            "only",
            true_mu,
            true_beta,
            outlier_direction,
            error=(18.0, -14.0),
        )
    )

    result = solve_position_and_bias(candidates, SolverConfig(huber_delta=0.15))

    np.testing.assert_allclose(result.mu, true_mu, atol=0.12)
    assert result.beta == pytest.approx(true_beta, abs=0.12)
    assert "outlier" in result.diagnostics.downweighted_observations
    assert result.diagnostics.robust_weights["outlier"] < 0.02


def test_beta_interval_is_enforced_during_candidate_selection() -> None:
    true_mu = np.array((1.0, 1.5))
    true_beta = 2.0
    candidates = [
        _candidate(
            "path-0",
            "valid",
            true_mu,
            true_beta,
            _unit(0.0),
            beta_interval=(1.0, 3.0),
        ),
        CandidateTrajectory(
            observation_id="path-0",
            candidate_id="expired",
            anchor_m=np.array((1.0, 1.5)),
            direction=_unit(45.0),
            beta_min_m=8.0,
            beta_max_m=10.0,
        ),
        _candidate(
            "path-1",
            "valid",
            true_mu,
            true_beta,
            _unit(90.0),
            beta_interval=(0.0, 4.0),
        ),
    ]

    result = solve_position_and_bias(candidates)

    assert result.selected_candidate_ids["path-0"] == "valid"
    assert 1.0 <= result.beta <= 3.0


def test_position_covariance_is_symmetric_positive_semidefinite() -> None:
    true_mu = np.array((3.0, -4.0))
    true_beta = 6.0
    candidates = [
        _candidate(
            f"path-{index}",
            "only",
            true_mu,
            true_beta,
            _unit(angle),
            error=(0.03 * index, -0.01 * index),
        )
        for index, angle in enumerate((-60.0, 5.0, 70.0, 135.0))
    ]

    result = solve_position_and_bias(candidates)

    assert result.sigma.shape == (2, 2)
    np.testing.assert_allclose(result.sigma, result.sigma.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(result.sigma)) >= -1e-12
    assert result.diagnostics.joint_covariance.shape == (3, 3)


def test_candidate_input_order_does_not_change_result() -> None:
    true_mu = np.array((2.5, 9.0))
    true_beta = -1.75
    candidates = [
        _candidate("path-c", "only", true_mu, true_beta, _unit(130.0)),
        _candidate("path-a", "wrong", true_mu + 20.0, true_beta, _unit(-40.0)),
        _candidate("path-b", "only", true_mu, true_beta, _unit(40.0)),
        _candidate("path-a", "correct", true_mu, true_beta, _unit(-25.0)),
    ]

    forward = solve_position_and_bias(candidates)
    reversed_result = solve_position_and_bias(reversed(candidates))

    np.testing.assert_allclose(forward.mu, reversed_result.mu, atol=1e-12)
    np.testing.assert_allclose(forward.sigma, reversed_result.sigma, atol=1e-12)
    assert forward.beta == pytest.approx(reversed_result.beta, abs=1e-12)
    assert forward.selected_candidate_ids == reversed_result.selected_candidate_ids


def test_parallel_trajectories_report_unidentifiable_problem() -> None:
    candidates = [
        CandidateTrajectory("path-0", "only", (1.0, 2.0), (1.0, 0.0)),
        CandidateTrajectory("path-1", "only", (3.0, 4.0), (1.0, 0.0)),
    ]

    with pytest.raises(SolverError, match="可辨识"):
        solve_position_and_bias(candidates)


def test_returned_state_is_refit_with_the_returned_candidate_assignment() -> None:
    raw = [
        (0, 0, (-3.6648332058049427, 5.947309146654682), (-0.1419211795596572, -0.9898779615651596), -2.5899626339922035, 3.128876404815272),
        (0, 1, (1.9661750717437965, -6.265316287925733), (0.4971710504601014, -0.8676525494599779), -2.6179516478830296, 7.534422922159497),
        (1, 0, (8.977623036666365, 3.3447490620074483), (-0.011022276868061875, -0.9999392528612144), -7.232816515247103, 3.5347173293425023),
        (1, 1, (3.949069997640443, -3.4705427185977573), (-0.7562278317979771, 0.6543083878524945), -2.128574693359468, 1.7610796443638899),
        (2, 0, (-6.802087978499049, -3.197996300905894), (-0.8714361009229213, -0.49050904375786597), -4.2784547703835925, 2.1313682263261677),
        (2, 1, (-6.13411221421011, -7.410618476455994), (-0.4016202173721171, 0.9158063119448203), -7.266681987640513, 4.7885441093193055),
    ]
    candidates = [
        CandidateTrajectory(
            observation_id=observation_id,
            candidate_id=candidate_id,
            anchor_m=anchor,
            direction=direction,
            beta_min_m=beta_min,
            beta_max_m=beta_max,
        )
        for observation_id, candidate_id, anchor, direction, beta_min, beta_max in raw
    ]
    config = SolverConfig(max_iterations=10, max_irls_iterations=30)

    result = solve_position_and_bias(candidates, config)
    state = np.asarray([result.mu[0], result.mu[1], result.beta])
    refit_state, _, _, _, _, converged = _fit_assignment(
        result.selected_candidates, state, config
    )

    assert converged
    np.testing.assert_allclose(state, refit_state, atol=1e-9)
