import numpy as np
import pytest
import torch
from fleetweaver.core import (
    generate,
    instance,
    validate_routes,
    actions_to_routes,
    make_policy,
    predict,
)
from fleetweaver.baselines import greedy, two_opt, ortools


@pytest.mark.parametrize(
    "distribution",
    [
        "uniform",
        "clustered",
        "corner",
        "ring",
        "stripe",
        "heavy_demand",
        "heterogeneous",
    ],
)
def test_baselines_feasible(distribution):
    inst = instance(generate(12, 3, 1, 777, distribution)[0])
    for balanced in [True, False]:
        routes = greedy(inst, balanced)
        value, _ = validate_routes(inst, routes)
        improved, _ = validate_routes(inst, two_opt(inst, routes))
        assert improved <= value + 1e-6


def test_validator_rejects_invalid_and_accepts_refills():
    inst = dict(
        locs=np.array([[1.0, 0.0], [2.0, 0.0]]),
        depot=np.array([0.0, 0.0]),
        demand=np.array([6, 6]),
        capacity=np.array([10]),
        speed=np.array([2.0]),
    )
    assert validate_routes(inst, [[0, 1, 0, 2, 0]])[0] == 3
    for route in [[0, 1, 2, 0], [0, 1, 1, 0], [0, 1, 0], [1, 2, 0]]:
        with pytest.raises(ValueError):
            validate_routes(inst, [route])
    paths = ortools(inst, 0.2)
    assert paths is not None
    assert validate_routes(inst, paths)[0] == pytest.approx(3)


def test_original_environment_matches_independent_evaluator():
    torch.set_num_threads(2)
    torch.manual_seed(101)
    td = generate(12, 3, 4, 333)
    policy = make_policy()
    for samples in [1, 4]:
        out = predict(policy, td, samples=samples)
        for i in range(4):
            value, _ = validate_routes(
                instance(td[i]), actions_to_routes(out["actions"][i], 3)
            )
            assert value == pytest.approx(float(-out["reward"][i]), rel=1e-5)


def test_generator_reproducible_and_does_not_modify_global_rng():
    torch.manual_seed(22)
    expected = torch.rand(5)
    torch.manual_seed(22)
    a = generate(12, 3, 4, 333)
    assert torch.equal(expected, torch.rand(5))
    b = generate(12, 3, 4, 333)
    for k in a.keys():
        assert torch.equal(a[k], b[k])


def test_solver_uses_minmax_and_heterogeneous_speeds():
    inst = dict(
        locs=np.array([[-1.0, 0.0], [2.0, 0.0]]),
        depot=np.array([0.0, 0.0]),
        demand=np.array([1, 1]),
        capacity=np.array([10, 10]),
        speed=np.array([1.0, 2.0]),
    )
    routes = ortools(inst, 0.2)
    assert validate_routes(inst, routes)[0] == pytest.approx(2.0)


@pytest.mark.parametrize("seed", range(5))
def test_multitrip_solver_feasible_and_keeps_warm_start(seed):
    inst = instance(generate(12, 3, 1, 100 + seed, "heterogeneous")[0])
    warm = validate_routes(inst, two_opt(inst, greedy(inst)))[0]
    solved = validate_routes(inst, ortools(inst, 0.15))[0]
    assert solved <= warm + 1e-3  # integer time rounding cannot change more than this


@pytest.mark.parametrize(
    "n,m,dist", [(10, 2, "ring"), (40, 8, "heavy_demand"), (100, 10, "heterogeneous")]
)
def test_policy_supports_unseen_problem_shapes(n, m, dist):
    torch.set_num_threads(2)
    torch.manual_seed(303)
    td = generate(n, m, 1, 700123, dist)
    result = predict(make_policy(), td)
    value, _ = validate_routes(
        instance(td[0]), actions_to_routes(result["actions"][0], m)
    )
    assert value == pytest.approx(float(-result["reward"][0]), rel=2e-5)
