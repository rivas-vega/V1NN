import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from v1nn.graph import Graph
from v1nn.model import ConnectomeRNN

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
try:
    spec = importlib.util.spec_from_file_location("rectification", ROOT/"scripts/audit_sensory_rectification.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
finally:
    sys.path.pop(0)


def test_relu_rank_loss_zero_state_boundary_and_tolerance_sensitivity():
    matrix = torch.diag(torch.tensor([1., 2e-5, 1.], dtype=torch.float64))
    z = torch.tensor([[1., 1., -1.], [-1., -1., -1.], [1., 0., 1.]])
    r = audit.inspect(matrix, z, [1e-5, 1e-4], 1e-6, time.monotonic()+10)
    assert r["ranks"]["1e-05"]["histogram"] == {"0": 1, "2": 2}
    assert r["ranks"]["0.0001"]["histogram"] == {"0": 1, "1": 1, "2": 1}
    assert r["threshold_disagreement_samples"] == 1
    assert r["samples_with_near_boundary"] == r["exact_zero_preactivations"] == 1
    assert r["ranks"]["1e-05"]["rank_deficient_away_from_boundary"] == 2
    with pytest.raises(TimeoutError):
        audit.inspect(matrix, z, [1e-5, 1e-4], 1e-6, time.monotonic()-1)


def test_first_state_jacobian_matches_actual_core_autograd_with_repeated_routes():
    graph = Graph(4, np.array([0, 1, 2]), np.array([1, 2, 3]), np.ones(4))
    core = ConnectomeRNN(graph, 3, 2, alpha=.3, fan_in=2, routing_seed=7).double()
    with torch.no_grad():
        core.routing.copy_(torch.tensor([[0, 0], [1, 1], [2, 2], [0, 1]]))
        core.bias.copy_(torch.tensor([.1, -.1, .2, .3]))
        core.log_input_gain.copy_(torch.tensor([.3, -.4, .6, -.2]))
    x = torch.tensor([.4, -.7, .8], dtype=torch.float64, requires_grad=True)
    gain = torch.nn.functional.softplus(core.log_input_gain)/np.log(2)
    z = audit.sensory_preactivation(x[None], core.routing, gain, core.bias)[0]
    matrix = audit.routing_matrix(core.routing, 3)*gain[:, None]
    expected = core.alpha*matrix*(z > 0)[:, None]
    jacobian = torch.autograd.functional.jacobian(lambda v: core(v[None, None])[1][0, 0], x)
    assert torch.allclose(jacobian, expected, rtol=1e-12, atol=1e-12)
    assert torch.equal(core(x[None, None])[1][0, 0], core.alpha*torch.relu(z))
    assert audit.rank_at_tolerance(audit.active_singular_values(matrix, z), 1e-5) == 3


def test_support_collision_is_real_and_coverage_is_only_an_upper_bound():
    sys.path.insert(0, str(ROOT/"scripts"))
    try:
        spec = importlib.util.spec_from_file_location("support", ROOT/"scripts/audit_rectification_support.py")
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
    finally:
        sys.path.pop(0)
    routing = torch.tensor([[0, 0], [1, 1]])
    x = torch.tensor([[1., -2.]])
    gain, bias = torch.ones(2), torch.zeros(2)
    matrix = audit.routing_matrix(routing, 2)
    z = audit.sensory_preactivation(x, routing, gain, bias)
    witness = support.collision_witness(x, routing, gain, bias, matrix, z)
    assert witness["feature_index"] == 1 and witness["applied_feature_delta"] > 0
    assert witness["relu_state_exactly_identical"]
    # All features covered does not guarantee full rank: identical rows.
    dependent = torch.ones(2, 2, dtype=torch.float64)
    _, covered = support.support_counts(dependent, torch.ones(1, 2))
    assert int(covered[0]) == 2 and int(torch.linalg.matrix_rank(dependent)) == 1
    with pytest.raises(ValueError, match="nonnegative"):
        support.support_counts(-matrix, z)
