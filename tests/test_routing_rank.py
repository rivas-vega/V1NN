import importlib.util
import sys
from pathlib import Path

import torch


def test_population_routing_audit_identifies_missing_feature_and_duplicate_routes():
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("routing_audit", root/"scripts/audit_routing_rank.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root/"scripts"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    routing = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    matrix = module.routing_matrix(routing, 3)
    assert torch.equal(matrix, torch.tensor([[.5, .5, 0], [0, 1, 0]], dtype=torch.float64))
    x = torch.eye(3).unsqueeze(0)
    sensory = x[:, :, routing].mean(-1)
    result = module.diagnose(matrix, {"train": sensory}, {"train": (x, None)})
    assert result["rank"] == 2
    assert result["reconstruction"]["train"]["max_absolute_error"] == 1
