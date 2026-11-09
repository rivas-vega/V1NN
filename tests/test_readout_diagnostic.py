import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/"scripts"))
from readout_diagnostic import ridge_predictions


def test_ridge_matches_direct_solution_and_does_not_fit_validation():
    generator = torch.Generator().manual_seed(52)
    x = torch.randn(32, 5, generator=generator).double()
    # Include a dependent and a constant feature to exercise singular designs.
    x = torch.cat([x, x[:, :1], torch.ones(32, 1)], dim=1)
    target = torch.randn(32, 2, generator=generator).double()
    predictions, audit = ridge_predictions(x, target, {"train": x, "validation": x+20}, (0, .01))
    changed, _ = ridge_predictions(x, target, {"train": x, "validation": x-50}, (0, .01))
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-8)
    z = (x-mean)/scale
    direct = torch.linalg.solve(z.T@z/len(x)+.01*torch.eye(x.shape[1]),
                                z.T@(target-target.mean(0))/len(x))
    assert torch.allclose(predictions['0.01']['train'], z@direct+target.mean(0), atol=1e-9)
    assert torch.equal(predictions['0']['train'], changed['0']['train'])
    assert audit['numerical_rank'] == 5
    ordinary = torch.linalg.lstsq(z, target-target.mean(0), driver='gelsd').solution
    assert torch.allclose(predictions['0']['train'], z@ordinary+target.mean(0), atol=1e-9)
