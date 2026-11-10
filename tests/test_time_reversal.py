import sys
from pathlib import Path

import torch

from v1nn.benchmark import pixel_motion_reference
from v1nn.model import FixedRetina
from v1nn.synthetic import moving_textures

sys.path.insert(0, str(Path(__file__).parents[1]/"scripts"))
try:
    from audit_time_reversal import reversal_metrics
finally:
    sys.path.pop(0)


def test_reversal_is_a_valid_motion_transform_and_retina_commutes():
    clips, _, labels = moving_textures(batch=32, seed=123)
    reverse = clips.flip(1)
    assert torch.equal(pixel_motion_reference(reverse), -labels)
    assert torch.equal(FixedRetina()(reverse), FixedRetina()(clips).flip(1))
    assert torch.equal(reverse.flip(1), clips)


def test_paired_identity_and_projection_do_not_guarantee_forward_improvement():
    y = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.float64)
    p, q = y.clone(), y.clone()  # Correct forwards; wrong backwards.
    result = reversal_metrics(p, q, y)
    assert result["original"]["mean_endpoint_error"] == 0
    assert result["odd_projection"]["mean_endpoint_error"] == 1
    assert result["paired_mse_identity_absolute_residual"] == 0
    odd = (p-q)/2
    assert torch.equal((q-p)/2, -odd)
    perfect = reversal_metrics(y, -y, y)
    assert perfect["even_component_squared_norm"] == 0
    assert perfect["odd_projection"]["mean_endpoint_error"] == 0
