import importlib.util
from pathlib import Path

import pytest
import torch

from v1nn.benchmark import (PilotConfig, build_model, paired_histories,
                            metrics, pixel_motion_reference, prepare_data, split_seed)


def test_counterfactual_prevents_static_solution():
    clips, target = paired_histories(batch=2, steps=4, size=8, seed=9)
    for offset in (0, 8):
        assert torch.equal(clips[offset:offset+8, -1], clips[offset, -1].expand(8, -1, -1, -1))
        assert torch.equal(target[offset:offset+8].sum(0), torch.zeros(2))
    assert not torch.equal(clips[0, 0], clips[1, 0])
    assert torch.equal(pixel_motion_reference(clips), target)


def test_split_identity_and_training_only_calibration():
    config = PilotConfig(train_size=24, validation_size=16, image_size=8, frames=4)
    data, manifest = prepare_data(config)
    assert len({split_seed(11, x) for x in ("train", "validation", "counterfactual")}) == 3
    with pytest.raises(ValueError):
        split_seed(11, "test")
    config.validation_size = 32
    changed, other = prepare_data(config)
    assert torch.equal(data["train"][0], changed["train"][0])
    assert manifest["normalization"] == other["normalization"]
    assert manifest["train"] == other["train"]
    assert manifest["train"]["features_sha256"] != manifest["validation"]["features_sha256"]


def test_graph_controls_preserve_interface_and_freeze_reservoir():
    config = PilotConfig(model="connectome", hidden=16)
    original, _ = build_model(config)
    config.model = "no_edges"
    no_edges, _ = build_model(config)
    assert no_edges.core.src.numel() == 0
    assert torch.equal(original.core.routing, no_edges.core.routing)
    config.model = "reservoir"
    reservoir, _ = build_model(config)
    assert all(p.requires_grad == name.startswith("core.readout.")
               for name, p in reservoir.named_parameters())


def test_static_and_reset_ignore_history():
    x = torch.randn(3, 5, 32)
    altered = x.clone()
    altered[:, :-1] += 10
    for kind in ("static", "reset"):
        model, _ = build_model(PilotConfig(model=kind))
        assert torch.equal(model(x), model(altered))


def test_pilot_reproducibility_and_checkpoint(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "pilot", Path(__file__).parents[1]/"scripts/pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = PilotConfig(model="gru", steps=3, train_size=24, validation_size=16,
                         frames=4, image_size=8, hidden=8, batch=8)
    first = module.run(config, tmp_path/"first.json")
    second = module.run(config, tmp_path/"second.json")
    assert first["evaluation"] == second["evaluation"]
    assert first["training"] == second["training"]
    assert first["data"] == second["data"]
    checkpoint = torch.load(tmp_path/"first.pt", weights_only=True)
    assert checkpoint["completed_steps"] == 3
    model, _ = build_model(config)
    model.load_state_dict(checkpoint["model"])
    data, _ = prepare_data(config)
    with torch.no_grad():
        x, target = data["train"]
        # Verify the reported metric uses the saved final weights and every
        # training example, rather than the last stochastic training batch.
        expected = metrics(torch.cat([model(chunk) for chunk in x.split(config.batch)]), target)
    assert first["evaluation"]["train"]["model"] == expected
    assert any(p["change_rms"] > 0 for p in first["optimization"]["parameter_changes"].values())
    assert first["completed_steps"] == 3 and first["scientific_claim"] is None


def test_invalid_budget_rejected():
    for seconds in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            PilotConfig(max_seconds=seconds).validate()


def test_alpha_changes_single_neuron_memory_without_changing_routing():
    config = PilotConfig(model="no_edges", alpha=0.3)
    leaky, _ = build_model(config)
    config.alpha = 1.0
    instantaneous, _ = build_model(config)
    x = torch.zeros(1, 2, 32)
    x[:, 0] = 1
    assert torch.equal(leaky.core.routing, instantaneous.core.routing)
    assert leaky.core(x)[1][:, -1].sum() > 0
    assert instantaneous.core(x)[1][:, -1].sum() == 0
    for alpha in (0, -1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            PilotConfig(alpha=alpha).validate()
