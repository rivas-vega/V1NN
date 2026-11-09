import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from test_nulls import small_bundle
from v1nn.benchmark import PilotConfig, build_model
from v1nn.crossed import PAIRS, validate_crossed_plan, crossed_trials
from v1nn.nulls import as_graph

ROOT = Path(__file__).parents[1]


def test_crossed_design_rejects_duplicates_diagonal_and_changed_budget():
    plan = json.loads((ROOT/"configs/spatial_crossed_plan.json").read_text())
    assert [(r, g) for _, r, g in validate_crossed_plan(plan)] == PAIRS
    for change in ({"steps": 3001}, {"campaign_seconds": 1800},
                   {"new_trials_only": plan["new_trials_only"]+[dict(seed=7, null_seed=7)]},
                   {"new_trials_only": [plan["new_trials_only"][0]]*6},
                   {"null_bundle_manifest_sha256": "0"*64}):
        with pytest.raises(ValueError):
            validate_crossed_plan({**plan, **change})
    kwargs = dict(models="spatial_null", seeds="7,8,9", steps=3000, seconds=1200,
                  grid=4, graph="spatial_nulls", alpha=0.3, train_size=2048,
                  batch=64, lr=0.003, null_manifest_sha256=plan["null_bundle_manifest_sha256"])
    assert len(crossed_trials(json.dumps(plan), **kwargs)) == 6
    with pytest.raises(ValueError, match="Launcher"):
        crossed_trials(json.dumps(plan), **{**kwargs, "seeds": "8,7,9"})


def test_explicit_null_seed_preserves_diagonal_and_independently_selects_graph(tmp_path):
    path, original, _ = small_bundle(tmp_path)
    spec = importlib.util.spec_from_file_location("pilot", ROOT/"scripts/pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    config = PilotConfig(model="type_null", seed=7, steps=2, train_size=16,
                         validation_size=8, frames=4, image_size=8, batch=8)
    a = pilot.run(config, tmp_path/"implicit.json", path)
    b = pilot.run(replace(config, null_seed=7), tmp_path/"explicit.json", path)
    assert a["evaluation"] == b["evaluation"] and a["training"] == b["training"]
    off = pilot.run(replace(config, seed=8, null_seed=7), tmp_path/"off.json", path)
    c = torch.load(tmp_path/"off.pt", weights_only=True)
    measured, _ = build_model(replace(config, model="connectome", seed=8), as_graph(original))
    assert torch.equal(c["model"]["core.routing"], measured.core.routing)
    assert not torch.equal(c["model"]["core.routing"], torch.load(tmp_path/"implicit.pt", weights_only=True)["model"]["core.routing"])
    assert off["graph_sha256"] == a["graph_sha256"]
    assert c["prepared_null"] == off["model"]["prepared_null"]
    assert c["config"]["seed"] == 8 and c["config"]["null_seed"] == 7
    for invalid in (replace(config, null_seed=-1), replace(config, null_seed=1.5),
                    replace(config, model="connectome", null_seed=7)):
        with pytest.raises(ValueError, match="null_seed"):
            invalid.validate()


def test_crossed_summary_averages_all_graphs_without_changing_seed_pairing():
    spec = importlib.util.spec_from_file_location("comparison", ROOT/"scripts/compare_spatial_crossed.py")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(ROOT/"scripts"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    summary = module.summarize([2, 3, 4], [[1, 2, 3], [2, 3, 4], [2, 3, 4]])
    assert summary["paired_measured_minus_null_mean"] == [0, 0, 1]
    assert summary["mean_paired_difference"] == pytest.approx(1/3)
    assert summary["null_std_within_training_seed"] == [1, 1, 1]
    with pytest.raises(ValueError):
        module.summarize([1, 2, 3], [[1, 2], [2, 3], [3, 4]])
