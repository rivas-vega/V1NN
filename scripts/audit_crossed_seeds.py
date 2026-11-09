"""CPU preflight: historical diagonal compatibility and full real-graph identity."""
import hashlib
import importlib.util
import json
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from pilot import atomic_json, run
from v1nn.benchmark import PilotConfig, build_model
from v1nn.crossed import PAIRS, SPATIAL_MANIFEST
from v1nn.nulls import as_graph, read_arrays, load_spatial_null, sha256

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = "3f5ce8436c24e4062f70ae16202dd600fa163ce2"


def tree_equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    torch.set_num_threads(1)
    bundle = ROOT/"data/processed/spatial_nulls_v2"
    assert sha256(bundle/"manifest.json") == SPATIAL_MANIFEST
    original = as_graph(read_arrays(bundle/"original.npz"))
    audit = {"historical_commit": HISTORICAL, "null_manifest_sha256": SPATIAL_MANIFEST,
             "device": "cpu", "diagonal": {}, "off_diagonal": {},
             "limits": "Two-step engineering check on reduced data, not scientific performance evidence."}
    # Historical pilot + benchmark; all model, graph, data and null implementations are unchanged.
    for path in ("src/v1nn/model.py", "src/v1nn/graph.py", "src/v1nn/synthetic.py", "src/v1nn/nulls.py"):
        assert (ROOT/path).read_bytes() == subprocess.check_output(["git", "show", f"{HISTORICAL}:{path}"], cwd=ROOT)
    with tempfile.TemporaryDirectory(dir=ROOT.parent) as temp:
        temp = Path(temp)
        modules = []
        for name, path in (("v1nn._historical_benchmark", "src/v1nn/benchmark.py"),
                           ("_historical_pilot", "scripts/pilot.py")):
            local = temp/Path(path).name
            local.write_bytes(subprocess.check_output(["git", "show", f"{HISTORICAL}:{path}"], cwd=ROOT))
            spec = importlib.util.spec_from_file_location(name, local)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
        old_benchmark, old_pilot = modules
        for key in ("MODELS", "PilotConfig", "build_model", "config_dict", "metrics", "prepare_data"):
            setattr(old_pilot, key, getattr(old_benchmark, key))
        # Source identity is metadata, never part of inference or optimization.
        old_pilot.source_identity = lambda: {"historical_commit": HISTORICAL}
        base = PilotConfig(model="spatial_null", steps=2, train_size=16,
                           validation_size=8, frames=4, image_size=8, batch=8)
        for seed in (7, 8, 9):
            path = bundle/f"spatial_null_{seed}.npz"
            graph, _ = load_spatial_null(path, seed, SPATIAL_MANIFEST)
            cfg = replace(base, seed=seed)
            old_cfg = old_benchmark.PilotConfig(**{k: v for k, v in vars(cfg).items() if k != "null_seed"})
            old = old_pilot.run(old_cfg, temp/"old.json", path)
            old_ckpt = torch.load(temp/"old.pt", weights_only=True)
            for label, config in (("implicit", cfg), ("explicit", replace(cfg, null_seed=seed))):
                result = run(config, temp/f"{label}.json", path)
                ckpt = torch.load(temp/f"{label}.pt", weights_only=True)
                for key in ("data", "training", "evaluation"):
                    assert result[key] == old[key]
                for key in ("model", "optimizer", "normalization"):
                    assert tree_equal(ckpt[key], old_ckpt[key])
            audit["diagonal"][str(seed)] = {"historical_training_metrics_model_optimizer_exact": True}
        for r, g in PAIRS:
            path = bundle/f"spatial_null_{g}.npz"
            graph, info = load_spatial_null(path, g, SPATIAL_MANIFEST)
            config = replace(base, seed=r, null_seed=g)
            measured, _ = build_model(replace(config, model="connectome", null_seed=None), original)
            null, _ = build_model(config, graph)
            assert torch.equal(null.core.routing, measured.core.routing)
            for name, p in measured.named_parameters():
                if name != "core.theta":
                    assert torch.equal(p, dict(null.named_parameters())[name])
            result = run(config, temp/"off.json", path)
            ckpt = torch.load(temp/"off.pt", weights_only=True)
            assert torch.equal(ckpt["model"]["core.src"], torch.from_numpy(graph.src))
            assert torch.equal(ckpt["model"]["core.dst"], torch.from_numpy(graph.dst))
            assert torch.equal(ckpt["model"]["core.routing"], measured.core.routing)
            assert ckpt["config"]["seed"] == r and ckpt["config"]["null_seed"] == g
            assert ckpt["prepared_null"] == result["model"]["prepared_null"] == info
            audit["off_diagonal"][f"{r},{g}"] = {"graph_hash": sha256(path),
                "routing_and_node_initialization_paired": True, "checkpoint_provenance_verified": True}
    audit["implementation_sha256"] = {p: sha256(ROOT/p) for p in
        ("src/v1nn/benchmark.py", "src/v1nn/crossed.py", "scripts/pilot.py", "scripts/modal_pilot.py")}
    atomic_json(ROOT/"reports/crossed_seed_preflight.json", audit)
    print("Historical diagonal and all six independent pairs verified on CPU")


if __name__ == "__main__":
    main()
