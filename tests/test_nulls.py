import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from v1nn.benchmark import PilotConfig, build_model
from v1nn.graph import Graph, rewire
from v1nn.nulls import as_graph, audit_type_null, load_type_null, read_arrays, sha256


def small_bundle(tmp_path):
    rng = np.random.default_rng(19)
    n = 24
    mask = rng.random((n, n)) < 0.15
    np.fill_diagonal(mask, False)
    src, dst = np.nonzero(mask)
    groups = np.where(np.arange(n) % 3, "excitatory", "inhibitory")
    arrays = dict(src=src, dst=dst, groups=groups, signs=np.where(groups == "excitatory", 1, -1),
                  root_ids=np.arange(n).astype(str), positions_um=rng.normal(size=(n, 3)))
    np.savez(tmp_path/"original.npz", **arrays)
    null, rewiring = rewire(as_graph(arrays), len(src)*5, seed=7, groups=groups, max_seconds=10)
    candidate = {**arrays, "src": null.src, "dst": null.dst}
    path = tmp_path/"type_null_7.npz"
    np.savez(path, **candidate)
    audit = audit_type_null(arrays, candidate)
    manifest = {"kind": "directed_degrees_and_node_specific_EI_v1", "swaps_per_edge": 5,
                "source_graph_sha256": sha256(tmp_path/"original.npz"),
                "minimum_edge_change_fraction": 0.5,
                "nulls": {"7": {"file": path.name, "sha256": sha256(path),
                                  "rewiring": rewiring, "audit": audit}}}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    return path, arrays, candidate


def test_type_null_preserves_interface_and_parameter_budget(tmp_path):
    path, arrays, candidate = small_bundle(tmp_path)
    graph, info = load_type_null(path, 7, sha256(tmp_path/"manifest.json"))
    original, _ = build_model(PilotConfig(model="connectome", seed=7), as_graph(arrays))
    null, _ = build_model(PilotConfig(model="type_null", seed=7), graph)
    assert info["audit"]["node_specific_EI_degrees_exact"]
    assert torch.equal(original.core.routing, null.core.routing)
    for name, param in original.named_parameters():
        if name != "core.theta":
            assert torch.equal(param, dict(null.named_parameters())[name])
    # Edge parameters follow the destination degree, not the storage index.
    # Compare each target's initial signed weight multiset after rewiring.
    for node in range(graph.n):
        old_theta = original.core.theta[original.core.dst == node]
        new_theta = null.core.theta[null.core.dst == node]
        assert torch.equal(old_theta.sort().values, new_theta.sort().values)
        old = original.core.effective_weights()[original.core.dst == node]
        new = null.core.effective_weights()[null.core.dst == node]
        # CPU softplus vectorization can differ by one float32 ULP by array slot.
        assert torch.allclose(old.sort().values, new.sort().values, rtol=2e-7, atol=1e-9)
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in null.parameters())
    assert not np.array_equal(arrays["dst"], candidate["dst"])


def test_null_loader_rejects_wrong_identity_and_tampering(tmp_path):
    path, arrays, candidate = small_bundle(tmp_path)
    with pytest.raises(ValueError, match="manifest hash"):
        load_type_null(path, 7, "0"*64)
    with pytest.raises((ValueError, KeyError)):
        load_type_null(path, 8)
    changed = {**candidate, "root_ids": candidate["root_ids"][::-1]}
    with pytest.raises(ValueError, match="identity"):
        audit_type_null(arrays, changed)
    with pytest.raises(ValueError, match="Too few"):
        audit_type_null(arrays, arrays)
    path.write_bytes(path.read_bytes()+b"tampered")
    with pytest.raises(ValueError, match="graph hash"):
        load_type_null(path, 7)


def test_type_audit_rejects_degree_only_null(tmp_path):
    _, arrays, _ = small_bundle(tmp_path)
    graph, _ = rewire(as_graph(arrays), 1000, seed=21)
    with pytest.raises(ValueError, match="node-specific E/I"):
        audit_type_null(arrays, {**arrays, "src": graph.src, "dst": graph.dst})


def test_rewiring_time_limit_fails_closed(monkeypatch):
    graph = Graph(3, np.array([0, 1]), np.array([1, 2]), np.ones(3))
    clock = iter([0.0, 2.0])
    monkeypatch.setattr("v1nn.graph.time.monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="time limit"):
        rewire(graph, 1, max_seconds=1)


def test_type_null_pilot_requires_audited_bundle_and_saves_provenance(tmp_path):
    path, _, _ = small_bundle(tmp_path)
    spec = importlib.util.spec_from_file_location("pilot", Path(__file__).parents[1]/"scripts/pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    config = PilotConfig(model="type_null", seed=7, steps=2, train_size=16,
                         validation_size=8, frames=4, image_size=8, batch=8)
    with pytest.raises(ValueError, match="audited"):
        pilot.run(config, tmp_path/"missing.json")
    result = pilot.run(config, tmp_path/"pilot.json", path)
    assert result["completed_steps"] == 2
    assert result["model"]["prepared_null"]["null_graph_sha256"] == sha256(path)
