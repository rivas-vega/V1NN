"""Inspect preserved interfaces and remaining spatial confounds before GPU use."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pilot import atomic_json
from v1nn.benchmark import PilotConfig, build_model
from v1nn.nulls import as_graph, load_type_null, read_arrays, sha256


def spatial_summary(arrays):
    distances = np.linalg.norm(arrays["positions_um"][arrays["src"]]-
                               arrays["positions_um"][arrays["dst"]], axis=1)
    return {"mean_um": float(distances.mean()),
            "quantiles_10_50_90_um": np.quantile(distances, [0.1, 0.5, 0.9]).tolist(),
            "interpretation": "soma separation; not axonal path length"}


def audit_bundle(folder):
    folder = Path(folder)
    torch.set_num_threads(1)
    manifest = json.loads((folder/"manifest.json").read_text())
    original = read_arrays(folder/"original.npz")
    graph = as_graph(original)
    original_edges = set(zip(graph.src.tolist(), graph.dst.tolist()))
    report = {"bundle_manifest_sha256": sha256(folder/"manifest.json"),
              "generation_manifest": manifest, "original_spatial_summary": spatial_summary(original),
              "nulls": {}}
    for seed in manifest["seeds"]:
        path = folder/f"type_null_{seed}.npz"
        null, audit = load_type_null(path, seed, report["bundle_manifest_sha256"])
        candidate = read_arrays(path)
        measured, _ = build_model(PilotConfig(model="connectome", seed=seed), graph)
        randomized, _ = build_model(PilotConfig(model="type_null", seed=seed), null)
        assert torch.equal(measured.core.routing, randomized.core.routing)
        for name, parameter in measured.named_parameters():
            if name != "core.theta":
                assert torch.equal(parameter, dict(randomized.named_parameters())[name])
        assert torch.equal(measured.core.theta.sort().values, randomized.core.theta.sort().values)
        assert sum(p.numel() for p in measured.parameters()) == sum(p.numel() for p in randomized.parameters())
        initial_mass_max_error = 0.0
        with torch.no_grad():
            for sign in (-1, 1):
                def signed_mass(model):
                    core = model.core
                    weights = core.effective_weights().abs()*(core.edge_sign == sign)
                    return torch.zeros(core.n).index_add_(0, core.dst, weights)
                a, b = signed_mass(measured), signed_mass(randomized)
                assert torch.allclose(a, b, rtol=2e-6, atol=1e-7)
                initial_mass_max_error = max(initial_mass_max_error, float((a-b).abs().max()))
        null_edges = set(zip(null.src.tolist(), null.dst.tolist()))
        by_group = {}
        groups = original["groups"]
        for pre in ("excitatory", "inhibitory"):
            for post in ("excitatory", "inhibitory"):
                block = {(a, b) for a, b in original_edges if groups[a] == pre and groups[b] == post}
                by_group[f"{pre}->{post}"] = {"edges": len(block),
                                               "fraction_changed": len(block-null_edges)/len(block)}
        report["nulls"][str(seed)] = {
            "verified_loader": audit, "routing_equal": True,
            "node_and_readout_initial_parameters_equal": True,
            "initial_theta_multiset_equal": True, "parameter_budget_equal": True,
            "initial_EI_mass_max_absolute_difference": initial_mass_max_error,
            "by_group": by_group, "spatial_summary": spatial_summary(candidate)}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_bundle(args.bundle)
    atomic_json(args.output, report)
    print(json.dumps({"manifest_sha256": report["bundle_manifest_sha256"],
                      "seeds_verified": list(report["nulls"])}))
