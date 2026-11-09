"""Complete frozen crossed design, with independent checkpoint and source checks."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from pilot import atomic_json
from v1nn.crossed import PAIRS, SEEDS, SPATIAL_MANIFEST, validate_crossed_plan
from v1nn.nulls import sha256

ROOT = Path(__file__).resolve().parents[1]


def summarize(measured, matrix):
    measured, matrix = np.asarray(measured, dtype=float), np.asarray(matrix, dtype=float)
    if measured.shape != (3,) or matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.isfinite(measured).all():
        raise ValueError("Require complete finite 3x3 matrix and three measured values")
    means = matrix.mean(axis=1)
    differences = measured-means
    return dict(measured_by_training_seed=measured.tolist(), null_matrix=matrix.tolist(),
                null_mean_by_training_seed=means.tolist(),
                null_std_within_training_seed=matrix.std(axis=1, ddof=1).tolist(),
                null_mean_by_graph_seed=matrix.mean(axis=0).tolist(),
                null_std_across_training_seed_means=float(means.std(ddof=1)),
                paired_measured_minus_null_mean=differences.tolist(),
                mean_paired_difference=float(differences.mean()),
                measured_mean=float(measured.mean()), null_mean=float(matrix.mean()),
                relative_reduction_of_mean_error=float(1-measured.mean()/matrix.mean()))


def source_hash(commit):
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True).splitlines()
    files = sorted(f for f in files if f.endswith(".py") and
                   (f.startswith("src/") or f.startswith("scripts/") and f.count("/") == 1))
    h = hashlib.sha256()
    for name in files:
        h.update(name.encode()+b"\0"+subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT))
    return h.hexdigest()


def analyze(original, diagonal, crossed, plan_path):
    folders = list(map(Path, (original, diagonal, crossed)))
    original, diagonal, crossed = folders
    plan = json.loads(Path(plan_path).read_text())
    validate_crossed_plan(plan)
    campaigns = [json.loads((folder/"campaign.json").read_text()) for folder in folders]
    assert campaigns[0]["run_id"] == plan["reuse_measured_run_id"]
    assert campaigns[1]["run_id"] == plan["reuse_diagonal_run_id"]
    assert campaigns[2]["crossed_plan"] == plan
    assert campaigns[2]["crossed_plan_sha256"] == sha256(plan_path)
    assert [t["file"] for t in campaigns[2]["trials"]] == [f"spatial_null_{r}_null_{g}.json" for r, g in PAIRS]
    assert all(c["status"] == "completed" and not c["source_dirty"] for c in campaigns)
    hashes = {str(folder): source_hash(c["source_commit"]) for folder, c in zip(folders, campaigns)}
    measured, nulls, identities = {}, {}, []
    for r in SEEDS:
        measured[r] = json.loads((original/f"connectome_{r}.json").read_text())
        for g in SEEDS:
            folder, name = (diagonal, f"spatial_null_{r}") if r == g else (crossed, f"spatial_null_{r}_null_{g}")
            nulls[r, g] = json.loads((folder/f"{name}.json").read_text())
        for g in [None]+SEEDS:
            ref = measured[r]
            if g is None:
                folder, name, result = original, f"connectome_{r}", ref
                graph_path = folder/"data/column_v1718.npz"
            else:
                folder, name = (diagonal, f"spatial_null_{r}") if r == g else (crossed, f"spatial_null_{r}_null_{g}")
                result = nulls[r, g]
                graph_path = folder/f"data/spatial_nulls/spatial_null_{g}.npz"
                info = result["model"]["prepared_null"]
                assert info["manifest_sha256"] == SPATIAL_MANIFEST
                assert info["source_graph_sha256"] == ref["graph_sha256"]
                assert result["config"].get("null_seed", r) == g
            cfg = result["config"]
            assert cfg["seed"] == r and result["status"] == "steps_completed" and result["completed_steps"] == 3000
            ignored = {"model", "max_seconds", "null_seed"}
            assert {k: v for k, v in cfg.items() if k not in ignored} == {k: v for k, v in ref["config"].items() if k not in ignored}
            assert result["data"] == ref["data"] and result["environment"] == ref["environment"]
            assert result["parameters"] == ref["parameters"]
            assert result["source"]["python_sources_sha256"] == hashes[str(folder)]
            assert sha256(graph_path) == result["graph_sha256"]
            ckpt_path = folder/f"{name}.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            assert ckpt["config"] == cfg and ckpt["completed_steps"] == 3000
            assert ckpt["normalization"] == result["data"]["normalization"]
            with np.load(graph_path, allow_pickle=False) as graph:
                for key, array in (("src", graph["src"]), ("dst", graph["dst"]), ("edge_sign", graph["signs"][graph["src"]])):
                    assert torch.equal(ckpt["model"]["core."+key], torch.from_numpy(array))
            if g is None:
                routing = ckpt["model"]["core.routing"]
            else:
                assert torch.equal(ckpt["model"]["core.routing"], routing)
                if r != g:
                    assert ckpt["prepared_null"] == result["model"]["prepared_null"]
            identities.append(dict(file=name, seed=r, null_seed=g, checkpoint_sha256=sha256(ckpt_path),
                                   source_and_graph_verified=True, paired_routing_verified=True))
    metrics = {}
    for split in ("train", "validation", "counterfactual"):
        metrics[split] = {}
        for metric in ("mean_endpoint_error", "mean_squared_vector_error"):
            m = [measured[r]["evaluation"][split]["model"][metric] for r in SEEDS]
            n = [[nulls[r, g]["evaluation"][split]["model"][metric] for g in SEEDS] for r in SEEDS]
            metrics[split][metric] = summarize(m, n)
    consistent = all(d < 0 for d in metrics["validation"]["mean_endpoint_error"]["paired_measured_minus_null_mean"])
    return dict(purpose="Development only; complete fixed design, no independent confirmation",
                training_routing_seeds=SEEDS, null_graph_seeds=SEEDS, plan_sha256=sha256(plan_path),
                null_manifest_sha256=SPATIAL_MANIFEST, reused_trials=6, new_trials=6,
                paired_config_data_environment_verified=True, checkpoint_audits=identities,
                metrics=metrics, consistent_lower_measured_epe_by_training_seed=consistent,
                decision=("Conditional consistent EPE; evaluate magnitude and limitations before any new confirmation" if consistent else
                          "Close this comparison without a robust fine-wiring advantage claim; audit sensory interface before scaling"),
                limits="Three graph realizations crossed with three combined routing/initialization/minibatch seeds. Cells are not nine independent replicates; no animal inference, equivalence or isolated routing effect. Reused development clips; remaining within-bin distance differences and uncertified null mixing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "diagonal", "crossed", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT/"configs/spatial_crossed_plan.json")
    args = parser.parse_args()
    report = analyze(args.original, args.diagonal, args.crossed, args.plan)
    atomic_json(args.output, report)
    print(json.dumps(report["metrics"]["validation"], indent=2))
