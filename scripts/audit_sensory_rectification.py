"""Frozen CPU audit of first-state sensory Jacobians; no training or new split."""
import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from audit_routing_rank import routing_matrix
from pilot import atomic_json, source_identity
from v1nn.benchmark import PilotConfig, build_model, prepare_data, tensor_hash
from v1nn.nulls import as_graph, read_arrays, sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA256 = "57752a8941056a04d4bd1526b5672009e976bdffefaa87e846658ac1a5551899"


def sensory_preactivation(x, routing, gain, bias):
    return x[:, routing].mean(-1)*gain+bias


def active_singular_values(matrix, preactivation):
    """Derivative of ReLU away from boundaries; zero rows use PyTorch's convention."""
    active = preactivation > 0
    singular = torch.linalg.svdvals(matrix[active])
    # Missing rows imply zero singular values in ambient feature space.
    return torch.nn.functional.pad(singular, (0, matrix.shape[1]-len(singular)))


def rank_at_tolerance(singular, tolerance):
    return int((singular > singular.max()*tolerance).sum())


def quantiles(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    return dict(zip(("min", "q25", "median", "q75", "max"),
                    torch.quantile(values, torch.tensor([0, .25, .5, .75, 1], dtype=torch.float64)).tolist()))


def inspect(matrix, preactivation, tolerances, boundary_threshold, deadline):
    ranks = {str(t): [] for t in tolerances}
    minimum_ratios = []
    for z in preactivation:
        if time.monotonic() >= deadline:
            raise TimeoutError("CPU rectification audit reached its frozen time limit")
        singular = active_singular_values(matrix, z)
        minimum_ratios.append(float(singular[-1]/singular[0]) if singular[0] > 0 else 0.0)
        for threshold in tolerances:
            ranks[str(threshold)].append(rank_at_tolerance(singular, threshold))
    near = preactivation.abs() <= boundary_threshold
    boundary_rows = near.any(dim=1)
    result = dict(samples=len(preactivation), ambient_features=matrix.shape[1],
                  active_neurons=quantiles((preactivation > 0).sum(dim=1)),
                  near_boundary_preactivations=int(near.sum()),
                  near_boundary_fraction=float(near.double().mean()),
                  samples_with_near_boundary=int(boundary_rows.sum()),
                  exact_zero_preactivations=int((preactivation == 0).sum()),
                  smallest_to_largest_singular_ratio=quantiles(minimum_ratios), ranks={})
    for threshold, values in ranks.items():
        histogram = torch.bincount(torch.tensor(values), minlength=matrix.shape[1]+1)
        result["ranks"][threshold] = dict(
            histogram={str(i): int(v) for i, v in enumerate(histogram) if v},
            full_rank_fraction=values.count(matrix.shape[1])/len(values),
            rank_deficient_samples=sum(v < matrix.shape[1] for v in values),
            rank_deficient_away_from_boundary=sum(v < matrix.shape[1] and not bool(b) for v, b in zip(values, boundary_rows)))
    result["threshold_disagreement_samples"] = sum(a != b for a, b in zip(*ranks.values()))
    return result


def audit(checkpoints, plan_path, output):
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    if sha256(plan_path) != PLAN_SHA256:
        raise ValueError("Plan changed from frozen entry 015")
    plan = json.loads(Path(plan_path).read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    deadline = started+plan["max_seconds"]
    checkpoints = Path(checkpoints)
    if checkpoints.name != plan["source_measured_run_id"]:
        raise ValueError("Wrong frozen checkpoint run")
    prior = json.loads((ROOT/"reports/gpu_spatial_crossed/summary.json").read_text())
    expected_hashes = {r["file"]: r["checkpoint_sha256"] for r in prior["checkpoint_audits"]}
    graph_path = ROOT/"data/processed/column_v1718.npz"
    graph = as_graph(read_arrays(graph_path))
    first = json.loads((ROOT/"reports/gpu_long_training/connectome_7.json").read_text())
    config = PilotConfig(**{**first["config"], "device": "cpu"})
    data, manifest = prepare_data(config)
    assert manifest == first["data"] and config.data_seed == plan["data_seed"]
    count = plan["clips_per_split"]
    features = {split: data[split][0][:count, 0].contiguous() for split in plan["splits"]}
    assert all(len(x) == count and x.shape[1] == 32 for x in features.values())
    centered = features["train"].double()-features["train"].double().mean(dim=0)
    covariance = centered.T@centered/(count-1)
    report = dict(status="running", started_at_utc=started_at, purpose=plan["stage"], plan_sha256=PLAN_SHA256,
                  source=source_identity(), code_sha256=sha256(Path(__file__)),
                  environment=dict(python=platform.python_version(), torch=torch.__version__, device="cpu", threads=1),
                  unique_clips=2*count, first_frame_only=True,
                  selection=plan["selection"], repetitions_per_clip="three routing seeds, two sensory parameter conditions; not independent clips",
                  data_manifest=manifest, selected_features_sha256={s: tensor_hash(x) for s, x in features.items()},
                  training_covariance=dict(samples=count, frame=0, centered=True,
                      eigenvalues_descending=torch.linalg.eigvalsh(covariance).flip(0).tolist(),
                      empirical_rank=int(torch.linalg.matrix_rank(centered))),
                  thresholds=plan["rank_relative_singular_value_thresholds"],
                  near_zero_preactivation_threshold=plan["near_zero_preactivation_threshold"],
                  boundary_convention="ReLU derivative taken as zero at exactly zero; near-boundary counts are reported separately",
                  alpha_omitted_from_rank="Positive alpha scales every Jacobian singular value equally; relative rank unchanged",
                  seeds={}, limits=plan["interpretation"])
    partial = Path(output).with_suffix(".partial.json")
    try:
        for seed in plan["seeds"]:
            path = checkpoints/f"connectome_{seed}.pt"
            assert sha256(path) == expected_hashes[f"connectome_{seed}"]
            reference = json.loads((ROOT/f"reports/gpu_long_training/connectome_{seed}.json").read_text())
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            assert ckpt["config"] == reference["config"] and ckpt["completed_steps"] == 3000
            assert reference["data"] == manifest and ckpt["normalization"] == manifest["normalization"]
            assert reference["graph_sha256"] == sha256(graph_path)
            cfg = PilotConfig(**{**ckpt["config"], "device": "cpu"})
            assert cfg.seed == seed
            for key in ("data_seed", "train_size", "validation_size", "grid", "frames", "image_size"):
                assert getattr(cfg, key) == getattr(config, key)
            model, _ = build_model(cfg, graph)
            assert torch.equal(model.core.routing, ckpt["model"]["core.routing"])
            for key in ("src", "dst", "edge_sign"):
                assert torch.equal(getattr(model.core, key), ckpt["model"]["core."+key])
            row = dict(checkpoint_sha256=sha256(path), checkpoint_identity_verified=True, conditions={})
            for condition in ("initial", "trained"):
                if condition == "trained":
                    model.load_state_dict(ckpt["model"])
                model.eval()
                with torch.no_grad():
                    routing = model.core.routing
                    gain = torch.nn.functional.softplus(model.core.log_input_gain)/math.log(2)
                    bias = model.core.bias
                    assert torch.all(gain > 0)
                    if condition == "initial":
                        assert torch.equal(gain, torch.ones_like(gain)) and torch.count_nonzero(bias) == 0
                    matrix = routing_matrix(routing, 32)*gain.double()[:, None]
                    row["conditions"][condition] = {}
                    for split, x in features.items():
                        z = sensory_preactivation(x, routing, gain, bias)
                        expected = cfg.alpha*torch.relu(z)
                        _, states = model.core(x[:, None])
                        assert torch.equal(states[:, 0], expected), "Diagnostic differs from actual first state"
                        row["conditions"][condition][split] = inspect(
                            matrix, z, report["thresholds"], report["near_zero_preactivation_threshold"], deadline)
                        row["conditions"][condition][split]["actual_first_state_exactly_reproduced"] = True
            report["seeds"][str(seed)] = row
            atomic_json(partial, report)
        report["status"] = "completed"
        records = [r for seed in report["seeds"].values() for condition in seed["conditions"].values() for r in condition.values()]
        report["evaluated_jacobians"] = sum(r["samples"] for r in records)
        report["all_full_rank_at_both_tolerances"] = all(v["full_rank_fraction"] == 1 for r in records for v in r["ranks"].values())
        report["decision"] = plan["decision_if_full_rank"] if report["all_full_rank_at_both_tolerances"] else plan["decision_if_deficient"]
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        atomic_json(partial, report)
        raise
    finally:
        report["seconds"] = time.monotonic()-started
    atomic_json(output, report)
    partial.unlink(missing_ok=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--plan", type=Path, default=ROOT/"configs/sensory_rectification_plan.json")
    p.add_argument("--output", type=Path, default=ROOT/"reports/sensory_rectification_audit.json")
    a = p.parse_args()
    result = audit(a.checkpoints, a.plan, a.output)
    print(json.dumps({k: result[k] for k in ("status", "seconds", "evaluated_jacobians", "all_full_rank_at_both_tolerances", "decision")}, indent=2))
