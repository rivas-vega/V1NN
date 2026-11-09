"""Is the fixed population routing linearly information-losing before ReLU?"""
import argparse
import math
from pathlib import Path

import torch

from pilot import atomic_json
from v1nn.benchmark import PilotConfig, prepare_data
from v1nn.nulls import sha256


def routing_matrix(routing, features):
    matrix = torch.zeros(len(routing), features, dtype=torch.float64)
    matrix.scatter_add_(1, routing, torch.full(routing.shape, 1/routing.shape[1], dtype=torch.float64))
    return matrix


def diagnose(matrix, sensory_by_split, data):
    singular = torch.linalg.svdvals(matrix)
    rank = int(torch.linalg.matrix_rank(matrix))
    inverse = torch.linalg.pinv(matrix)
    result = dict(rank=rank, feature_dimensions=matrix.shape[1],
                  singular_values=singular.tolist(), condition_number=float(singular[0]/singular[-1]),
                  reconstruction={})
    for split, sensory in sensory_by_split.items():
        target = data[split][0].flatten(0, 1).double()
        reconstruction = sensory.flatten(0, 1).double()@inverse.T
        residual = reconstruction-target
        result["reconstruction"][split] = dict(
            max_absolute_error=float(residual.abs().max()),
            relative_frobenius_error=float(residual.norm()/target.norm()))
    return result


def audit(folder):
    torch.set_num_threads(1)
    records = {}
    for seed in (7, 8, 9):
        path = Path(folder)/f"connectome_{seed}.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        config = PilotConfig(**ckpt["config"])
        assert config.seed == seed and ckpt["completed_steps"] == 3000
        data, manifest = prepare_data(config)
        assert manifest["normalization"] == ckpt["normalization"]
        routing = ckpt["model"]["core.routing"]
        features = data["train"][0].shape[-1]
        matrix = routing_matrix(routing, features)
        gain = torch.nn.functional.softplus(ckpt["model"]["core.log_input_gain"])/math.log(2)
        assert torch.all(gain > 0)
        # Use the actual float32 gather/mean implementation, not only matrix arithmetic.
        sensory = {split: data[split][0][:, :, routing].mean(-1) for split in ("train", "validation")}
        initial = diagnose(matrix, sensory, data)
        trained = diagnose(matrix*gain.double()[:, None], {s: x*gain for s, x in sensory.items()}, data)
        records[str(seed)] = dict(checkpoint_sha256=sha256(path), initial_routing=initial,
                                 trained_positive_gain_before_bias_and_ReLU=trained,
                                 per_feature_number_of_routes=torch.bincount(routing.flatten(), minlength=features).tolist())
    return dict(question="Does fixed routing discard linear retina information before neuronal rectification?",
                scope="Initial routing and trained positive sensory gains only; excludes ReLU, recurrent state and readout.",
                data_seed=20261109, splits="Existing development training/validation; no fresh confirmation data",
                decision_rule="Full rank 32 and reconstruction relative error below 1e-6 rule out rank loss at this linear stage only",
                seeds=records,
                linear_stage_passed=all(r[k]["rank"] == 32 and all(m["relative_frobenius_error"] < 1e-6 for m in r[k]["reconstruction"].values()) for r in records.values() for k in ("initial_routing", "trained_positive_gain_before_bias_and_ReLU")),
                limitations="Invertibility across the whole population is not biological retinotopy, does not ensure recoverability after ReLU or through the recurrent model, and says nothing about optimization difficulty or anatomy-specific utility.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoints)
    atomic_json(args.output, result)
    print({"linear_stage_passed": result["linear_stage_passed"],
           "ranks": [r["initial_routing"]["rank"] for r in result["seeds"].values()]})
