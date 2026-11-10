"""Targeted follow-up: explain observed rank loss without repeating SVD or training."""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from audit_routing_rank import routing_matrix
from audit_sensory_rectification import sensory_preactivation
from pilot import atomic_json, source_identity
from v1nn.benchmark import PilotConfig, prepare_data, tensor_hash
from v1nn.nulls import sha256

ROOT = Path(__file__).resolve().parents[1]


def support_counts(matrix, preactivation):
    if torch.any(matrix < 0):
        raise ValueError("Coverage diagnostic requires nonnegative routing")
    covered = (preactivation > 0).double()@(matrix > 0).double() > 0
    return covered, covered.sum(dim=1)


def collision_witness(x, routing, gain, bias, matrix, z):
    """One constructive collision in ambient feature space, not a generated video."""
    covered, counts = support_counts(matrix, z)
    deficient = torch.nonzero(counts < x.shape[1]).flatten()
    if not len(deficient):
        return None
    index = int(deficient[0])
    feature = int(torch.nonzero(~covered[index]).flatten()[0])
    affected = matrix[:, feature] > 0
    if not torch.any(affected) or torch.any(z[index, affected] >= 0):
        raise ValueError("Witness requires routed feature with strictly negative affected units")
    margin = (-z[index, affected].double()/matrix[affected, feature]).min()
    epsilon = float(margin/4)
    altered = x[index:index+1].clone()
    altered[0, feature] += epsilon
    after = sensory_preactivation(altered, routing, gain, bias)
    before = z[index:index+1]
    assert torch.equal(torch.relu(before), torch.relu(after))
    assert torch.equal(before > 0, after > 0)
    assert not torch.equal(altered, x[index:index+1])
    return dict(selected_clip_index=index, feature_index=feature,
                applied_feature_delta=float(altered[0, feature]-x[index, feature]),
                relu_state_exactly_identical=True, active_set_exactly_identical=True,
                scope="Ambient feature-vector perturbation; not claimed to correspond to a realizable video")


def audit(checkpoints, primary_path):
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    primary = json.loads(Path(primary_path).read_text())
    assert primary["status"] == "completed"
    assert sha256(ROOT/"scripts/audit_sensory_rectification.py") == primary["code_sha256"]
    seed7 = torch.load(Path(checkpoints)/"connectome_7.pt", map_location="cpu", weights_only=True)
    data, manifest = prepare_data(PilotConfig(**{**seed7["config"], "device": "cpu"}))
    assert manifest == primary["data_manifest"]
    features = {split: data[split][0][:256, 0].contiguous() for split in ("train", "validation")}
    assert {s: tensor_hash(x) for s, x in features.items()} == primary["selected_features_sha256"]
    result = dict(purpose="Explain observed local rank deficiencies with active-feature coverage; no repeated SVD, new split or training",
                  primary_report_sha256=sha256(primary_path), source=source_identity(),
                  code_sha256=sha256(Path(__file__)), seeds={})
    for seed, previous in primary["seeds"].items():
        path = Path(checkpoints)/f"connectome_{seed}.pt"
        assert sha256(path) == previous["checkpoint_sha256"]
        state = torch.load(path, map_location="cpu", weights_only=True)["model"]
        routing = state["core.routing"]
        base = routing_matrix(routing, 32)
        result["seeds"][seed] = {}
        for condition, splits in previous["conditions"].items():
            gain = torch.ones(len(routing)) if condition == "initial" else torch.nn.functional.softplus(state["core.log_input_gain"])/math.log(2)
            bias = torch.zeros(len(routing)) if condition == "initial" else state["core.bias"]
            matrix = base*gain.double()[:, None]
            result["seeds"][seed][condition] = {}
            for split, reference in splits.items():
                if time.monotonic()-started > 60:
                    raise TimeoutError("Support follow-up exceeded 60 seconds")
                x = features[split]
                z = sensory_preactivation(x, routing, gain, bias)
                covered, counts = support_counts(matrix, z)
                hist = torch.bincount(counts, minlength=33)
                coverage_hist = {str(i): int(v) for i, v in enumerate(hist) if v}
                same = all(coverage_hist == r["histogram"] for r in reference["ranks"].values())
                # Rank cannot exceed nonzero-column count for any sample. Equal
                # sums on exactly the same samples establish equality pointwise.
                gaps = {t: int(counts.sum())-sum(int(k)*v for k, v in r["histogram"].items())
                        for t, r in reference["ranks"].items()}
                assert all(v >= 0 for v in gaps.values())
                result["seeds"][seed][condition][split] = dict(
                    active_feature_count_histogram=coverage_hist,
                    number_of_samples_missing_each_feature=(~covered).sum(dim=0).tolist(),
                    rank_histogram_matches_coverage_at_both_tolerances=same,
                    remaining_total_rank_deficit_after_missing_columns=gaps,
                    collision_witness=collision_witness(x, routing, gain, bias, matrix, z))
    records = [r for seed in result["seeds"].values() for condition in seed.values() for r in condition.values()]
    result["all_rank_loss_explained_by_missing_active_feature_columns"] = all(r["rank_histogram_matches_coverage_at_both_tolerances"] for r in records)
    result["constructive_collision_witnesses"] = sum(r["collision_witness"] is not None for r in records)
    result["seconds"] = time.monotonic()-started
    result["limits"] = "Sparse local failures in the first frame. No evidence that missing directions encode motion or that these events cause graph-comparison errors; no global invertibility or biological claim. Not an architecture-selection experiment."
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--primary", type=Path, default=ROOT/"reports/sensory_rectification_audit.json")
    p.add_argument("--output", type=Path, default=ROOT/"reports/rectification_support_audit.json")
    a = p.parse_args()
    report = audit(a.checkpoints, a.primary)
    atomic_json(a.output, report)
    print({k: report[k] for k in ("all_rank_loss_explained_by_missing_active_feature_columns", "constructive_collision_witnesses", "seconds")})
