"""Paired development comparison; never promote this reused split to a new test."""
import argparse
import json
from pathlib import Path

import numpy as np

from pilot import atomic_json


def compare(original_folder, typed_folder):
    original_folder, typed_folder = Path(original_folder), Path(typed_folder)
    seeds = [7, 8, 9]
    models = ("connectome", "degree_null", "no_edges", "type_null")
    records = {}
    for model in models:
        folder = typed_folder if model == "type_null" else original_folder
        records[model] = [json.loads((folder/f"{model}_{seed}.json").read_text()) for seed in seeds]
    ignored_config = {"model", "max_seconds"}
    for index, seed in enumerate(seeds):
        reference = records["connectome"][index]
        for model in models:
            trial = records[model][index]
            assert trial["status"] == "steps_completed" and trial["completed_steps"] == 3000
            assert trial["config"]["seed"] == seed
            assert trial["data"] == reference["data"], "Different datasets or normalization"
            config = {k: v for k, v in trial["config"].items() if k not in ignored_config}
            refconfig = {k: v for k, v in reference["config"].items() if k not in ignored_config}
            assert config == refconfig, "Different training configuration"
            assert trial["environment"] == reference["environment"], "Different numerical environment"
            if model != "no_edges":
                assert trial["parameters"] == reference["parameters"]
            if model == "type_null":
                assert trial["model"]["prepared_null"]["source_graph_sha256"] == reference["graph_sha256"]
    result = {"purpose": "development validation reused; descriptive, not independent confirmation",
              "seeds": seeds, "all_trials_completed_steps": 3000,
              "paired_data_config_environment_verified": True,
              "models": {}, "paired_connectome_minus_type_null": {},
              "uncertainty": "Three seeds jointly change routing, initialization, batches and null realization; no animal/population inference or equivalence test."}
    for model in models:
        result["models"][model] = {}
        for split in ("train", "validation", "counterfactual"):
            result["models"][model][split] = {}
            for metric in ("mean_endpoint_error", "mean_squared_vector_error"):
                values = np.array([trial["evaluation"][split]["model"][metric] for trial in records[model]])
                result["models"][model][split][metric] = {
                    "values_by_seed": values.tolist(), "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1))}
    for metric in ("mean_endpoint_error", "mean_squared_vector_error"):
        measured = result["models"]["connectome"]["validation"][metric]
        null = result["models"]["type_null"]["validation"][metric]
        delta = np.array(measured["values_by_seed"])-null["values_by_seed"]
        result["paired_connectome_minus_type_null"][metric] = {
            "values_by_seed": delta.tolist(), "mean": float(delta.mean()),
            "relative_reduction_of_mean_error": 1-measured["mean"]/null["mean"]}
    delta = result["paired_connectome_minus_type_null"]["mean_endpoint_error"]["values_by_seed"]
    result["development_decision"] = (
        "consistent_lower_connectome_EPE; next audit spatial controls" if all(d < 0 for d in delta)
        else "no_consistent_connectome_advantage; first separate null and routing variation")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("reports/gpu_long_training"))
    parser.add_argument("--typed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.original, args.typed)
    atomic_json(args.output, report)
    print(json.dumps(report["paired_connectome_minus_type_null"], indent=2))
