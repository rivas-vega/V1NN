"""Conditional evaluation of frozen 3000-step models on previously unused clips."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pilot import atomic_json, source_identity
from v1nn.benchmark import PilotConfig, build_model, split_seed, tensor_hash
from v1nn.graph import Graph
from v1nn.model import FixedRetina
from v1nn.synthetic import moving_textures


def paired_interval(first, second, repetitions=2000, seed=20261109):
    """Resample whole clips after averaging paired seeds, not seed/clip cells."""
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] < 2:
        raise ValueError("Require matching seed-by-clip error matrices")
    difference = first-second
    per_clip = difference.mean(axis=0)
    rng = np.random.default_rng(seed)
    sampled = np.array([per_clip[rng.integers(len(per_clip), size=len(per_clip))].mean()
                        for _ in range(repetitions)])
    return {"mean_difference": float(per_clip.mean()),
            "per_seed_mean_difference": difference.mean(axis=1).tolist(),
            "clip_bootstrap_percentile_95": np.quantile(sampled, [.025, .975]).tolist(),
            "replicates": repetitions, "bootstrap_seed": seed,
            "unit": "independent clip; seeds averaged within clip",
            "scope": "conditional on these trained models and this column; not animal or training-population uncertainty"}


def confirm(campaign_folder, reference_folder, graph_path, predictions_path):
    torch.set_num_threads(1)
    campaign_folder, reference_folder = Path(campaign_folder), Path(reference_folder)
    seeds = (7, 8, 9)
    models = ("connectome", "degree_null", "no_edges", "gru")
    records = {}
    for model in models:
        for seed in seeds:
            folder = reference_folder if model == "gru" else campaign_folder
            report = json.loads((folder/f"{model}_{seed}.json").read_text())
            expected = 500 if model == "gru" else 3000
            if report["completed_steps"] != expected or report["status"] != "steps_completed":
                raise ValueError("Only the frozen complete training budgets are allowed")
            records[(model, seed)] = report
    for seed in seeds:
        validation = records[("connectome", seed)]["evaluation"]["validation"]
        if not all(validation["model"][metric] < validation["zero_velocity"][metric]
                   for metric in ("mean_squared_vector_error", "mean_endpoint_error")):
            raise ValueError("Preregistered entry condition is not met; leave new clips unexamined")
    original = records[("connectome", 7)]
    config = PilotConfig(**original["config"])
    graph_path = Path(graph_path)
    graph_hash = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    if graph_hash != original["graph_sha256"]:
        raise ValueError("Graph identity mismatch")
    with np.load(graph_path, allow_pickle=False) as archive:
        graph = Graph(len(archive["signs"]), archive["src"], archive["dst"], archive["signs"])
    seed = split_seed(20261110, "validation")
    if seed in [original["data"][s]["seed"] for s in ("train", "validation", "counterfactual")]:
        raise ValueError("Confirmation seed overlaps development")
    clips, _, target = moving_textures(batch=2048, steps=config.frames,
                                      size=config.image_size, seed=seed)
    with torch.no_grad():
        raw_features = FixedRetina(config.grid)(clips)
    del clips
    calibration = original["data"]["normalization"]
    x = (raw_features-torch.tensor(calibration["mean"]))/torch.tensor(calibration["std"])
    predictions, errors, trial_rows = {}, {}, []
    for model_name in models:
        errors[model_name] = {"mse": [], "epe": []}
        for model_seed in seeds:
            report = records[(model_name, model_seed)]
            if report["data"] != original["data"]:
                raise ValueError("Models do not share original development data and calibration")
            local_config = PilotConfig(**report["config"])
            if any(getattr(local_config, key) != getattr(config, key)
                   for key in ("frames", "image_size", "grid", "train_size", "data_seed")):
                raise ValueError("Input interface differs between models")
            folder = reference_folder if model_name == "gru" else campaign_folder
            checkpoint = folder/f"{model_name}_{model_seed}.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["config"] != report["config"] or saved["normalization"] != calibration:
                raise ValueError("Checkpoint/report identity mismatch")
            local_config.device = "cpu"
            model, _ = build_model(local_config, graph if model_name != "gru" else None)
            model.load_state_dict(saved["model"])
            model.eval()
            with torch.no_grad():
                prediction = torch.cat([model(chunk) for chunk in x.split(config.batch)])
            predictions[f"{model_name}_{model_seed}"] = prediction.numpy()
            mse = (prediction-target).double().square().sum(-1).numpy()
            epe = np.sqrt(mse)
            errors[model_name]["mse"].append(mse)
            errors[model_name]["epe"].append(epe)
            trial_rows.append({"model": model_name, "seed": model_seed,
                               "training_steps": report["completed_steps"],
                               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                               "prediction_sha256": tensor_hash(prediction),
                               "mean_squared_vector_error": float(mse.mean()),
                               "mean_endpoint_error": float(epe.mean())})
            print(json.dumps(trial_rows[-1]), flush=True)
    zero_mse = target.double().square().sum(-1).numpy()
    errors["zero"] = {"mse": np.tile(zero_mse, (3, 1)),
                      "epe": np.tile(np.sqrt(zero_mse), (3, 1))}
    comparisons = {other: {metric: paired_interval(errors["connectome"][metric], errors[other][metric])
                           for metric in ("epe", "mse")}
                   for other in ("degree_null", "no_edges", "zero", "gru")}
    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(predictions_path, targets=target.numpy(), **predictions)
    return {"purpose": "Confirmation of frozen configuration; no training or calibration on new clips",
            "source": source_identity(), "entry_condition_met": True,
            "primary_metric": "mean_endpoint_error", "secondary_metric": "mean_squared_vector_error",
            "data": {"clips": 2048, "data_seed": 20261110, "generator_seed": seed,
                     "features_sha256": tensor_hash(raw_features), "targets_sha256": tensor_hash(target),
                     "normalization": "original training calibration; unchanged", "new_velocities": False},
            "graph_sha256": graph_hash, "trials": trial_rows,
            "zero": {"mean_squared_vector_error": float(zero_mse.mean()),
                     "mean_endpoint_error": float(np.sqrt(zero_mse).mean())},
            "paired_differences_connectome_minus_other": comparisons,
            "predictions_file_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
            "limitations": ["One column and development-selected configuration",
                            "Same eight velocities; only textures are new",
                            "GRU trained for 500 steps, graph models for 3000",
                            "Intervals resample clips, not independent animals or trained models",
                            "If used for subsequent tuning, these clips become development data"],
            "environment": {"torch": torch.__version__, "device": "cpu"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("campaign_folder", "reference_folder", "graph", "predictions", "output"):
        parser.add_argument("--"+key.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    result = confirm(args.campaign_folder, args.reference_folder, args.graph, args.predictions)
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
