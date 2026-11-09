"""Fit a linear readout to frozen saved graph states, with training-only scaling."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from pilot import atomic_json, source_identity
from v1nn.benchmark import PilotConfig, build_model, metrics, prepare_data
from v1nn.graph import Graph


def ridge_predictions(train, target, features, penalties=(0, 1e-4, 1e-2, 1, 100)):
    """Minimize mean squared vector error + lambda * ||W||_F^2.

    The intercept is unpenalized. All centering/scaling is fit on training only.
    Lambda zero uses a truncated pseudoinverse with relative threshold 1e-10
    on eigenvalues of the training Gram matrix; it is a numerical diagnostic.
    """
    if any(not math.isfinite(value) or value < 0 for value in penalties):
        raise ValueError("Ridge penalties must be finite and nonnegative")
    train, target = train.double(), target.double()
    mean, scale = train.mean(0), train.std(0).clamp_min(1e-8)
    target_mean = target.mean(0)
    z = (train-mean)/scale
    gram = z.T @ z / len(z)
    rhs = z.T @ (target-target_mean) / len(z)
    eigenvalues, vectors = torch.linalg.eigh(gram)
    cutoff = max(float(eigenvalues[-1])*1e-10, 1e-12)
    projected = vectors.T @ rhs
    predictions = {}
    for penalty in penalties:
        if penalty == 0:
            inverse = torch.where(eigenvalues > cutoff,
                                  eigenvalues.clamp_min(cutoff).reciprocal(), 0)
        else:
            inverse = (eigenvalues.clamp_min(0)+penalty).reciprocal()
        weights = vectors @ (inverse[:, None]*projected)
        predictions[str(penalty)] = {
            split: ((value.double()-mean)/scale) @ weights + target_mean
            for split, value in features.items()}
    return predictions, {"arithmetic": "float64", "scaling_fit_split": "train",
                         "penalty_objective": "mean squared vector error + lambda * squared Frobenius weight norm",
                         "numerical_rank": int((eigenvalues > cutoff).sum()),
                         "state_dimensions": train.shape[1], "gram_eigenvalue_cutoff": cutoff,
                         "gram_max_eigenvalue": float(eigenvalues[-1])}


def diagnose(checkpoint_path, report_path, graph_path):
    started = time.monotonic()
    torch.set_num_threads(1)
    original = json.loads(Path(report_path).read_text())
    checkpoint_path, graph_path = Path(checkpoint_path), Path(graph_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved["config"] != original["config"]:
        raise ValueError("Checkpoint/report configuration mismatch")
    config = PilotConfig(**saved["config"])
    if config.model not in ("connectome", "no_edges", "degree_null", "reset", "reservoir"):
        raise ValueError("This diagnostic requires a graph model")
    config.device = "cpu"
    if hashlib.sha256(graph_path.read_bytes()).hexdigest() != original["graph_sha256"]:
        raise ValueError("Graph identity mismatch")
    with np.load(graph_path, allow_pickle=False) as archive:
        graph = Graph(len(archive["signs"]), archive["src"], archive["dst"], archive["signs"])
    data, manifest = prepare_data(config)
    if manifest != original["data"] or manifest["normalization"] != saved["normalization"]:
        raise ValueError("Data/calibration reconstruction mismatch")
    model, _ = build_model(config, graph)
    model.load_state_dict(saved["model"])
    model.eval()
    states, original_metrics = {}, {}
    with torch.no_grad():
        for split, (x, target) in data.items():
            chunks, predictions = [], []
            for chunk in x.split(config.batch):
                predicted, hidden = model.core(chunk, reset_each_frame=model.reset)
                chunks.append(hidden[:, -1])
                predictions.append(predicted[:, -1])
            states[split] = torch.cat(chunks)
            original_metrics[split] = metrics(torch.cat(predictions), target)
    old = original["evaluation"]["validation"]["model"]["mean_squared_vector_error"]
    new = original_metrics["validation"]["mean_squared_vector_error"]
    if not math.isclose(old, new, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Original validation error is not reproduced")
    predictions, audit = ridge_predictions(states["train"], data["train"][1], states)
    evaluation = {penalty: {split: metrics(value, data[split][1].double())
                            for split, value in splits.items()}
                  for penalty, splits in predictions.items()}
    return {"purpose": "frozen saved core; CPU readout optimization diagnostic; validation used for tuning",
            "config": saved["config"], "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "original_report_sha256": hashlib.sha256(Path(report_path).read_bytes()).hexdigest(),
            "original_source": original["source"], "analysis_source": source_identity(),
            "data_reconstruction_exact": True, "original_readout": original_metrics,
            "ridge_evaluation": evaluation, "solver": audit, "seconds": time.monotonic()-started,
            "environment": {"torch": torch.__version__, "device": "cpu"}, "scientific_claim": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("checkpoint", "report", "graph", "output"):
        parser.add_argument("--"+field, type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.checkpoint, args.report, args.graph)
    atomic_json(args.output, result)
    print(json.dumps(result["ridge_evaluation"], indent=2))


if __name__ == "__main__":
    main()
