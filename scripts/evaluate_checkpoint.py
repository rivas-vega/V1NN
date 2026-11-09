"""Reevaluate a saved pilot without training or GPU; verify data identity."""
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


def evaluate(checkpoint_path, report_path, graph_path=None):
    started = time.monotonic()
    torch.set_num_threads(1)
    checkpoint_path, report_path = Path(checkpoint_path), Path(report_path)
    original = json.loads(report_path.read_text())
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved["config"] != original["config"]:
        raise ValueError("Checkpoint and original report configurations differ")
    config = PilotConfig(**saved["config"])
    config.device = "cpu"
    graph = None
    if original.get("graph_sha256"):
        if graph_path is None:
            raise ValueError("Original trial requires its graph")
        graph_path = Path(graph_path)
        if hashlib.sha256(graph_path.read_bytes()).hexdigest() != original["graph_sha256"]:
            raise ValueError("Graph identity differs from original trial")
        with np.load(graph_path, allow_pickle=False) as data:
            graph = Graph(len(data["signs"]), data["src"], data["dst"], data["signs"])
    data, manifest = prepare_data(config)
    if manifest != original["data"] or manifest["normalization"] != saved["normalization"]:
        raise ValueError("Cannot reconstruct the original data/calibration exactly")
    model, _ = build_model(config, graph)
    model.load_state_dict(saved["model"])
    model.eval()
    evaluations = {}
    with torch.no_grad():
        for split in ("train", "validation"):
            x, target = data[split]
            prediction = torch.cat([model(chunk) for chunk in x.split(config.batch)])
            evaluations[split] = {"model": metrics(prediction, target),
                                  "zero_velocity": metrics(torch.zeros_like(target), target)}
    old = original["evaluation"]["validation"]["model"]["mean_squared_vector_error"]
    new = evaluations["validation"]["model"]["mean_squared_vector_error"]
    if not math.isclose(old, new, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Saved weights do not reproduce original validation error")
    return {"purpose": "CPU reevaluation of existing checkpoint; no further training",
            "config": saved["config"], "training_steps": saved["completed_steps"],
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "original_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "original_source": original["source"], "evaluation_source": source_identity(),
            "data_reconstruction_exact": True,
            "validation_mse_absolute_difference": abs(old-new),
            "evaluation": evaluations, "seconds": time.monotonic()-started,
            "environment": {"torch": torch.__version__, "device": "cpu"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.report, args.graph)
    atomic_json(args.output, result)
    print(json.dumps(result["evaluation"], indent=2))


if __name__ == "__main__":
    main()
