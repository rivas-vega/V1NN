"""CPU inference lesion of frozen measured-graph models; no new training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pilot import atomic_json
from v1nn.benchmark import PilotConfig, build_model, metrics, prepare_data
from v1nn.nulls import as_graph, read_arrays, sha256


def evaluate(checkpoints, reports, graph_path):
    checkpoints, reports = Path(checkpoints), Path(reports)
    torch.set_num_threads(1)
    graph = as_graph(read_arrays(graph_path))
    result = {"purpose": "Frozen-model inference lesion on reused development validation; no retraining or new confirmation",
              "intervention": "Set recurrent current scale to zero, preserving leak, sensory gains, biases and trained readout",
              "limitation": "A lesion can shift hidden-state distributions; this tests dependence of the trained predictor, not the ability of another model to learn or anatomical specificity.",
              "seeds": [7, 8, 9], "trials": {}}
    for seed in result["seeds"]:
        reference = json.loads((reports/f"connectome_{seed}.json").read_text())
        assert reference["completed_steps"] == 3000 and reference["graph_sha256"] == sha256(graph_path)
        config = PilotConfig(**{**reference["config"], "device": "cpu"})
        data, manifest = prepare_data(config)
        assert manifest == reference["data"]
        checkpoint_path = checkpoints/f"connectome_{seed}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["config"] == reference["config"] and checkpoint["completed_steps"] == 3000
        model, _ = build_model(config, graph)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        x, target = data["validation"]
        outcomes = {}
        with torch.no_grad():
            for label, scale in (("intact", 1.0), ("no_recurrent_current", 0.0)):
                prediction = torch.cat([model.core(chunk, edge_scale=scale)[0][:, -1]
                                        for chunk in x.split(config.batch)])
                outcomes[label] = metrics(prediction, target)
        expected = reference["evaluation"]["validation"]["model"]
        for key, value in expected.items():
            assert np.isclose(outcomes["intact"][key], value, rtol=1e-5, atol=1e-6)
        result["trials"][str(seed)] = {"checkpoint_sha256": sha256(checkpoint_path),
                                       "intact_reproduces_gpu_report": True, "evaluation": outcomes,
                                       "lesion_minus_intact": {key: outcomes["no_recurrent_current"][key]-value
                                                                for key, value in outcomes["intact"].items()}}
    result["mean_lesion_minus_intact"] = {
        key: float(np.mean([r["lesion_minus_intact"][key] for r in result["trials"].values()]))
        for key in ("mean_endpoint_error", "mean_squared_vector_error")}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--reports", type=Path, default=Path("reports/gpu_long_training"))
    parser.add_argument("--graph", type=Path, default=Path("data/processed/column_v1718.npz"))
    parser.add_argument("--output", type=Path, default=Path("reports/recurrent_lesion.json"))
    args = parser.parse_args()
    report = evaluate(args.checkpoints, args.reports, args.graph)
    atomic_json(args.output, report)
    print(json.dumps(report["mean_lesion_minus_intact"], indent=2))
