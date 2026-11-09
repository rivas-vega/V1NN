"""CPU diagnostic of parameter movement and gradients; no optimizer updates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pilot import atomic_json, source_identity
from v1nn.benchmark import PilotConfig, build_model, prepare_data, tensor_hash
from v1nn.graph import Graph


def audit(checkpoint_path, report_path, graph_path):
    torch.set_num_threads(1)
    checkpoint_path, graph_path = Path(checkpoint_path), Path(graph_path)
    original = json.loads(Path(report_path).read_text())
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved["config"] != original["config"]:
        raise ValueError("Checkpoint/report configuration mismatch")
    if hashlib.sha256(graph_path.read_bytes()).hexdigest() != original["graph_sha256"]:
        raise ValueError("Graph identity mismatch")
    with np.load(graph_path, allow_pickle=False) as archive:
        graph = Graph(len(archive["signs"]), archive["src"], archive["dst"], archive["signs"])
    config = PilotConfig(**saved["config"])
    config.device = "cpu"
    data, manifest = prepare_data(config)
    if manifest != original["data"] or manifest["normalization"] != saved["normalization"]:
        raise ValueError("Data/calibration reconstruction mismatch")
    model, _ = build_model(config, graph)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    initial_weights = model.core.effective_weights().detach().clone()
    model.load_state_dict(saved["model"])
    model.eval()
    # A fixed prefix, not a fresh random draw and not the validation partition.
    x, target = (value[:64] for value in data["train"])
    loss = (model(x)-target).square().sum(-1).mean()
    loss.backward()
    parameters = {}
    for name, parameter in model.named_parameters():
        if not parameter.numel():
            continue
        gradient = parameter.grad
        parameters[name] = {
            "elements": parameter.numel(),
            "initial_rms": float(initial[name].square().mean().sqrt()),
            "final_rms": float(parameter.detach().square().mean().sqrt()),
            "change_rms": float((parameter.detach()-initial[name]).square().mean().sqrt()),
            "gradient_l2": float(gradient.norm()) if gradient is not None else None,
            "gradient_rms": float(gradient.square().mean().sqrt()) if gradient is not None else None,
            "gradient_nonzero_fraction": float((gradient != 0).float().mean()) if gradient is not None else None}
    total_squared = sum(p["gradient_l2"]**2 for p in parameters.values() if p["gradient_l2"] is not None)
    for parameter in parameters.values():
        parameter["fraction_squared_gradient_norm"] = (
            parameter["gradient_l2"]**2/total_squared
            if total_squared and parameter["gradient_l2"] is not None else None)
    weights = model.core.effective_weights().detach()
    recurrence = {"edges": weights.numel()}
    if weights.numel():
        row_mass = weights.new_zeros(model.core.n).index_add(0, model.core.dst, weights.abs())
        recurrence.update({
            "relative_l2_effective_weight_change": float((weights-initial_weights).norm()/initial_weights.norm()),
            "mean_incoming_absolute_sum": float(row_mass.mean()),
            "max_incoming_absolute_sum": float(row_mass.max()),
            "fraction_rows_at_gain_cap": float((row_mass >= model.core.gain_cap-1e-6).float().mean())})
    return {"purpose": "Diagnostic gradients on fixed training prefix; no further training",
            "checkpoint_config": saved["config"], "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "analysis_source": source_identity(), "data_reconstruction_exact": True,
            "batch": {"split": "train", "indices": [0, len(x)], "stop_index_exclusive": True,
                      "features_sha256": tensor_hash(x), "targets_sha256": tensor_hash(target),
                      "mean_squared_vector_error": float(loss.detach())},
            "parameters": parameters, "recurrence": recurrence,
            "gradient_norm_before_clip": total_squared**0.5,
            "caveat": "Gradient sizes depend on parameterization. Dominant readout norm alone does not prove blocked core learning, especially with Adam.",
            "environment": {"torch": torch.__version__, "device": "cpu"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "report", "graph", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint, args.report, args.graph)
    atomic_json(args.output, result)
    print(json.dumps({"batch": result["batch"], "recurrence": result["recurrence"]}, indent=2))


if __name__ == "__main__":
    main()
