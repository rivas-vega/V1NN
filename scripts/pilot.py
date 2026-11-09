"""Bounded, reproducible development trial; no final test or topology claims."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from v1nn.benchmark import (MODELS, PilotConfig, build_model, config_dict,
                            metrics, prepare_data)
from v1nn.graph import Graph


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def source_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(list((root/"src").rglob("*.py"))+list((root/"scripts").glob("*.py"))):
        digest.update(str(path.relative_to(root)).encode()+b"\0"+path.read_bytes())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty, "python_sources_sha256": digest.hexdigest()}


def run(config, output, graph_path=None):
    config.validate()
    torch.set_num_threads(1)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    started = time.perf_counter()
    graph, graph_hash = None, None
    if graph_path:
        graph_path = Path(graph_path)
        graph_hash = hashlib.sha256(graph_path.read_bytes()).hexdigest()
        with np.load(graph_path, allow_pickle=False) as data:
            graph = Graph(len(data["signs"]), data["src"], data["dst"], data["signs"])
    data, manifest = prepare_data(config)
    model, model_info = build_model(config, graph)
    model.to(config.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    train_x, train_y = (v.to(config.device) for v in data["train"])
    batches = torch.Generator().manual_seed(config.seed)
    history, completed = [], 0
    reason = "steps_completed"
    for step in range(config.steps):
        if time.perf_counter()-started >= config.max_seconds:
            reason = "time_limit"
            break
        ids = torch.randint(len(train_x), (config.batch,), generator=batches).to(config.device)
        prediction = model(train_x[ids])
        loss = (prediction-train_y[ids]).square().sum(-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        completed = step+1
        if step == 0 or completed % 50 == 0 or completed == config.steps:
            history.append({"step": completed, "training_loss": float(loss.detach())})
    model.eval()
    evaluations = {}
    with torch.no_grad():
        for split in ("validation", "counterfactual"):
            x, target = data[split]
            predictions = torch.cat([model(chunk.to(config.device)).cpu()
                                     for chunk in x.split(config.batch)])
            evaluations[split] = {"model": metrics(predictions, target),
                                  "zero_velocity": metrics(torch.zeros_like(target), target)}
    if config.device == "cuda":
        torch.cuda.synchronize()
    result = {"status": reason, "completed_steps": completed,
              "purpose": "development pilot; validation used for tuning; no final test",
              "config": config_dict(config), "source": source_identity(),
              "graph_sha256": graph_hash, "model": model_info,
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in parameters),
              "data": manifest, "training": history, "evaluation": evaluations,
              "seconds": time.perf_counter()-started,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "device": config.device,
                              "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name() if config.device == "cuda" else None},
              "scientific_claim": None}
    atomic_json(output, result)
    checkpoint = Path(output).with_suffix(".pt")
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config_dict(config), "completed_steps": completed,
                "normalization": manifest["normalization"]}, temporary)
    temporary.replace(checkpoint)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="gru")
    for key in ("seed", "data_seed", "steps", "batch", "train_size", "validation_size",
                "frames", "image_size", "grid", "hidden"):
        parser.add_argument("--"+key.replace("_", "-"), type=int, default=getattr(PilotConfig(), key))
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/pilot.json"))
    args = vars(parser.parse_args())
    output, graph = args.pop("output"), args.pop("graph")
    result = run(PilotConfig(**args), output, graph)
    print(json.dumps({k: result[k] for k in ("status", "completed_steps", "evaluation", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
