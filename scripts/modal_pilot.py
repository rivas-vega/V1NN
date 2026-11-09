"""One bounded GPU campaign. Authentication stays on the submitting machine.

modal run --detach scripts/modal_pilot.py --models gru,static --seeds 7,8
"""
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
if modal.is_local():
    SOURCE_COMMIT = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    SOURCE_DIRTY = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True))
else:
    SOURCE_COMMIT, SOURCE_DIRTY = None, None

app = modal.App("bionn-v1nn-pilot")
volume = modal.Volume.from_name("bionn-v1nn-experiments", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "numpy==2.5.4", "pandas==3.1.1", "pyarrow==26.0.1")
         .env({"PYTHONPATH": "/root/v1nn/src:/root/v1nn/scripts"})
         .add_local_dir(ROOT/"src", "/root/v1nn/src", ignore=["**/__pycache__/**"])
         .add_local_dir(ROOT/"scripts", "/root/v1nn/scripts", ignore=["**/__pycache__/**"]))


def validate_campaign(models, seeds, steps, seconds, grid, alpha=0.3,
                      train_size=2048, batch=64, lr=0.003):
    allowed = {"static", "linear_history", "gru", "temporal_cnn", "connectome",
               "no_edges", "reset", "reservoir", "degree_null", "type_null", "spatial_null"}
    model_list = models.split(",")
    seed_list = [int(seed) for seed in seeds.split(",")]
    if not model_list or any(model not in allowed for model in model_list):
        raise ValueError("Unsupported model list")
    if not 1 <= len(model_list)*len(seed_list) <= 12:
        raise ValueError("Use between one and twelve trials per campaign")
    if len(set(model_list)) != len(model_list) or len(set(seed_list)) != len(seed_list):
        raise ValueError("Duplicate trials are not allowed")
    if not 1 <= steps <= 3000 or not 60 <= seconds <= 1500 or not 1 <= grid <= 8:
        raise ValueError("Bounds: 1..3000 steps, 60..1500 seconds, grid 1..8")
    if not 0 < alpha <= 1:
        raise ValueError("Require 0 < alpha <= 1")
    if not 1 <= train_size <= 4096 or not 1 <= batch <= 128:
        raise ValueError("Bounds: 1..4096 training clips, 1..128 batch size")
    if not math.isfinite(lr) or not 0 < lr <= 0.1:
        raise ValueError("Require finite 0 < learning rate <= 0.1")
    return [(model, seed) for seed in seed_list for model in model_list]


def validate_graph_choice(graph, models, null_manifest_sha256):
    prepared = {"typed_nulls": "type_null", "spatial_nulls": "spatial_null"}
    if graph not in ("synthetic", "microns", *prepared):
        raise ValueError("Choose synthetic, microns, typed_nulls or spatial_nulls")
    if graph in prepared:
        if models != prepared[graph] or not re.fullmatch(r"[0-9a-f]{64}", null_manifest_sha256):
            raise ValueError("Prepared graphs require the matching single model and pinned manifest SHA256")
    elif set(models.split(",")) & set(prepared.values()) or null_manifest_sha256:
        raise ValueError("Prepared null models and manifests require their prepared graph mode")


@app.function(image=image, gpu="T4", cpu=2, memory=8192, timeout=1800,
              max_containers=1, min_containers=0, scaledown_window=2,
              retries=0, volumes={"/results": volume})
def campaign(models: str, seeds: str, steps: int, seconds: int, grid: int, graph: str, alpha: float,
             train_size: int, batch: int, lr: float,
             source_commit: str, source_dirty: bool, null_manifest_sha256: str = "", crossed_plan_json: str = ""):
    import time
    import uuid
    from datetime import datetime, timezone
    from pilot import atomic_json, run
    from v1nn.benchmark import PilotConfig
    from v1nn.nulls import load_prepared_null
    from v1nn.crossed import crossed_trials

    trials = validate_campaign(models, seeds, steps, seconds, grid, alpha, train_size, batch, lr)
    validate_graph_choice(graph, models, null_manifest_sha256)
    trials = [(model, seed, None) for model, seed in trials]
    if crossed_plan_json:
        trials = crossed_trials(crossed_plan_json, models=models, seeds=seeds, steps=steps,
                                seconds=seconds, grid=grid, graph=graph, alpha=alpha,
                                train_size=train_size, batch=batch, lr=lr,
                                null_manifest_sha256=null_manifest_sha256)
    started = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    folder = Path("/results")/run_id
    summary = {"run_id": run_id, "status": "running", "gpu": "T4",
               "source_commit": source_commit, "source_dirty": source_dirty,
               "time_budget_seconds": seconds, "server_timeout_seconds": 1800,
               "trials": [], "graph": graph, "alpha": alpha,
               "train_size": train_size, "batch": batch, "lr": lr,
               "null_manifest_sha256": null_manifest_sha256 or None}
    if crossed_plan_json:
        summary["crossed_plan"] = json.loads(crossed_plan_json)
        summary["crossed_plan_sha256"] = hashlib.sha256(crossed_plan_json.encode()).hexdigest()
    atomic_json(folder/"campaign.json", summary)
    volume.commit()
    try:
        graph_path = None
        if graph in ("typed_nulls", "spatial_nulls"):
            bundle_root = "type-null-bundles" if graph == "typed_nulls" else "spatial-null-bundles"
            bundle = Path("/results")/bundle_root/null_manifest_sha256
            for seed in sorted({seed if null_seed is None else null_seed for _, seed, null_seed in trials}):
                load_prepared_null(bundle/f"{models}_{seed}.npz", seed, null_manifest_sha256, model=models)
            shutil.copytree(bundle, folder/"data"/graph)
            volume.commit()
        if graph == "microns":
            graph_path = folder/"data/column_v1718.npz"
            subprocess.run(["python", "/root/v1nn/scripts/prepare_microns.py",
                            "--cache", "/results/microns-cache",
                            "--output", str(graph_path)], check=True, timeout=180)
            volume.commit()
        for model, seed, null_seed in trials:
            graph_seed = seed if null_seed is None else null_seed
            if graph in ("typed_nulls", "spatial_nulls"):
                graph_path = folder/"data"/graph/f"{model}_{graph_seed}.npz"
            remaining = seconds-(time.monotonic()-started)
            if remaining < 30:
                summary["status"] = "time_limit"
                break
            name = f"{model}_{seed}" + (f"_null_{null_seed}" if null_seed is not None else "")
            output = folder/f"{name}.json"
            result = run(PilotConfig(model=model, seed=seed, null_seed=null_seed, steps=steps, grid=grid, alpha=alpha,
                                     train_size=train_size, batch=batch, lr=lr,
                                     device="cuda", max_seconds=min(remaining-20, 300)), output, graph_path)
            summary["trials"].append({"file": output.name,
                                      "completed_steps": result["completed_steps"],
                                      "status": result["status"], "evaluation": result["evaluation"]})
            atomic_json(folder/"campaign.json", summary)
            volume.commit()
        else:
            summary["status"] = "completed"
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary["seconds"] = time.monotonic()-started
        atomic_json(folder/"campaign.json", summary)
        volume.commit()
        print(json.dumps(summary, indent=2))
    return summary


@app.local_entrypoint()
def main(models: str = "gru,static,linear_history,temporal_cnn", seeds: str = "7,8,9",
         steps: int = 500, seconds: int = 1200, grid: int = 4, graph: str = "synthetic", alpha: float = 0.3,
         train_size: int = 2048, batch: int = 64, lr: float = 0.003, null_manifest_sha256: str = "", crossed_plan: str = ""):
    validate_campaign(models, seeds, steps, seconds, grid, alpha, train_size, batch, lr)
    validate_graph_choice(graph, models, null_manifest_sha256)
    payload = ""
    if crossed_plan:
        from v1nn.crossed import crossed_trials
        payload = Path(crossed_plan).read_text()
        crossed_trials(payload, models=models, seeds=seeds, steps=steps, seconds=seconds,
                       grid=grid, graph=graph, alpha=alpha, train_size=train_size,
                       batch=batch, lr=lr, null_manifest_sha256=null_manifest_sha256)
    if SOURCE_DIRTY:
        raise RuntimeError("Commit the reviewed source before launching paid GPU work")
    result = campaign.remote(models, seeds, steps, seconds, grid, graph, alpha,
                             train_size, batch, lr, SOURCE_COMMIT, SOURCE_DIRTY, null_manifest_sha256, payload)
    print(json.dumps(result, indent=2))
