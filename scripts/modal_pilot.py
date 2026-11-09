"""One bounded GPU campaign. Authentication stays on the submitting machine.

modal run --detach scripts/modal_pilot.py --models gru,static --seeds 7,8
"""
import json
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


def validate_campaign(models, seeds, steps, seconds, grid, alpha=0.3):
    allowed = {"static", "linear_history", "gru", "temporal_cnn", "connectome",
               "no_edges", "reset", "reservoir", "degree_null"}
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
    return [(model, seed) for seed in seed_list for model in model_list]


@app.function(image=image, gpu="T4", cpu=2, memory=8192, timeout=1800,
              max_containers=1, min_containers=0, scaledown_window=2,
              retries=0, volumes={"/results": volume})
def campaign(models: str, seeds: str, steps: int, seconds: int, grid: int, graph: str, alpha: float,
             source_commit: str, source_dirty: bool):
    import time
    import uuid
    from datetime import datetime, timezone
    from pilot import atomic_json, run
    from v1nn.benchmark import PilotConfig

    trials = validate_campaign(models, seeds, steps, seconds, grid, alpha)
    if graph not in ("synthetic", "microns"):
        raise ValueError("Choose synthetic or microns")
    started = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    folder = Path("/results")/run_id
    summary = {"run_id": run_id, "status": "running", "gpu": "T4",
               "source_commit": source_commit, "source_dirty": source_dirty,
               "time_budget_seconds": seconds, "server_timeout_seconds": 1800,
               "trials": [], "graph": graph, "alpha": alpha}
    atomic_json(folder/"campaign.json", summary)
    volume.commit()
    try:
        graph_path = None
        if graph == "microns":
            graph_path = folder/"data/column_v1718.npz"
            subprocess.run(["python", "/root/v1nn/scripts/prepare_microns.py",
                            "--cache", "/results/microns-cache",
                            "--output", str(graph_path)], check=True, timeout=180)
            volume.commit()
        for model, seed in trials:
            remaining = seconds-(time.monotonic()-started)
            if remaining < 30:
                summary["status"] = "time_limit"
                break
            output = folder/f"{model}_{seed}.json"
            result = run(PilotConfig(model=model, seed=seed, steps=steps, grid=grid, alpha=alpha,
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
         steps: int = 500, seconds: int = 1200, grid: int = 4, graph: str = "synthetic", alpha: float = 0.3):
    validate_campaign(models, seeds, steps, seconds, grid, alpha)
    if graph not in ("synthetic", "microns"):
        raise ValueError("Choose synthetic or microns")
    if SOURCE_DIRTY:
        raise RuntimeError("Commit the reviewed source before launching paid GPU work")
    result = campaign.remote(models, seeds, steps, seconds, grid, graph, alpha, SOURCE_COMMIT, SOURCE_DIRTY)
    print(json.dumps(result, indent=2))
