"""Frozen-checkpoint reversal diagnostic; no training, GPU or independent test."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from pilot import atomic_json, source_identity
from v1nn.benchmark import PilotConfig, build_model, metrics, prepare_data
from v1nn.nulls import as_graph, read_arrays, sha256

ROOT = Path(__file__).resolve().parents[1]


def reversal_metrics(forward, reverse, target):
    p, q, y = (v.double() for v in (forward, reverse, target))
    odd, even = (p-q)/2, (p+q)/2
    result = {"original": metrics(p, y), "reversed": metrics(q, -y),
              "odd_projection": metrics(odd, y),
              "even_component_squared_norm": float(even.square().sum(-1).mean())}
    mse = "mean_squared_vector_error"
    result["paired_mse"] = (result["original"][mse]+result["reversed"][mse])/2
    residual = result["paired_mse"]-result["odd_projection"][mse]-result["even_component_squared_norm"]
    assert abs(residual) < 1e-10
    result["paired_mse_identity_absolute_residual"] = abs(residual)
    return result


def trial_paths():
    work = ROOT.parent
    measured = work/"gpu-long-training/20261109T192317Z-94f1cd63"
    diagonal = work/"gpu-spatial/20261109T211945Z-5e0040f0"
    crossed = work/"gpu-spatial-crossed/20261109T230837Z-de15dfee"
    refs = work/"gpu-reference/20261109T183120Z-d416befc"
    for seed in (7, 8, 9):
        yield "measured", seed, None, measured, f"connectome_{seed}", measured/"data/column_v1718.npz"
        for g in (7, 8, 9):
            folder, name = ((diagonal, f"spatial_null_{seed}") if seed == g else
                            (crossed, f"spatial_null_{seed}_null_{g}"))
            yield "spatial_null", seed, g, folder, name, folder/f"data/spatial_nulls/spatial_null_{g}.npz"
        yield "gru", seed, None, refs, f"gru_{seed}", None


def summarize(trials):
    summary = {}
    metric = "mean_endpoint_error"
    for family in ("measured", "spatial_null", "gru"):
        rows = [t for t in trials if t["family"] == family]
        summary[family] = {}
        for split in ("validation", "counterfactual"):
            outcomes = {}
            for condition in ("original", "reversed", "odd_projection"):
                outcomes[condition] = {
                    key: float(np.mean([t["evaluation"][split][condition][key] for t in rows]))
                    for key in (metric, "mean_squared_vector_error")}
            outcomes["even_component_squared_norm"] = float(np.mean([
                t["evaluation"][split]["even_component_squared_norm"] for t in rows]))
            outcomes["relative_epe_reduction_by_training_seed"] = [
                1-np.mean([t["evaluation"][split]["odd_projection"][metric] for t in rows if t["seed"] == seed])/
                np.mean([t["evaluation"][split]["original"][metric] for t in rows if t["seed"] == seed])
                for seed in (7, 8, 9)]
            summary[family][split] = outcomes
    gate = all(
        min(summary[f]["validation"]["relative_epe_reduction_by_training_seed"]) >= .05 and
        summary[f]["counterfactual"]["odd_projection"][metric] <= summary[f]["counterfactual"]["original"][metric]
        for f in ("measured", "spatial_null"))
    return summary, gate


def main():
    torch.set_num_threads(1)
    started, cpu_started = time.monotonic(), time.process_time()
    plan_path = ROOT/"configs/time_reversal_plan.json"
    plan = json.loads(plan_path.read_text())
    deadline = started+plan["execution"]["max_seconds"]
    trials, routing_by_seed = [], {}
    data, manifest = prepare_data(PilotConfig(data_seed=plan["data_seed"]))
    for family, seed, graph_seed, folder, name, graph_path in trial_paths():
        report_path, checkpoint_path = folder/f"{name}.json", folder/f"{name}.pt"
        reference = json.loads(report_path.read_text())
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert reference["data"] == manifest
        assert saved["normalization"] == manifest["normalization"]
        assert saved["config"] == reference["config"]
        assert saved["completed_steps"] == reference["completed_steps"] == (500 if family == "gru" else 3000)
        config = PilotConfig(**{**saved["config"], "device": "cpu"})
        graph = None
        if graph_path:
            assert sha256(graph_path) == reference["graph_sha256"]
            graph = as_graph(read_arrays(graph_path))
        model, _ = build_model(config, graph)
        if graph_path:
            for key in ("src", "dst", "edge_sign", "routing"):
                assert torch.equal(getattr(model.core, key), saved["model"]["core."+key])
            if family == "measured":
                routing_by_seed[seed] = saved["model"]["core.routing"]
            else:
                assert config.effective_null_seed == graph_seed
                assert torch.equal(saved["model"]["core.routing"], routing_by_seed[seed])
        model.load_state_dict(saved["model"])
        model.eval()
        evaluation = {}
        with torch.inference_mode():
            for split in plan["splits"]:
                x, y = data[split]
                assert len(y) == plan["samples"][split]
                p, q = [], []
                for chunk in x.split(config.batch):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Frozen CPU budget exhausted; no automatic retry")
                    p.append(model(chunk))
                    q.append(model(chunk.flip(1)))
                evaluation[split] = reversal_metrics(torch.cat(p), torch.cat(q), y)
                for key, expected in reference["evaluation"][split]["model"].items():
                    assert np.isclose(evaluation[split]["original"][key], expected, rtol=1e-5, atol=1e-6)
        trials.append(dict(family=family, seed=seed, null_seed=graph_seed,
                           checkpoint_sha256=sha256(checkpoint_path), report_sha256=sha256(report_path),
                           graph_sha256=reference["graph_sha256"],
                           original_source=reference["source"], parameters=reference["parameters"],
                           completed_training_steps=reference["completed_steps"],
                           original_metrics_reproduced=True, evaluation=evaluation))
        print(f"Completed {name}", flush=True)
    summary, gate = summarize(trials)
    report = dict(purpose=plan["purpose"], plan_sha256=sha256(plan_path), source=source_identity(),
                  script_sha256=sha256(__file__), data=manifest, trials=trials, summary=summary,
                  candidate_gate_passed=gate, independent_confirmation=False,
                  seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
                  environment={"torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "threads": 1},
                  limitations=plan["limits"])
    atomic_json(ROOT/"reports/time_reversal_audit.json", report)
    print(json.dumps({"summary": summary, "gate": gate, "seconds": report["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
