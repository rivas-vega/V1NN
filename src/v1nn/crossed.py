"""Frozen six-trial development design; no seed search or repeated diagonal."""
import json

SPATIAL_MANIFEST = "076965c316800bc945c69c803033b37257a420e68f81fc0fb491d9fdbdc4a309"
SEEDS = [7, 8, 9]
PAIRS = [(r, g) for r in SEEDS for g in SEEDS if r != g]
FIXED = dict(model="spatial_null", null_bundle_manifest_sha256=SPATIAL_MANIFEST,
             training_routing_seeds=SEEDS, null_graph_seeds=SEEDS,
             reuse_diagonal_run_id="20261109T211945Z-5e0040f0",
             reuse_measured_run_id="20261109T192317Z-94f1cd63",
             steps=3000, data_seed=20261109, train_size=2048,
             validation_size=512, batch=64, lr=0.003, alpha=0.3, grid=4,
             campaign_seconds=1200, server_timeout_seconds=1800,
             gpu="T4", max_containers=1, retries=0)


def validate_crossed_plan(plan):
    if any(plan.get(key) != value for key, value in FIXED.items()):
        raise ValueError("Crossed plan differs from frozen configuration")
    expected = [dict(seed=r, null_seed=g) for r, g in PAIRS]
    if plan.get("new_trials_only") != expected:
        raise ValueError("Require exactly six frozen off-diagonal trials in order")
    return [("spatial_null", r, g) for r, g in PAIRS]


def crossed_trials(payload, *, models, seeds, steps, seconds, grid, graph,
                   alpha, train_size, batch, lr, null_manifest_sha256):
    plan = json.loads(payload)
    trials = validate_crossed_plan(plan)
    requested = (models, seeds, steps, seconds, grid, graph, alpha,
                 train_size, batch, lr, null_manifest_sha256)
    expected = ("spatial_null", "7,8,9", 3000, 1200, 4, "spatial_nulls",
                0.3, 2048, 64, 0.003, SPATIAL_MANIFEST)
    if requested != expected:
        raise ValueError("Launcher settings differ from frozen crossed plan")
    return trials
