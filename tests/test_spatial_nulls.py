import json

import numpy as np
import pytest

from v1nn.graph import Graph
from v1nn.nulls import as_graph, audit_spatial_null, load_spatial_null, load_type_null, sha256
from v1nn.spatial_nulls import distance_ids, joint_degrees, rewire_joint_spatial


def fixture():
    rng = np.random.default_rng(120)
    n = 80
    positions = rng.uniform(0, 100, (n, 2))
    mask = rng.random((n, n)) < 0.15
    np.fill_diagonal(mask, False)
    source, target = np.nonzero(mask)
    groups = np.where(np.arange(n) % 3, "excitatory", "inhibitory")
    return dict(src=source, dst=target, groups=groups, positions_um=positions,
                signs=np.where(groups == "excitatory", 1, -1), root_ids=np.arange(n).astype(str))


def test_spatial_null_joint_degrees_and_repeatability():
    arrays = fixture()
    graph = as_graph(arrays)
    args = (graph, len(graph.src)*5, arrays["groups"], arrays["positions_um"], [25, 50, 100])
    result, report = rewire_joint_spatial(*args, seed=7, max_seconds=10, max_attempts=100000)
    repeated, second = rewire_joint_spatial(*args, seed=7, max_seconds=10, max_attempts=100000)
    assert np.array_equal(result.dst, repeated.dst)
    assert report["attempts"] == second["attempts"]
    # Explicit node/group/bin checks independent of the vectorized audit.
    for node in range(graph.n):
        for group in np.unique(arrays["groups"]):
            for distance_bin in range(4):
                for direction in ("in", "out"):
                    counts = []
                    for candidate in (graph, result):
                        endpoint, neighbor = ((candidate.dst, candidate.src) if direction == "in"
                                              else (candidate.src, candidate.dst))
                        bins = distance_ids(candidate, arrays["positions_um"], [25, 50, 100])
                        counts.append(int(((endpoint == node) & (arrays["groups"][neighbor] == group)
                                           & (bins == distance_bin)).sum()))
                    assert counts[0] == counts[1]
    assert report["edge_change_fraction"] > 0.5 and not report["uniform_sampling_claim"]
    candidate = {**arrays, "src": result.src, "dst": result.dst}
    assert audit_spatial_null(arrays, candidate, [25, 50, 100])["joint_in_out_type_distance_degrees_exact"]


def test_outgoing_only_distance_control_is_rejected():
    arrays = dict(src=np.array([0, 2]), dst=np.array([1, 3]), signs=np.array([1, -1, 1, -1]),
                  groups=np.array(["excitatory", "inhibitory", "excitatory", "inhibitory"]),
                  positions_um=np.array([[0, 0], [1, 0], [10, 0], [2, 0]]), root_ids=np.arange(4))
    candidate = {**arrays, "dst": np.array([3, 1])}
    old = joint_degrees(as_graph(arrays), arrays["groups"], arrays["positions_um"], [5])
    new = joint_degrees(as_graph(candidate), arrays["groups"], arrays["positions_um"], [5])
    assert np.array_equal(old[0], new[0]) and not np.array_equal(old[1], new[1])
    with pytest.raises(ValueError, match="joint node-specific"):
        audit_spatial_null(arrays, candidate, [5])


def test_spatial_bundle_cannot_be_loaded_as_only_typed_null(tmp_path):
    arrays = fixture(); graph = as_graph(arrays); boundaries = [25, 50, 100]
    result, rewiring = rewire_joint_spatial(graph, len(graph.src)*5, arrays["groups"],
                                           arrays["positions_um"], boundaries, seed=7, max_seconds=10)
    candidate = {**arrays, "src": result.src, "dst": result.dst}
    np.savez(tmp_path/"original.npz", **arrays)
    path = tmp_path/"spatial_null_7.npz"; np.savez(path, **candidate)
    manifest = {"kind": "joint_node_EI_in_out_distance_v1", "swaps_per_edge": 5,
                "minimum_edge_change_fraction": 0.5, "distance_boundaries_um": boundaries,
                "distance_boundary_side": "left", "source_graph_sha256": sha256(tmp_path/"original.npz"),
                "nulls": {"7": {"file": path.name, "sha256": sha256(path), "rewiring": rewiring,
                                  "audit": audit_spatial_null(arrays, candidate, boundaries)}}}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    loaded, _ = load_spatial_null(path, 7, sha256(tmp_path/"manifest.json"))
    assert np.array_equal(loaded.dst, result.dst)
    with pytest.raises(ValueError, match="kind"):
        load_type_null(path, 7)


def test_spatial_limits_and_boundary_convention(monkeypatch):
    graph = Graph(4, np.array([0, 2]), np.array([1, 3]), np.ones(4))
    positions = np.array([[0, 0], [5, 0], [10, 0], [15, 0]])
    assert np.array_equal(distance_ids(graph, positions, [5]), [0, 0])
    with pytest.raises(RuntimeError, match="attempt limit"):
        rewire_joint_spatial(graph, 10, np.zeros(4), positions, [5], max_attempts=1)
    clock = iter([0.0, 2.0])
    monkeypatch.setattr("v1nn.spatial_nulls.time.monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="time limit"):
        rewire_joint_spatial(graph, 1, np.zeros(4), positions, [5], max_seconds=1)
