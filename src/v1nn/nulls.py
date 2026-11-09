"""Independently audit, hash and load CPU-prepared E/I nulls before training."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .graph import Graph
from .spatial_nulls import distance_ids, joint_degrees

IDENTITY = ("signs", "groups", "root_ids", "positions_um")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("src", "dst", *IDENTITY)}


def as_graph(data):
    return Graph(len(data["signs"]), data["src"], data["dst"], data["signs"])


def audit_type_null(original, candidate, minimum_change=0.5):
    before, after = as_graph(original), as_graph(candidate)
    if not 0 <= minimum_change <= 1:
        raise ValueError("Invalid change threshold")
    for key in IDENTITY:
        if not np.array_equal(original[key], candidate[key]):
            raise ValueError(f"Changed node identity: {key}")
    groups = original["groups"]
    if groups.shape != (before.n,) or set(groups) != {"excitatory", "inhibitory"}:
        raise ValueError("This audit requires explicitly annotated E/I groups")
    if not np.array_equal(before.signs, np.where(groups == "excitatory", 1, -1)):
        raise ValueError("Signs disagree with E/I groups")
    if np.any(before.src == before.dst) or np.any(after.src == after.dst):
        raise ValueError("This bundle requires graphs without autapses")
    for a, b in zip(before.degrees(), after.degrees()):
        if not np.array_equal(a, b):
            raise ValueError("Changed total directed degrees")
    for group in np.unique(groups):
        for endpoint, neighbor in (("src", "dst"), ("dst", "src")):
            def counts(data):
                return np.bincount(data[endpoint][groups[data[neighbor]] == group], minlength=before.n)
            if not np.array_equal(counts(original), counts(candidate)):
                raise ValueError("Changed node-specific E/I degrees")
    edges_before = set(zip(before.src.tolist(), before.dst.tolist()))
    edges_after = set(zip(after.src.tolist(), after.dst.tolist()))
    change = len(edges_before-edges_after)/max(1, len(edges_before))
    if change < minimum_change:
        raise ValueError("Too few changed edges for the preregistered method screen")
    return {"nodes": before.n, "edges": len(before.src), "edge_change_fraction": change,
            "total_directed_degrees_exact": True, "node_specific_EI_degrees_exact": True,
            "node_identity_exact": True, "autapses": 0, "duplicate_edges": 0,
            "minimum_change_screen": minimum_change, "spatial_constraints": False,
            "uniform_sampling_claim": False}


def audit_spatial_null(original, candidate, boundaries, minimum_change=0.5):
    audit = audit_type_null(original, candidate, minimum_change)
    old = joint_degrees(as_graph(original), original['groups'], original['positions_um'], boundaries)
    new = joint_degrees(as_graph(candidate), candidate['groups'], candidate['positions_um'], boundaries)
    if not all(np.array_equal(a, b) for a, b in zip(old, new)):
        raise ValueError("Changed joint node-specific type/distance degrees")
    graph = as_graph(original)
    bins = distance_ids(graph, original['positions_um'], boundaries)
    after_edges = set(zip(candidate['src'].tolist(), candidate['dst'].tolist()))
    by_stratum = {}
    groups = original['groups']
    for pre in np.unique(groups):
        for post in np.unique(groups):
            for index in range(len(boundaries)+1):
                mask = (groups[graph.src] == pre) & (groups[graph.dst] == post) & (bins == index)
                before_edges = set(zip(graph.src[mask].tolist(), graph.dst[mask].tolist()))
                changed = len(before_edges-after_edges)
                by_stratum[f"{pre}->{post}/bin_{index}"] = {
                    "original_edges": len(before_edges), "changed_edges": changed,
                    "fraction_changed": changed/len(before_edges) if before_edges else None}
    audit['change_by_type_distance_stratum'] = by_stratum
    audit.update(spatial_constraints=True, joint_in_out_type_distance_degrees_exact=True,
                 distance_boundaries_um=list(boundaries), distance_boundary_side="left",
                 exact_continuous_distances_preserved=False)
    return audit


def load_type_null(path, seed, expected_manifest_sha256=None):
    return load_prepared_null(path, seed, expected_manifest_sha256, model="type_null")


def load_spatial_null(path, seed, expected_manifest_sha256=None):
    return load_prepared_null(path, seed, expected_manifest_sha256, model="spatial_null")


def load_prepared_null(path, seed, expected_manifest_sha256=None, *, model="type_null"):
    path = Path(path)
    manifest_path = path.parent/"manifest.json"
    manifest_hash = sha256(manifest_path)
    if expected_manifest_sha256 is not None and manifest_hash != expected_manifest_sha256:
        raise ValueError("Null bundle manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    kinds = {"type_null": "directed_degrees_and_node_specific_EI_v1",
             "spatial_null": "joint_node_EI_in_out_distance_v1"}
    if model not in kinds or manifest["kind"] != kinds[model]:
        raise ValueError("Unknown null bundle kind")
    entry = manifest["nulls"][str(seed)]
    if path.name != f"{model}_{seed}.npz" or entry["file"] != path.name:
        raise ValueError("Null seed/file mismatch")
    source = path.parent/"original.npz"
    if sha256(source) != manifest["source_graph_sha256"] or sha256(path) != entry["sha256"]:
        raise ValueError("Null bundle graph hash mismatch")
    original, candidate = read_arrays(source), read_arrays(path)
    if model == "spatial_null":
        if manifest["distance_boundary_side"] != "left":
            raise ValueError("Unknown bin boundary convention")
        audit = audit_spatial_null(original, candidate, manifest["distance_boundaries_um"],
                                   manifest["minimum_edge_change_fraction"])
    else:
        audit = audit_type_null(original, candidate, manifest["minimum_edge_change_fraction"])
    if audit != entry["audit"]:
        raise ValueError("Stored null audit differs from independent audit")
    if entry["rewiring"]["accepted_swaps"] != manifest["swaps_per_edge"]*audit["edges"]:
        raise ValueError("Incomplete null generation")
    return as_graph(candidate), {"manifest_sha256": manifest_hash,
                                "source_graph_sha256": manifest["source_graph_sha256"],
                                "null_graph_sha256": entry["sha256"], "seed": seed,
                                "audit": audit, "rewiring": entry["rewiring"]}
