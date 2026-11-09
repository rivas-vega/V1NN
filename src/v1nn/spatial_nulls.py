"""Joint type/distance degree nulls: both incoming and outgoing per node."""
import math
import time

import numpy as np

from .graph import Graph


def validate_geometry(n, positions, boundaries):
    positions, boundaries = np.asarray(positions, dtype=float), np.asarray(boundaries, dtype=float)
    if (positions.ndim != 2 or positions.shape[0] != n or positions.shape[1] < 1 or
            not np.isfinite(positions).all() or boundaries.ndim != 1 or
            not np.isfinite(boundaries).all() or (boundaries <= 0).any() or
            (np.diff(boundaries) <= 0).any()):
        raise ValueError("Invalid finite positions or positive increasing distance boundaries")
    return positions, boundaries


def distance_ids(graph, positions, boundaries):
    positions, boundaries = validate_geometry(graph.n, positions, boundaries)
    distances = np.linalg.norm(positions[graph.src]-positions[graph.dst], axis=1)
    return np.searchsorted(boundaries, distances, side="left")


def joint_degrees(graph, groups, positions, boundaries):
    """Independent edge-based counting; no use of the sampler's bin matrix."""
    groups = np.asarray(groups)
    if groups.shape != (graph.n,):
        raise ValueError("groups must have length N")
    labels, group_ids = np.unique(groups, return_inverse=True)
    bins = distance_ids(graph, positions, boundaries)
    width = len(boundaries)+1
    results = []
    for endpoint, neighbor in ((graph.src, graph.dst), (graph.dst, graph.src)):
        codes = (endpoint*len(labels)+group_ids[neighbor])*width+bins
        results.append(np.bincount(codes, minlength=graph.n*len(labels)*width)
                       .reshape(graph.n, len(labels), width))
    return tuple(results)


def rewire_joint_spatial(graph, swaps, groups, positions, boundaries, seed=0,
                         max_attempts=20000000, max_seconds=300):
    """Symmetric proposals within fixed (pre-type, post-type, distance) strata.

    Select a stratum with probability proportional to m*(m-1), then an ordered
    pair of distinct edge slots uniformly. Reject unless both new distances
    retain that bin. All four edges thus share a bin, preserving BOTH ends'
    joint degrees. Strata stay fixed after each swap. No certified mixing claim.
    """
    if swaps < 0 or max_attempts < 1 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Invalid bounded swap budget")
    if graph.n > 4096:
        raise ValueError("Pairwise geometry cache limited to 4096 nodes")
    groups = np.asarray(groups)
    if groups.shape != (graph.n,) or np.any(graph.src == graph.dst):
        raise ValueError("Require groups per node and no autapses")
    positions, boundaries = validate_geometry(graph.n, positions, boundaries)
    if len(boundaries) > 254:
        raise ValueError("Too many bins for the bounded geometry cache")
    started = time.monotonic()
    distances = np.linalg.norm(positions[:, None, :]-positions[None, :, :], axis=2)
    bin_matrix = np.searchsorted(boundaries, distances, side="left").astype(np.uint8)
    del distances
    labels, group_ids = np.unique(groups, return_inverse=True)
    source, target = graph.src.copy(), graph.dst.copy()
    edge_bins = bin_matrix[source, target]
    assert np.array_equal(edge_bins, distance_ids(graph, positions, boundaries))
    codes = (group_ids[source]*len(labels)+group_ids[target])*(len(boundaries)+1)+edge_bins
    strata = [np.flatnonzero(codes == code) for code in np.unique(codes)]
    strata = [ids for ids in strata if len(ids) >= 2]
    if swaps and not strata:
        raise RuntimeError("No compatible edge pair for spatial rewiring")
    cumulative = np.cumsum([len(ids)*(len(ids)-1) for ids in strata], dtype=np.int64)
    edges = set(zip(source.tolist(), target.tolist()))
    original_edges = edges.copy()
    rng = np.random.default_rng(seed)
    accepted = attempts = 0
    while accepted < swaps and attempts < max_attempts:
        if attempts % 1024 == 0 and time.monotonic()-started >= max_seconds:
            raise RuntimeError(f"Spatial null time limit: {accepted}/{swaps} swaps, {attempts} proposals")
        block = strata[int(np.searchsorted(cumulative, rng.integers(int(cumulative[-1])), side="right"))]
        i = int(rng.integers(len(block)))
        j = int(rng.integers(len(block)-1))
        if j >= i:
            j += 1
        i, j = int(block[i]), int(block[j])
        attempts += 1
        a, b, c, d = int(source[i]), int(target[i]), int(source[j]), int(target[j])
        if a == c or b == d or a == d or c == b or (a, d) in edges or (c, b) in edges:
            continue
        distance_bin = edge_bins[i]
        if bin_matrix[a, d] != distance_bin or bin_matrix[c, b] != distance_bin:
            continue
        edges.remove((a, b)); edges.remove((c, d))
        edges.add((a, d)); edges.add((c, b))
        target[i], target[j] = d, b
        accepted += 1
    if accepted != swaps:
        raise RuntimeError(f"Spatial null attempt limit: {accepted}/{swaps} swaps, {attempts} proposals")
    candidate = Graph(graph.n, source, target, graph.signs.copy())
    old = joint_degrees(graph, groups, positions, boundaries)
    new = joint_degrees(candidate, groups, positions, boundaries)
    if not all(np.array_equal(a, b) for a, b in zip(old, new)):
        raise AssertionError("Joint incoming/outgoing type-distance degrees changed")
    return candidate, {"accepted_swaps": accepted, "attempts": attempts,
                       "seconds": time.monotonic()-started,
                       "edge_change_fraction": len(original_edges-edges)/max(1, len(edges)),
                       "proposal": "ordered edge pairs uniform conditional on fixed pre/post type and distance stratum; stratum weight m*(m-1)",
                       "compatible_strata": len(strata), "uniform_sampling_claim": False}
