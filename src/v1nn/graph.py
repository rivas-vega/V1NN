"""Directed graphs and constrained nulls. Indices always mean pre -> post."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Graph:
    n: int
    src: np.ndarray
    dst: np.ndarray
    signs: np.ndarray

    def __post_init__(self):
        self.src = np.asarray(self.src)
        self.dst = np.asarray(self.dst)
        self.signs = np.asarray(self.signs, dtype=np.float32)
        if self.n < 1 or self.src.ndim != 1 or self.src.shape != self.dst.shape:
            raise ValueError("Invalid graph dimensions")
        if self.src.dtype.kind not in "iu" or self.dst.dtype.kind not in "iu":
            raise ValueError("Node indices must be integer arrays")
        self.src, self.dst = self.src.astype(np.int64), self.dst.astype(np.int64)
        if len(self.src) and (min(self.src.min(), self.dst.min()) < 0 or
                             max(self.src.max(), self.dst.max()) >= self.n):
            raise ValueError("Edge endpoint outside node table")
        if len(set(zip(self.src.tolist(), self.dst.tolist()))) != len(self.src):
            raise ValueError("Aggregate parallel synapses before constructing Graph")
        if self.signs.shape != (self.n,) or not np.isin(self.signs, [-1, 0, 1]).all():
            raise ValueError("Signs must be -1, 0 (unknown), or 1 per neuron")

    def degrees(self):
        return (np.bincount(self.src, minlength=self.n),
                np.bincount(self.dst, minlength=self.n))


def rewire(graph, swaps, seed=0, groups=None, positions=None, distance_bins=None,
           max_attempts=None):
    """Directed double swaps, preserving exact in/out degrees.

    Groups preserve each neuron's group-specific degrees. Distance constraints
    preserve each source edge's bin. Autapses stay untouched. This is NOT a
    certified uniform sample or proof of mixing. Fail on insufficient swaps.
    """
    if swaps < 0:
        raise ValueError("swaps must be nonnegative")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (graph.n,):
            raise ValueError("groups must have length N")
    if (positions is None) != (distance_bins is None):
        raise ValueError("Supply positions and distance_bins together")
    if positions is not None:
        positions = np.asarray(positions, dtype=float)
        distance_bins = np.asarray(distance_bins, dtype=float)
        if (positions.ndim != 2 or positions.shape[0] != graph.n or
            not np.isfinite(positions).all() or distance_bins.ndim != 1 or
            not np.isfinite(distance_bins).all() or
            (np.diff(distance_bins) <= 0).any()):
            raise ValueError("Invalid positions or strictly increasing distance bins")
    src, dst = graph.src.copy(), graph.dst.copy()
    edges = set(zip(src.tolist(), dst.tolist()))
    original = edges.copy()
    eligible = np.flatnonzero(src != dst)
    if swaps and len(eligible) < 2:
        raise ValueError("Too few non-autapse edges for rewiring")
    rng = np.random.default_rng(seed)
    limit = max_attempts if max_attempts is not None else max(1000, swaps * 100)
    done = attempts = 0
    def dbin(a, b):
        return np.searchsorted(distance_bins, np.linalg.norm(positions[a]-positions[b]))
    while done < swaps and attempts < limit:
        attempts += 1
        i, j = rng.choice(eligible, size=2, replace=False)
        a, b, c, d = int(src[i]), int(dst[i]), int(src[j]), int(dst[j])
        if a == c or b == d or a == d or c == b:
            continue
        if (a, d) in edges or (c, b) in edges:
            continue
        if groups is not None and (groups[a] != groups[c] or groups[b] != groups[d]):
            continue
        if positions is not None and (dbin(a,b) != dbin(a,d) or dbin(c,d) != dbin(c,b)):
            continue
        edges.remove((a,b)); edges.remove((c,d))
        edges.add((a,d)); edges.add((c,b))
        dst[i], dst[j] = d, b
        done += 1
    if done != swaps:
        raise RuntimeError(f"Null not generated: {done}/{swaps} swaps in {attempts} attempts")
    result = Graph(graph.n, src, dst, graph.signs.copy())
    if not all(np.array_equal(a,b) for a,b in zip(graph.degrees(), result.degrees())):
        raise AssertionError("Degree preservation failed")
    return result, {"accepted_swaps":done, "attempts":attempts,
                    "edge_change_fraction":len(original-edges)/max(1,len(edges)),
                    "uniform_sampling_claim":False}
