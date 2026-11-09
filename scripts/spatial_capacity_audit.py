"""Upper bound on possible rewiring from saturated joint degree marginals."""
import argparse
from pathlib import Path

import numpy as np

from pilot import atomic_json
from prepare_spatial_nulls import BOUNDARIES
from v1nn.nulls import as_graph, read_arrays, sha256
from v1nn.spatial_nulls import distance_ids, joint_degrees


def audit(path):
    arrays = read_arrays(path); graph = as_graph(arrays)
    labels, groups = np.unique(arrays["groups"], return_inverse=True)
    positions = arrays["positions_um"]
    bins = np.searchsorted(BOUNDARIES, np.linalg.norm(positions[:, None]-positions[None, :], axis=2))
    capacity = np.zeros((graph.n, len(labels), len(BOUNDARIES)+1), dtype=int)
    for node in range(graph.n):
        eligible = np.arange(graph.n) != node
        for group in range(len(labels)):
            capacity[node, group] = np.bincount(bins[node, eligible & (groups == group)],
                                              minlength=len(BOUNDARIES)+1)
    outgoing, incoming = joint_degrees(graph, arrays["groups"], positions, BOUNDARIES)
    assert (outgoing <= capacity).all() and (incoming <= capacity).all()
    edge_bins = distance_ids(graph, positions, BOUNDARIES)
    forced_out = outgoing[graph.src, groups[graph.dst], edge_bins] == capacity[graph.src, groups[graph.dst], edge_bins]
    forced_in = incoming[graph.dst, groups[graph.src], edge_bins] == capacity[graph.dst, groups[graph.src], edge_bins]
    forced = forced_out | forced_in
    return {"source_graph_sha256": sha256(path), "distance_boundaries_um": BOUNDARIES,
            "edges": len(graph.src), "forced_by_saturated_outgoing_marginal": int(forced_out.sum()),
            "forced_by_saturated_incoming_marginal": int(forced_in.sum()),
            "forced_by_either_marginal": int(forced.sum()),
            "upper_bound_change_fraction": float(1-forced.mean()),
            "interpretation": "An existing edge must remain when its source or target uses every eligible neighbor in that type/distance bin. This is only an optimistic upper bound; additional constraints may force more edges."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/column_v1718.npz"))
    parser.add_argument("--output", type=Path, default=Path("reports/spatial_capacity_audit.json"))
    args = parser.parse_args()
    report = audit(args.source)
    atomic_json(args.output, report)
    print(report)
