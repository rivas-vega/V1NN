"""CPU generation under the fixed entry-011 spatial-control protocol."""
import argparse
import shutil
from pathlib import Path

import numpy as np

from pilot import atomic_json, source_identity
from v1nn.nulls import IDENTITY, as_graph, audit_spatial_null, read_arrays, sha256
from v1nn.spatial_nulls import rewire_joint_spatial

BOUNDARIES = [50, 100, 200, 400]


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh directory; preserve previous attempts")
    output.mkdir(parents=True)
    shutil.copyfile(source, output/"original.npz")
    arrays = read_arrays(source)
    graph = as_graph(arrays)
    manifest = {"kind": "joint_node_EI_in_out_distance_v1", "source": source_identity(),
                "source_graph_sha256": sha256(source), "seeds": [7, 8, 9],
                "distance_boundaries_um": BOUNDARIES, "distance_boundary_side": "left",
                "swaps_per_edge": 5, "max_attempts_per_graph": 20000000,
                "max_seconds_per_graph": 300, "minimum_edge_change_fraction": 0.5, "nulls": {}}
    try:
        for seed in manifest["seeds"]:
            null, rewiring = rewire_joint_spatial(graph, 5*len(graph.src), arrays["groups"],
                arrays["positions_um"], BOUNDARIES, seed=seed, max_attempts=20000000, max_seconds=300)
            candidate = {key: arrays[key] for key in IDENTITY}
            candidate.update(src=null.src, dst=null.dst)
            audit = audit_spatial_null(arrays, candidate, BOUNDARIES, minimum_change=0.5)
            path = output/f"spatial_null_{seed}.npz"
            np.savez_compressed(path, **candidate)
            manifest["nulls"][str(seed)] = {"file": path.name, "sha256": sha256(path),
                                           "rewiring": rewiring, "audit": audit}
            print(f"seed={seed}, changed={audit['edge_change_fraction']:.3%}, "
                  f"attempts={rewiring['attempts']}, seconds={rewiring['seconds']:.2f}", flush=True)
        atomic_json(output/"manifest.json", manifest)
    except Exception as error:
        atomic_json(output/"failed_attempt.json", {"manifest_so_far": manifest,
                    "error": f"{type(error).__name__}: {error}", "gpu_permitted": False})
        raise
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/column_v1718.npz"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
