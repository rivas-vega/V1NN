"""Generate bounded, audited E/I-preserving nulls on CPU, before allocating GPU."""
import argparse
import shutil
from pathlib import Path

import numpy as np

from pilot import atomic_json, source_identity
from v1nn.graph import rewire
from v1nn.nulls import IDENTITY, as_graph, audit_type_null, read_arrays, sha256


def prepare(source, output, seeds=(7, 8, 9), swaps_per_edge=5,
            max_seconds=180, max_attempts=10000000, minimum_change=0.5):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Use a new bundle directory; do not overwrite audited graphs")
    output.mkdir(parents=True)
    shutil.copyfile(source, output/"original.npz")
    arrays = read_arrays(source)
    graph = as_graph(arrays)
    manifest = {"kind": "directed_degrees_and_node_specific_EI_v1",
                "source": source_identity(), "source_graph_sha256": sha256(source),
                "seeds": list(seeds), "swaps_per_edge": swaps_per_edge,
                "max_seconds_per_graph": max_seconds, "max_attempts_per_graph": max_attempts,
                "minimum_edge_change_fraction": minimum_change, "nulls": {}}
    for seed in seeds:
        null, rewiring = rewire(graph, swaps=swaps_per_edge*len(graph.src), seed=seed,
                               groups=arrays["groups"], max_attempts=max_attempts,
                               max_seconds=max_seconds)
        candidate = {key: arrays[key] for key in IDENTITY}
        candidate.update(src=null.src, dst=null.dst)
        audit = audit_type_null(arrays, candidate, minimum_change)
        path = output/f"type_null_{seed}.npz"
        np.savez_compressed(path, **candidate)
        manifest["nulls"][str(seed)] = {"file": path.name, "sha256": sha256(path),
                                       "rewiring": rewiring, "audit": audit}
        print(f"seed={seed}: changed={audit['edge_change_fraction']:.3%}, "
              f"attempts={rewiring['attempts']}, seconds={rewiring['seconds']:.2f}", flush=True)
    # A manifest exists only after the complete bundle passes all checks.
    atomic_json(output/"manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/column_v1718.npz"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
