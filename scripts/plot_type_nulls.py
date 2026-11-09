"""Paired graph-control results on the reused development validation split."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(summary, output, png=None, spatial_summary=None):
    report = json.loads(Path(summary).read_text())
    assert report["paired_data_config_environment_verified"]
    models = ("connectome", "degree_null", "type_null")
    labels = ("Conectoma", "Nulo de\ngrados", "Nulo de\ngrados + E/I")
    if spatial_summary:
        spatial = json.loads(Path(spatial_summary).read_text())
        assert spatial["paired_data_config_environment_verified"]
        assert spatial["models"]["connectome"] == report["models"]["connectome"]
        report["models"]["spatial_null"] = spatial["models"]["spatial_null"]
        models += ("spatial_null",)
        labels += ("Nulo E/I +\ndistancias",)
    count = len(models)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "v1nn-type-null"})
    fig, axes = plt.subplots(1, 2, figsize=(11.4 if spatial_summary else 9.6, 5.2), layout="constrained")
    colors = ("#197b8c", "#b65d34", "#775598")
    for ax, split, metric, title in zip(
            axes, ("train", "validation"),
            ("mean_squared_vector_error", "mean_endpoint_error"),
            ("Ajuste al entrenamiento · MSE", "Generalización · EPE")):
        values = np.array([report["models"][m][split][metric]["values_by_seed"] for m in models])
        for index, seed in enumerate(report["seeds"]):
            ax.plot(range(count), values[:, index], "o-", color=colors[index],
                    label=f"Semilla {seed}", alpha=.85, linewidth=1.4, markersize=6)
        ax.scatter(range(count), values.mean(1), color="#142f42", marker="_", s=300,
                   linewidths=2.5, zorder=4, label="Media")
        ax.set(xticks=range(count), xticklabels=labels, title=title, xlim=(-.3, count-.7))
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Error · menor es mejor")
        ax.margins(y=.25)
    axes[0].legend(frameon=False, fontsize=10, loc="best")
    fig.suptitle("Controles de tipo neuronal y organización espacial" if spatial_summary else
                 "¿Persiste la ventaja al conservar los grados E/I?", fontsize=17)
    fig.supxlabel("3000 pasos · datos e interfaz emparejados por semilla\n"
                  "Validación de desarrollo ya utilizada; no es un test nuevo.", fontsize=10)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Date": None})
    if png:
        Path(png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("reports/gpu_type_null/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/type_null_comparison.svg"))
    parser.add_argument("--png", type=Path)
    parser.add_argument("--spatial-summary", type=Path)
    args = parser.parse_args()
    plot(args.summary, args.output, args.png, args.spatial_summary)
