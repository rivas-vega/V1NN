"""Plot matched development results; requires optional matplotlib installation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(reports, output, png=None):
    reports, output = Path(reports), Path(output)
    models = ("connectome", "degree_null", "no_edges")
    labels = ("Conectoma medido", "Nulo de grados", "Sin conexiones")
    folders = ("gpu_microns", "gpu_long_training")
    colors = ("#9aa6b2", "#197b8c")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "svg.fonttype": "none", "svg.hashsalt": "v1nn-training-budget"})
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    for index, folder in enumerate(folders):
        values = []
        for model in models:
            rows = [json.loads((reports/folder/f"{model}_{seed}.json").read_text())
                    for seed in (7, 8, 9)]
            expected_steps = 500 if index == 0 else 3000
            if any(row["completed_steps"] != expected_steps for row in rows):
                raise ValueError("Do not plot truncated trials as complete budgets")
            values.append([r["evaluation"]["validation"]["model"]["mean_squared_vector_error"]
                           for r in rows])
        values = np.array(values)
        positions = np.arange(3)+(index-.5)*.34
        ax.bar(positions, values.mean(1), .32, color=colors[index],
               label=f"{expected_steps} actualizaciones", zorder=2)
        for offset, seed_index in zip((-.065, 0, .065), range(3)):
            ax.scatter(positions+offset, values[:, seed_index], s=25, color="#172c3d",
                       edgecolor="white", linewidth=.5, zorder=3)
    baseline = json.loads((reports/"gpu_microns/connectome_7.json").read_text())
    zero = baseline["evaluation"]["validation"]["zero_velocity"]["mean_squared_vector_error"]
    ax.axhline(zero, color="#b55336", linestyle="--", linewidth=1.5,
               label="Predicción de velocidad cero", zorder=4)
    ax.set(xticks=np.arange(3), xticklabels=labels, ylim=(0, 2.2),
           ylabel="Error cuadrático de velocidad · menor es mejor")
    ax.set_title("Efecto de entrenar durante más pasos", loc="left", fontsize=17, pad=35)
    ax.text(0, 1.025, "Mismos datos y ajustes · barras: media · puntos: tres semillas",
            transform=ax.transAxes, fontsize=11, color="#45515c")
    ax.grid(axis="y", alpha=.16, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    fig.supxlabel("Validación utilizada para ajustar; no es la partición de confirmación.", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Date": None})
    if png:
        Path(png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=170)
    plt.close(fig)
    print(json.dumps({"matplotlib": matplotlib.__version__, "figure": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path, default=Path("reports/training_budget.svg"))
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    plot(args.reports, args.output, args.png)
