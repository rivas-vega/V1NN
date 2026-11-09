"""Plot all fixed crossed cells and the prespecified row-mean comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(report, svg, png):
    metric = report["metrics"]["validation"]["mean_endpoint_error"]
    matrix = np.array(metric["null_matrix"])
    measured = np.array(metric["measured_by_training_seed"])
    delta = measured[:, None]-matrix
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.1, 1]}, layout="constrained")
    limit = max(abs(delta.min()), abs(delta.max()))
    im = left.imshow(delta, cmap="RdBu_r", vmin=-limit, vmax=limit)
    for i in range(3):
        for j in range(3):
            left.text(j, i, f"{delta[i, j]:+.4f}\n({matrix[i, j]:.4f})", ha="center", va="center",
                      fontsize=10, color="white" if abs(delta[i, j]) > .65*limit else "black")
    left.set(xticks=range(3), yticks=range(3), xticklabels=[7, 8, 9], yticklabels=[7, 8, 9],
             xlabel="Semilla del grafo nulo", ylabel="Semilla de entrenamiento / routing",
             title="Diferencia EPE: medido − nulo\nEntre paréntesis: EPE del nulo")
    figure.colorbar(im, ax=left, shrink=.85, label="Negativo favorece al medido")
    for i in range(3):
        line, = right.plot([0, 1], [measured[i], matrix[i].mean()], "o-", label=f"Semilla {i+7}", lw=1.8)
        right.scatter(np.ones(3)*1.07, matrix[i], s=15, alpha=.55, color=line.get_color())
    right.set(xticks=[0, 1], xticklabels=["Medido", "Media de 3 nulos"], ylabel="EPE de validación (menor es mejor)",
              title=f"Contraste fijado por fila\nDiferencia media {metric['mean_paired_difference']:+.5f}")
    right.legend(frameon=False)
    right.grid(axis="y", alpha=.2)
    figure.suptitle("BIONN · Control espacial cruzado completo", fontsize=15)
    figure.supxlabel("Desarrollo reutilizado · 3 grafos × 3 semillas combinadas · sin inferencia biológica", fontsize=10)
    Path(svg).parent.mkdir(parents=True, exist_ok=True)
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(svg, metadata={"Date": None})
    figure.savefig(png, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--svg", type=Path, required=True)
    p.add_argument("--png", type=Path, required=True)
    a = p.parse_args()
    plot(json.loads(a.summary.read_text()), a.svg, a.png)
