"""Generate a 2D routing decision boundary figure for the learned classifier.

Sweeps max_hops (x-axis) and selectivity (y-axis, log scale) while fixing
has_traversal=1, avg_degree at training median, and all other features at
training-set median. The heatmap reveals the Regime-A boundary: ultra-selective
1-hop traversals (bottom-left) are routed to SQL while multi-hop traversals
consistently favour the GRAPH engine.

Usage:
    python experiments/plot_decision_boundary.py
"""

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.paths import LABELED_RUNS_CSV, RESULTS_DIR
from features.feature_extractor import FEATURE_NAMES


def main():
    # Load the production classifier
    model_path = os.path.join(PROJECT_ROOT, "model", "artifacts", "classifier_v1.pkl")
    clf = joblib.load(model_path)

    # Compute per-feature medians from training data (for fixed context)
    df = pd.read_csv(LABELED_RUNS_CSV)
    medians = df[FEATURE_NAMES].fillna(0).median().to_dict()

    # Fix traversal context: has_traversal=1
    medians["has_traversal"] = 1.0

    # Grid: max_hops on x-axis (continuous sweep), selectivity on y-axis (log)
    n_pts = 200
    max_hops_vals = np.linspace(0.5, 4.0, n_pts)        # 0.5 → 4 hops
    selectivity_vals = np.logspace(-3, 0, n_pts)         # 0.001 → 1.0

    # Predict over the grid
    rows = []
    for sel in selectivity_vals:
        for mh in max_hops_vals:
            row = dict(medians)
            row["max_hops"] = mh
            row["selectivity"] = sel
            rows.append(row)

    X_grid = pd.DataFrame(rows)[FEATURE_NAMES].fillna(0).values
    preds = clf.predict(X_grid)                   # returns integers: 0=SQL, 1=GRAPH
    Z = (preds == 1).astype(int).reshape(n_pts, n_pts)

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    cmap = mcolors.ListedColormap(["#cce5ff", "#ffe0cc"])   # blue=SQL, orange=GRAPH
    ax.pcolormesh(max_hops_vals, selectivity_vals, Z, cmap=cmap, shading="auto",
                  alpha=0.85)

    # Draw the decision boundary contour
    mh_mesh, sel_mesh = np.meshgrid(max_hops_vals, selectivity_vals)
    try:
        ax.contour(mh_mesh, sel_mesh, Z, levels=[0.5], colors=["#333333"],
                   linewidths=1.8, linestyles="--")
    except Exception:
        pass  # contour may fail if region is uniform

    ax.set_yscale("log")
    ax.set_xlabel(r"\texttt{max\_hops}", fontsize=11)
    ax.set_ylabel(r"\texttt{selectivity} (log scale)", fontsize=11)
    ax.set_title(
        r"Routing Decision: max\_hops $\times$ selectivity"
        "\n"
        r"(\texttt{has\_traversal}$=1$, others fixed at training median)",
        fontsize=10,
    )

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#cce5ff", edgecolor="k", label="SQL"),
        Patch(facecolor="#ffe0cc", edgecolor="k", label="GRAPH"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    plt.tight_layout()

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pdf_path = os.path.join(RESULTS_DIR, "decision_boundary.pdf")
    png_path = os.path.join(RESULTS_DIR, "decision_boundary.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to:")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")

    # Report SQL region characteristics
    sql_rows = np.where(Z[0, :] == 0)[0]
    if sql_rows.size:
        boundary_hop = max_hops_vals[sql_rows[-1]]
        print(f"SQL region extends to max_hops ≈ {boundary_hop:.2f} (at lowest selectivity)")
    else:
        print("No SQL region found at lowest selectivity row.")


if __name__ == "__main__":
    main()
