"""
Analyzes combined modality results from:
  experiments/exp_0002/results/data/combined/results.csv

Since alpha=0 or alpha=1 effectively represent single-modality baselines,
we can compare them directly to intermediate alpha values (like 0.25, 0.5, 0.75).

Steps:
1) Read the CSV into a DataFrame
2) Plot accuracy vs alpha (with error bars) for each representation on a single figure
3) Identify best alpha for each representation
4) Print summary, including whether alpha=0 or alpha=1 is outperformed by some mid-range alpha
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

COMBINED_RESULTS_CSV = "experiments/exp_0002/results/data/combined/results.csv"
OUTPUT_PLOT = "experiments/exp_0002/results/images/combined/accuracy_vs_alpha.png"
OUTPUT_BEST_CSV = (
    "experiments/exp_0002/results/data/combined/best_alpha_summary_svm.csv"
)

os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_BEST_CSV), exist_ok=True)

# A dictionary to label your representations
REP_LABELS = {
    "HighImg-HighText": "Detailed Image + Detailed Text",
    "LowImg-LowText": "Degraded Image + Sparse Text",
    "LowImg-HighText": "Degraded Image + Detailed Text",
    "HighImg-LowText": "Detailed Image + Sparse Text",
    # Add more if needed
}


def main():
    # 1. Read CSV
    if not os.path.isfile(COMBINED_RESULTS_CSV):
        print(f"Error: {COMBINED_RESULTS_CSV} does not exist.")
        return

    df = pd.read_csv(COMBINED_RESULTS_CSV)
    df["alpha"] = df["alpha"].astype(float)
    df["accuracy_mean"] = df["accuracy_mean"].astype(float)
    df["accuracy_std"] = df["accuracy_std"].astype(float)

    # 2. Plot all representations on one figure
    plt.figure(figsize=(6.5, 4.5))

    all_means = []
    reps_sorted = sorted(df["representation"].unique())
    for rep in reps_sorted:
        sub_df = df[df["representation"] == rep].copy()
        sub_df.sort_values("alpha", inplace=True)

        alphas = sub_df["alpha"].values
        means = sub_df["accuracy_mean"].values
        stds = sub_df["accuracy_std"].values

        label_str = REP_LABELS.get(rep, rep)
        plt.errorbar(alphas, means, yerr=stds, label=label_str, capsize=3, marker="o")
        all_means.extend(means)

    # Auto-scale y-axis
    if all_means:
        y_min, y_max = float(np.min(all_means)), float(np.max(all_means))
        margin = 0.02 * (y_max - y_min) if (y_max - y_min) > 0 else 0.01
        plt.ylim(y_min - margin, y_max + margin)

    plt.xlabel("Alpha (0 = All Text, 1 = All Image)")
    plt.ylabel("Accuracy")
    plt.title("Combined Representations: Accuracy vs. Alpha\n(SVM)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"[Saved plot] {OUTPUT_PLOT}")

    # 3. Find best alpha for each representation
    best_rows = []
    print("\n=== Best Alpha for Each Representation (SVM) ===")
    for rep in reps_sorted:
        sub_df = df[df["representation"] == rep]
        if sub_df.empty:
            continue
        best_idx = sub_df["accuracy_mean"].idxmax()
        best_alpha = sub_df.loc[best_idx, "alpha"]
        best_acc = sub_df.loc[best_idx, "accuracy_mean"]
        best_std = sub_df.loc[best_idx, "accuracy_std"]

        label_str = REP_LABELS.get(rep, rep)
        print(
            f"  {label_str}: best alpha={best_alpha:.2f}, "
            f"accuracy={best_acc:.3f} ± {best_std:.3f}"
        )
        best_rows.append(
            {
                "representation": label_str,
                "best_alpha": best_alpha,
                "best_accuracy_mean": best_acc,
                "best_accuracy_std": best_std,
            }
        )

    # 4. Save summary
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(OUTPUT_BEST_CSV, index=False)
    print(f"[Saved best-alpha summary] {OUTPUT_BEST_CSV}")

    print(
        "\nNote: alpha=0 or alpha=1 are single-modality baselines; "
        "comparing them to alpha in {0.25, 0.5, 0.75} reveals whether a 'mixed' embedding "
        "beats purely text or purely image in SVM classification."
    )


if __name__ == "__main__":
    main()
