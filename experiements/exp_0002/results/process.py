"""
Analyzes combined modality results from:
  experiements/exp_0002/results/data/combined/results.csv

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

COMBINED_RESULTS_CSV = "experiements/exp_0002/results/data/combined/results.csv"
OUTPUT_PLOT = "experiements/exp_0002/results/images/combined/accuracy_vs_alpha.png"
OUTPUT_BEST_CSV = (
    "experiements/exp_0002/results/data/combined/best_alpha_summary_svm.csv"
)


def main():
    # 1. Read the combined CSV
    if not os.path.isfile(COMBINED_RESULTS_CSV):
        print(f"Error: {COMBINED_RESULTS_CSV} does not exist.")
        return
    df = pd.read_csv(COMBINED_RESULTS_CSV)

    # Convert columns to numeric
    df["alpha"] = df["alpha"].astype(float)
    df["accuracy_mean"] = df["accuracy_mean"].astype(float)
    df["accuracy_std"] = df["accuracy_std"].astype(float)

    # 2. Plot accuracy vs. alpha for each representation
    plt.figure()
    representations = df["representation"].unique()
    for rep in representations:
        sub_df = df[df["representation"] == rep].copy()
        sub_df.sort_values(by="alpha", inplace=True)

        alphas = sub_df["alpha"].values
        means = sub_df["accuracy_mean"].values
        stds = sub_df["accuracy_std"].values

        plt.errorbar(alphas, means, yerr=stds, label=rep, capsize=3)

    plt.xlabel("Alpha")
    plt.ylabel("Accuracy")
    plt.title("Combined Representations: Accuracy vs. Alpha (SVM, Dropout 50)")
    plt.legend()
    plt.savefig(OUTPUT_PLOT)
    plt.close()
    print(f"[Saved plot] {OUTPUT_PLOT}")

    # 3. Identify best alpha for each representation
    best_rows = []
    print("\n=== Best Alpha for Each Combined Representation (SVM) ===")
    for rep in representations:
        sub_df = df[df["representation"] == rep]
        best_idx = sub_df["accuracy_mean"].idxmax()

        best_alpha = sub_df.loc[best_idx, "alpha"]
        best_acc = sub_df.loc[best_idx, "accuracy_mean"]
        best_std = sub_df.loc[best_idx, "accuracy_std"]

        print(
            f"  {rep}: best alpha={best_alpha:.2f}, "
            f"accuracy={best_acc:.3f} ± {best_std:.3f}"
        )

        best_rows.append(
            {
                "representation": rep,
                "best_alpha": best_alpha,
                "best_accuracy_mean": best_acc,
                "best_accuracy_std": best_std,
            }
        )

    # 4. Save best-alpha summary to CSV
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(OUTPUT_BEST_CSV, index=False)
    print(f"[Saved best-alpha summary] {OUTPUT_BEST_CSV}")

    print(
        "\nNote: alpha=0.0 or alpha=1.0 are single-modality baselines; "
        "this analysis checks if mid-range alpha improves classification."
    )


if __name__ == "__main__":
    main()
