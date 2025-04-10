"""
Analyzes "multi_dropout_results.csv" which has columns:
  [dropout_level, representation, alpha, accuracy_mean, accuracy_std]
Steps:
 1. Read the CSV into a DataFrame.
 2. Create Type A plots: for each dropout level, lines for each representation vs. alpha.
 3. Create Type B plots: for each representation, lines for each dropout level vs. alpha.
 4. Compute which alpha yields best accuracy for each (dropout_level, representation).
 5. Save a summary CSV ("best_alpha_summary.csv") and print stats.

No advanced stats; just basic comparisons and line plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Adjust paths as needed
RESULTS_CSV = "experiements/exp_0003/results/data/combined/multi_dropout_results.csv"
OUTPUT_DIR = "experiements/exp_0003/results/images/combined"
BEST_ALPHA_CSV = "experiements/exp_0003/results/data/combined/best_alpha_summary.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    if not os.path.isfile(RESULTS_CSV):
        print(f"Error: {RESULTS_CSV} does not exist.")
        return

    df = pd.read_csv(RESULTS_CSV)

    # Ensure correct dtypes
    df["alpha"] = df["alpha"].astype(float)
    df["accuracy_mean"] = df["accuracy_mean"].astype(float)
    df["accuracy_std"] = df["accuracy_std"].astype(float)

    # == 1) For each dropout_level, lines for each representation ==
    plot_accuracy_vs_alpha_by_dropout(df)

    # == 2) For each representation, lines for each dropout level ==
    plot_accuracy_vs_alpha_by_representation(df)

    # == 3) Find best alpha for each (dropout_level, representation) ==
    best_rows = []
    for dl in df["dropout_level"].unique():
        dl_df = df[df["dropout_level"] == dl]
        for rep in dl_df["representation"].unique():
            sub_df = dl_df[dl_df["representation"] == rep]
            best_idx = sub_df["accuracy_mean"].idxmax()
            best_alpha = sub_df.loc[best_idx, "alpha"]
            best_acc = sub_df.loc[best_idx, "accuracy_mean"]
            best_std = sub_df.loc[best_idx, "accuracy_std"]
            best_rows.append(
                {
                    "dropout_level": dl,
                    "representation": rep,
                    "best_alpha": best_alpha,
                    "best_accuracy_mean": round(best_acc, 3),
                    "best_accuracy_std": round(best_std, 3),
                }
            )

    # Save summary to CSV
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(BEST_ALPHA_CSV, index=False)
    print(f"\n[Saved best-alpha summary] -> {BEST_ALPHA_CSV}")

    # Print a quick console summary
    print("\n=== Best Alpha Summary (Dropout, Representation) ===")
    for row in best_rows:
        print(
            f"- {row['dropout_level']}, {row['representation']}: "
            f"alpha={row['best_alpha']}, accuracy={row['best_accuracy_mean']} ± {row['best_accuracy_std']}"
        )


def plot_accuracy_vs_alpha_by_dropout(df):
    """
    Creates one figure per dropout_level.
    X-axis: alpha
    Multiple lines: each representation
    """
    dropout_levels = df["dropout_level"].unique()
    for dl in dropout_levels:
        sub_df = df[df["dropout_level"] == dl]

        plt.figure()
        for rep in sorted(sub_df["representation"].unique()):
            rep_df = sub_df[sub_df["representation"] == rep].copy()
            rep_df.sort_values(by="alpha", inplace=True)
            alphas = rep_df["alpha"].values
            means = rep_df["accuracy_mean"].values
            stds = rep_df["accuracy_std"].values

            plt.errorbar(alphas, means, yerr=stds, label=rep, capsize=3)

        plt.xlabel("Alpha")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs. Alpha ({dl})")
        plt.ylim(0.7, 1.05)  # Adjust if needed
        plt.legend()
        outpath = os.path.join(OUTPUT_DIR, f"accuracy_vs_alpha_{dl}.png")
        plt.savefig(outpath)
        plt.close()
        print(f"[Saved figure for {dl}] -> {outpath}")


def plot_accuracy_vs_alpha_by_representation(df):
    """
    Creates one figure per representation.
    X-axis: alpha
    Multiple lines: each dropout level
    """
    representations = df["representation"].unique()
    for rep in representations:
        sub_df = df[df["representation"] == rep]

        plt.figure()
        for dl in sorted(sub_df["dropout_level"].unique()):
            dl_df = sub_df[sub_df["dropout_level"] == dl].copy()
            dl_df.sort_values(by="alpha", inplace=True)
            alphas = dl_df["alpha"].values
            means = dl_df["accuracy_mean"].values
            stds = dl_df["accuracy_std"].values

            plt.errorbar(alphas, means, yerr=stds, label=dl, capsize=3)

        plt.xlabel("Alpha")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs. Alpha for {rep}")
        plt.ylim(0.7, 1.05)
        plt.legend()
        outpath = os.path.join(OUTPUT_DIR, f"accuracy_vs_alpha_{rep}.png")
        plt.savefig(outpath)
        plt.close()
        print(f"[Saved figure for {rep}] -> {outpath}")


if __name__ == "__main__":
    main()
