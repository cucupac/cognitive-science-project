"""
Compare and visualize results from two experiments (Exp1: Logistic Regression, Exp2: SVM).

Steps:
1) Loads CSV data for both experiments.
2) Merges them into a single DataFrame with a 'classifier' column.
3) For each 'representation' (e.g., LowImg-HighText), plots Accuracy vs. Alpha
   with two lines (Logistic Regression vs. SVM).
4) Saves separate plots for each representation.
5) Prints summary statistics comparing the two classifiers at each alpha.

Usage:
  python compare_experiments.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

LOGREG_RESULTS = "experiments/exp_0002/log_reg/results/data/combined/results.csv"
SVM_RESULTS = "experiments/exp_0002/svm/results/data/combined/results.csv"
OUTPUT_DIR = "experiments/exp_0002/compare/images/"
SUMMARY_CSV = "experiments/exp_0002/compare/summary_comparison.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# A dictionary to label your representations
REP_LABELS = {
    "HighImg-HighText": "Detailed Image + Detailed Text",
    "LowImg-LowText": "Degraded Image + Sparse Text",
    "LowImg-HighText": "Degraded Image + Detailed Text",
    "HighImg-LowText": "Detailed Image + Sparse Text",
    # Add any other codes if necessary
}


def load_results(path, classifier_name):
    df = pd.read_csv(path)
    df["classifier"] = classifier_name
    df["alpha"] = df["alpha"].astype(float)
    df["accuracy_mean"] = df["accuracy_mean"].astype(float)
    df["accuracy_std"] = df["accuracy_std"].astype(float)
    return df


def plot_comparisons(df):
    """
    For each representation, create a figure:
      - x-axis: alpha (0 = All Text, 1 = All Image)
      - y-axis: accuracy
      - lines: Logistic Regression vs. SVM
    The title consists of two lines:
      1. "Performance Comparison: Logistic Regression vs. SVM"
      2. The representation (in bold and within parentheses)
    """
    unique_reps = sorted(df["representation"].unique())
    for rep in unique_reps:
        sub_df = df[df["representation"] == rep].copy()
        if sub_df.empty:
            continue

        # Prepare the figure
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        rep_label = REP_LABELS.get(rep, rep)

        # Gather means for auto-scaling
        all_means = []
        for clf in ["logistic_regression", "svm"]:
            clf_df = sub_df[sub_df["classifier"] == clf].copy()
            clf_df.sort_values("alpha", inplace=True)
            all_means.extend(clf_df["accuracy_mean"].values)

        if len(all_means) > 0:
            y_min, y_max = float(np.min(all_means)), float(np.max(all_means))
            margin = 0.02 * (y_max - y_min) if (y_max - y_min) > 0 else 0.01
            ax.set_ylim(y_min - margin, y_max + margin)

        # Plot each classifier
        for clf in ["logistic_regression", "svm"]:
            clf_df = sub_df[sub_df["classifier"] == clf].copy()
            clf_df.sort_values("alpha", inplace=True)
            alphas = clf_df["alpha"].values
            means = clf_df["accuracy_mean"].values
            stds = clf_df["accuracy_std"].values

            label_str = "Logistic Regression" if clf == "logistic_regression" else "SVM"
            ax.errorbar(
                alphas, means, yerr=stds, capsize=3, marker="o", label=label_str
            )

        ax.set_xlabel("Alpha (0 = All Text, 1 = All Image)")
        ax.set_ylabel("Accuracy")

        # Create a more intuitive title:
        # First line: a clear statement of what's being compared.
        # Second line: The representation label (in bold and within parentheses).
        rep_title_text = rep_label
        # Escape spaces so math text preserves them
        rep_title_text_escaped = rep_title_text.replace(" ", "\\ ")
        title = (
            "Performance Comparison: Logistic Regression vs. SVM\n"
            + r"$\mathbf{(Representation:\ "
            + rep_title_text_escaped
            + r")}$"
        )
        ax.set_title(title, fontsize=11)
        ax.legend(loc="best")
        plt.tight_layout()

        outpath = os.path.join(OUTPUT_DIR, f"comparison_{rep}.png")
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"[Saved figure for {rep}] -> {outpath}")


def print_clean_summary(df):
    """
    Print a table comparing Logistic Regression vs. SVM accuracy per alpha.
    Also writes out a CSV with row-by-row comparisons.
    """
    print("\n====== Accuracy Comparison: Logistic Regression vs SVM ======")
    summary_rows = []

    for rep in sorted(df["representation"].unique()):
        rep_df = df[df["representation"] == rep]
        if rep_df.empty:
            continue
        rep_label = REP_LABELS.get(rep, rep)

        alphas_sorted = sorted(rep_df["alpha"].unique())
        print(f"\nRepresentation: {rep_label}")
        print(f"{'Alpha':<6} | {'LogReg':>8} | {'SVM':>8} | {'Δ (SVM - LogReg)':>18}")
        print("-" * 45)

        for alpha in alphas_sorted:
            row_log = rep_df[
                (rep_df["alpha"] == alpha)
                & (rep_df["classifier"] == "logistic_regression")
            ]
            row_svm = rep_df[
                (rep_df["alpha"] == alpha) & (rep_df["classifier"] == "svm")
            ]
            if row_log.empty or row_svm.empty:
                continue

            acc_log = row_log.iloc[0]["accuracy_mean"]
            acc_svm = row_svm.iloc[0]["accuracy_mean"]
            delta = acc_svm - acc_log

            arrow = ""
            if delta > 0.002:
                arrow = "↑"
            elif delta < -0.002:
                arrow = "↓"

            print(
                f"{alpha:<6.2f} | {acc_log:>8.3f} | {acc_svm:>8.3f} | {delta:>+10.3f} {arrow}"
            )
            summary_rows.append(
                {
                    "representation": rep_label,
                    "alpha": alpha,
                    "logreg_acc": acc_log,
                    "svm_acc": acc_svm,
                    "delta": delta,
                }
            )

    out_df = pd.DataFrame(summary_rows)
    out_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n[Saved CSV summary] -> {SUMMARY_CSV}")


def main():
    if not os.path.isfile(LOGREG_RESULTS):
        print(f"Error: {LOGREG_RESULTS} does not exist.")
        return
    if not os.path.isfile(SVM_RESULTS):
        print(f"Error: {SVM_RESULTS} does not exist.")
        return

    df_logreg = load_results(LOGREG_RESULTS, "logistic_regression")
    df_svm = load_results(SVM_RESULTS, "svm")

    # Merge into one DataFrame
    df = pd.concat([df_logreg, df_svm], ignore_index=True)

    plot_comparisons(df)
    print_clean_summary(df)


if __name__ == "__main__":
    main()
