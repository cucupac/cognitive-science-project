"""
Compare and visualize results from two experiments (Exp1: Logistic Regression, Exp2: SVM).

It does the following:
1) Loads CSV data for both experiments.
2) Merges them into a single DataFrame with a 'classifier' column.
3) For each 'representation' (e.g. LowImg-HighText), plots Accuracy vs. Alpha
   with two lines on the same chart (Logistic Regression vs. SVM).
4) Saves separate plots for each representation.
5) Prints summary statistics comparing the two classifiers at each alpha.

Usage:
  python compare_experiments.py

You may need to adjust file paths if your directory structure differs.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

LOGREG_RESULTS = "experiements/exp_0001/results/data/combined/results.csv"
SVM_RESULTS = "experiements/exp_0002/results/data/combined/results.csv"
OUTPUT_DIR = "experiements/compare/images/"
SUMMARY_CSV = "experiements/compare/summary_comparison.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_results(path, classifier_name):
    df = pd.read_csv(path)
    df["classifier"] = classifier_name
    df["alpha"] = df["alpha"].astype(float)
    df["accuracy_mean"] = df["accuracy_mean"].astype(float)
    df["accuracy_std"] = df["accuracy_std"].astype(float)
    return df


def plot_comparisons(df):
    for rep in df["representation"].unique():
        plt.figure()
        for clf in ["logistic_regression", "svm"]:
            sub = df[(df["representation"] == rep) & (df["classifier"] == clf)]
            sub = sub.sort_values("alpha")
            plt.errorbar(
                sub["alpha"],
                sub["accuracy_mean"],
                yerr=sub["accuracy_std"],
                label=clf,
                capsize=3,
            )
        plt.title(f"Accuracy vs. Alpha: {rep}")
        plt.xlabel("Alpha")
        plt.ylabel("Accuracy")
        plt.ylim(0.8, 1.05)
        plt.legend()
        outpath = os.path.join(OUTPUT_DIR, f"comparison_{rep}.png")
        plt.savefig(outpath)
        plt.close()
        print(f"[Saved plot for {rep}] -> {outpath}")


def print_clean_summary(df):
    print("\n====== Accuracy Comparison: Logistic Regression vs SVM ======")
    summary_rows = []

    for rep in df["representation"].unique():
        print(f"\nRepresentation: {rep}")
        print(f"{'Alpha':<6} | {'LogReg':>8} | {'SVM':>8} | {'Δ (SVM - LogReg)':>18}")
        print("-" * 45)

        for alpha in sorted(df["alpha"].unique()):
            row_log = df[
                (df["representation"] == rep)
                & (df["alpha"] == alpha)
                & (df["classifier"] == "logistic_regression")
            ]
            row_svm = df[
                (df["representation"] == rep)
                & (df["alpha"] == alpha)
                & (df["classifier"] == "svm")
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
                    "representation": rep,
                    "alpha": alpha,
                    "logreg_acc": acc_log,
                    "svm_acc": acc_svm,
                    "delta": delta,
                }
            )

    # Save summary if needed
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    print(f"\n[Saved CSV summary] -> {SUMMARY_CSV}")


def main():
    df_logreg = load_results(LOGREG_RESULTS, "logistic_regression")
    df_svm = load_results(SVM_RESULTS, "svm")
    df = pd.concat([df_logreg, df_svm], ignore_index=True)
    plot_comparisons(df)
    print_clean_summary(df)


if __name__ == "__main__":
    main()
