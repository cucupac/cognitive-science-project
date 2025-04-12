import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
RESULTS_CSV = "experiments/exp_0003/results/data/combined/multi_dropout_results.csv"
OUTPUT_DIR = "experiments/exp_0003/results/images/rescue"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Focus on two key representations
TARGET_REPS = ["LowImg-HighText", "HighImg-LowText"]


def main():
    df = pd.read_csv(RESULTS_CSV)
    for col in ["alpha", "accuracy_mean", "accuracy_std"]:
        df[col] = df[col].astype(float)

    df = df[df["representation"].isin(TARGET_REPS)]
    df = df[df["dropout_level"] == "dropout_90"]

    rescue_df = compute_rescue_gains(df)
    plot_rescue_gains(rescue_df)


def compute_rescue_gains(df: pd.DataFrame) -> pd.DataFrame:
    image_baseline = df[df["alpha"] == 1.0].copy()
    image_baseline.rename(columns={"accuracy_mean": "acc_image_only"}, inplace=True)
    text_baseline = df[df["alpha"] == 0.0].copy()
    text_baseline.rename(columns={"accuracy_mean": "acc_text_only"}, inplace=True)

    merged = pd.merge(
        df,
        image_baseline[["dropout_level", "representation", "acc_image_only"]],
        on=["dropout_level", "representation"],
        how="left",
    )
    merged = pd.merge(
        merged,
        text_baseline[["dropout_level", "representation", "acc_text_only"]],
        on=["dropout_level", "representation"],
        how="left",
    )

    merged["delta_from_image_only"] = merged["accuracy_mean"] - merged["acc_image_only"]
    merged["delta_from_text_only"] = merged["accuracy_mean"] - merged["acc_text_only"]
    return merged


def plot_rescue_gains(df: pd.DataFrame):
    subset = df[(df["alpha"] != 0.0) & (df["alpha"] != 1.0)]

    for rep in sorted(subset["representation"].unique()):
        rep_df = subset[subset["representation"] == rep].copy()
        rep_df.sort_values(by="alpha", inplace=True)

        if rep_df.empty:
            print(f"No data found for {rep} at dropout=90.")
            continue

        alphas = rep_df["alpha"].values
        delta_img = rep_df["delta_from_image_only"].values
        delta_text = rep_df["delta_from_text_only"].values

        all_values = np.concatenate([delta_img, delta_text])
        y_min, y_max = all_values.min(), all_values.max()
        margin = 0.02 * (y_max - y_min) if (y_max - y_min) > 0 else 0.01
        auto_lower, auto_upper = y_min - margin, y_max + margin

        fig, ax = plt.subplots(figsize=(6.2, 4.2))

        ax.plot(
            alphas, delta_img, marker="o", label="Improvement over Image-only (α = 1.0)"
        )
        ax.plot(
            alphas, delta_text, marker="s", label="Improvement over Text-only (α = 0.0)"
        )
        ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=1)

        if rep == "LowImg-HighText":
            title = "Effect of Adding Detailed Text to Severely Degraded Images\n(90% Pixel Dropout)"
        elif rep == "HighImg-LowText":
            title = "Effect of Adding High-Quality Image to Sparse Text Descriptions"
        else:
            title = f"{rep} | Dropout=90"

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("α (0 = All Text, 1 = All Image)")
        ax.set_ylabel("Accuracy Improvement (Δ vs. Single-Modality)")
        ax.set_ylim(auto_lower, auto_upper)
        ax.legend(loc="best")
        plt.tight_layout()

        outname = f"rescue_gains_{rep}_dropout90.png"
        outpath = os.path.join(OUTPUT_DIR, outname)
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"[Saved] {outpath}")


if __name__ == "__main__":
    main()
