import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
RESULTS_CSV = "experiments/exp_0001/results/data/combined/multi_dropout_results.csv"
OUTPUT_DIR = "experiments/exp_0001/results/images/rescue"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Focus on two key representations
TARGET_REPS = ["LowImg-HighText", "HighImg-LowText"]


def main():
    df = pd.read_csv(RESULTS_CSV)
    for col in ["alpha", "accuracy_mean", "accuracy_std"]:
        df[col] = df[col].astype(float)

    # Only keep rows we care about
    df = df[df["representation"].isin(TARGET_REPS)]
    df = df[df["dropout_level"] == "dropout_90"]

    # Add rescue-gain columns
    rescue_df = compute_rescue_gains(df)

    # Plot using standardized y-axis across both representations
    plot_rescue_gains(rescue_df)


def compute_rescue_gains(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, compute the difference from the alpha=1.0 baseline (image only)
    and the alpha=0.0 baseline (text only).
    """
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

    # Compute rescue gains (accuracy improvement over a single modality)
    merged["delta_from_image_only"] = merged["accuracy_mean"] - merged["acc_image_only"]
    merged["delta_from_text_only"] = merged["accuracy_mean"] - merged["acc_text_only"]
    return merged


def plot_rescue_gains(df: pd.DataFrame):
    """
    Generate line plots for each representation showing rescue gain:
      - x-axis: α (0 = all text, 1 = all image)
      - y-axis: Accuracy improvement versus a pure-modality baseline (rescue gain)
    The y-axis is standardized across plots, and extra top space is added so that
    the legend (positioned in the upper-right) does not obscure the data.
    """
    # Filter out alpha=0.0 and alpha=1.0 since we're interested in the intermediate mixing
    subset = df[(df["alpha"] != 0.0) & (df["alpha"] != 1.0)]

    # Compute global y-limits across both representations
    all_deltas_img = subset["delta_from_image_only"].values
    all_deltas_text = subset["delta_from_text_only"].values
    global_values = np.concatenate([all_deltas_img, all_deltas_text])

    if len(global_values) == 0:
        print("No intermediate alpha data found. Nothing to plot.")
        return

    global_min = global_values.min()
    global_max = global_values.max()
    margin = 0.02 * (global_max - global_min) if (global_max - global_min) > 0 else 0.01
    global_lower, global_upper = global_min - margin, global_max + margin

    # Increase the top margin so the legend doesn't cover the data (adjust factor as needed)
    top_factor = 0.25
    global_upper += top_factor * (global_max - global_min)

    # Build figures per representation using the same y-axis scale
    for rep in sorted(subset["representation"].unique()):
        rep_df = subset[subset["representation"] == rep].copy()
        rep_df.sort_values(by="alpha", inplace=True)

        if rep_df.empty:
            print(f"No data found for {rep} at dropout=90.")
            continue

        alphas = rep_df["alpha"].values
        delta_img = rep_df["delta_from_image_only"].values
        delta_text = rep_df["delta_from_text_only"].values

        fig, ax = plt.subplots(figsize=(6.2, 4.2))

        # Plot rescue gain for each baseline
        ax.plot(alphas, delta_img, marker="o", label="Gain over Image-Only Baseline")
        ax.plot(alphas, delta_text, marker="s", label="Gain over Text-Only Baseline")
        ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=1)

        # Set intuitive title based on representation
        if rep == "LowImg-HighText":
            title = (
                "Rescue Gains: Adding Detailed Text to\n"
                "Enhance Severely Degraded Images (90% Pixel Dropout)"
            )
        elif rep == "HighImg-LowText":
            title = (
                "Rescue Gains: Adding High-Quality Images to\n"
                "Augment Sparse Text Descriptions (90% Pixel Dropout)"
            )
        else:
            title = f"Rescue Gains for {rep} (Dropout 90%)"
        ax.set_title(title, fontsize=11)

        # Update axis labels for clarity
        ax.set_xlabel("Alpha (0 = 100% Text, 1 = 100% Image)")
        ax.set_ylabel("Rescue Gain (Δ Accuracy vs. Pure Modality)")

        # Apply the standardized y-axis limits
        ax.set_ylim(global_lower, global_upper)

        # Place the legend at the upper-right without overlapping data
        ax.legend(loc="upper right")

        plt.tight_layout()

        outname = f"rescue_gains_{rep}_dropout90.png"
        outpath = os.path.join(OUTPUT_DIR, outname)
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"[Saved] {outpath}")


if __name__ == "__main__":
    main()
