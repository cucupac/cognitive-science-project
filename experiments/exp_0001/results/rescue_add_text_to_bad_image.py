import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to your CSV (adjust as needed)
RESULTS_CSV = "experiments/exp_0001/results/data/combined/multi_dropout_results.csv"
OUTPUT_DIR = "experiments/exp_0001/results/images/rescue"
OUTPUT_CSV = "experiments/exp_0001/results/data/rescue/best_rescue_low_vs_high_text.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_CSV)
    for col in ["alpha", "accuracy_mean", "accuracy_std"]:
        df[col] = df[col].astype(float)

    # Extract all representations, we need to include LowImg-LowText now
    all_reps = df["representation"].unique()

    # Extract dropout percentage for better x-axis
    df["dropout_pct"] = df["dropout_level"].apply(lambda x: int(x.split("_")[1]))

    # We need to analyze rescue effect for both text types with degraded images
    # and pristine images (HighImg-HighText and HighImg-LowText)
    rescue_reps = ["LowImg-HighText", "LowImg-LowText"]

    # Calculate rescue effect for each representation
    best_rows = []

    # Process normal (degraded image) representations
    for rep in rescue_reps:
        for dropout_pct in sorted(df["dropout_pct"].unique()):
            # Get data for this representation and dropout level
            subset = df[
                (df["representation"] == rep) & (df["dropout_pct"] == dropout_pct)
            ]

            if len(subset) == 0:
                continue

            # Get image-only baseline (alpha = 1.0)
            image_only_acc = subset[subset["alpha"] == 1.0]["accuracy_mean"].values[0]

            # Find the best accuracy across mixed alpha values (excluding 0 and 1)
            mixed_subset = subset[(subset["alpha"] > 0) & (subset["alpha"] < 1)]
            if len(mixed_subset) == 0:
                continue

            best_idx = mixed_subset["accuracy_mean"].idxmax()
            best_row = mixed_subset.loc[best_idx]
            best_alpha = best_row["alpha"]
            best_acc = best_row["accuracy_mean"]

            # Calculate rescue effect
            rescue_effect = best_acc - image_only_acc

            best_rows.append(
                {
                    "dropout_level": f"dropout_{dropout_pct}",
                    "dropout_pct": dropout_pct,
                    "representation": rep,
                    "best_alpha": best_alpha,
                    "rescue": rescue_effect,
                    "image_only_acc": image_only_acc,
                    "best_acc": best_acc,
                }
            )

    # Now add data for pristine images (0% dropout)
    # Map from high-quality representations to corresponding low-quality ones
    pristine_map = {
        "HighImg-HighText": "LowImg-HighText",
        "HighImg-LowText": "LowImg-LowText",
    }

    for pristine_rep, equivalent_rep in pristine_map.items():
        # Get data for this pristine representation
        pristine_subset = df[df["representation"] == pristine_rep]

        if len(pristine_subset) == 0:
            continue

        # Get image-only baseline (alpha = 1.0)
        image_only_acc = pristine_subset[pristine_subset["alpha"] == 1.0][
            "accuracy_mean"
        ].values[0]

        # Find the best accuracy across mixed alpha values (excluding 0 and 1)
        mixed_subset = pristine_subset[
            (pristine_subset["alpha"] > 0) & (pristine_subset["alpha"] < 1)
        ]
        if len(mixed_subset) == 0:
            continue

        best_idx = mixed_subset["accuracy_mean"].idxmax()
        best_row = mixed_subset.loc[best_idx]
        best_alpha = best_row["alpha"]
        best_acc = best_row["accuracy_mean"]

        # Calculate rescue effect
        rescue_effect = best_acc - image_only_acc

        best_rows.append(
            {
                "dropout_level": "dropout_0",
                "dropout_pct": 0,  # 0% dropout for pristine images
                "representation": equivalent_rep,
                "best_alpha": best_alpha,
                "rescue": rescue_effect,
                "image_only_acc": image_only_acc,
                "best_acc": best_acc,
            }
        )

    # Convert to DataFrame
    summary_df = pd.DataFrame(best_rows)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[Saved summary CSV] -> {OUTPUT_CSV}")

    # Plot the rescue effect
    plt.figure(figsize=(9, 6))

    # Create more readable labels for the plot
    rep_mapping = {
        "LowImg-HighText": "Detailed Text Added to Images",
        "LowImg-LowText": "Minimal Text Added to Images",
    }

    # Plot each representation
    for rep in rescue_reps:
        sub_df = summary_df[summary_df["representation"] == rep].copy()
        sub_df.sort_values("dropout_pct", inplace=True)

        plt.plot(
            sub_df["dropout_pct"],
            sub_df["rescue"],
            marker="o",
            markersize=8,
            linewidth=2,
            label=rep_mapping[rep],
        )

    # Formatting
    plt.xlabel("Image Degradation Level (% Pixel Dropout)", fontsize=12)
    plt.ylabel(
        "Improvement from Adding Text\n(Accuracy Gain vs. Image-Only)", fontsize=12
    )
    plt.title("How Adding Text Improves Classification of Degraded Images", fontsize=14)
    plt.grid(True, alpha=0.7)
    plt.legend(fontsize=11)

    # Ensure x-axis starts at 0 for pristine images
    plt.xlim(left=0)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "text_affects_degraded_images.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved plot] -> {outpath}")


if __name__ == "__main__":
    main()
