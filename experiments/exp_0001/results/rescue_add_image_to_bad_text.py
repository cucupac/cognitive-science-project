import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths (adjust as needed)
RESULTS_CSV = "experiments/exp_0001/results/data/combined/multi_dropout_results.csv"
OUTPUT_DIR = "experiments/exp_0001/results/images/rescue"
OUTPUT_CSV = "experiments/exp_0001/results/data/rescue/image_quality_rescue_effect.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_CSV)
    df["alpha"] = df["alpha"].astype(float)
    df["accuracy_mean"] = df["accuracy_mean"].astype(float)

    # Convert "dropout_25" -> integer 25, etc.
    df["dropout_pct"] = df["dropout_level"].apply(lambda x: int(x.split("_")[1]))

    # ----------------------------------------------------------------------------
    # 1) Create a new "image_quality" label:
    #    - If representation contains "HighImg", treat as "Pristine Image"
    #    - Otherwise, treat as "<dropout_pct>% Degraded Image"
    #
    #    This ensures we can plot "HighImg-LowText" even if the CSV says dropout_25, etc.
    # ----------------------------------------------------------------------------
    def determine_image_quality(row):
        if "HighImg" in row["representation"]:
            return "Pristine Image"
        else:
            return f"{row['dropout_pct']}% Degraded Image"

    df["image_quality"] = df.apply(determine_image_quality, axis=1)

    # ----------------------------------------------------------------------------
    # 2) Identify the text-only baseline:
    #    We'll keep using LowImg-LowText w/ alpha=0 for *the first row* we find.
    #    (You can refine this if you want separate baselines per dropout, etc.)
    # ----------------------------------------------------------------------------
    text_only_rows = df[
        (df["representation"] == "LowImg-LowText") & (df["alpha"] == 0.0)
    ]
    if text_only_rows.empty:
        print(
            "Warning: No text-only baseline found (LowImg-LowText, alpha=0). Aborting."
        )
        return

    # Just pick the first match as the baseline; or take mean if you prefer
    text_only_acc = text_only_rows["accuracy_mean"].iloc[0]
    print(f"Using text-only baseline accuracy = {text_only_acc:.3f}")

    # ----------------------------------------------------------------------------
    # 3) For each combination of image_quality and alpha>0, compute rescue effect
    #    relative to that single text-only baseline.
    # ----------------------------------------------------------------------------
    rows = []
    # We'll gather all unique image_qualities in order to plot them
    # We also force a particular order: Pristine first, then ascending dropout.
    # (If you prefer a different order, adjust the sort key.)
    all_img_types = sorted(
        df["image_quality"].unique(),
        key=lambda x: 0 if x == "Pristine Image" else int(x.split("%")[0]),
    )

    # For each image_quality, gather its alpha>0 rows
    for img_type in all_img_types:
        sub = df[df["image_quality"] == img_type].copy()
        # Sort by alpha so the final plot lines connect properly in alpha order
        sub = sub.sort_values("alpha")

        for alpha_val in sorted(sub["alpha"].unique()):
            if alpha_val == 0.0:
                continue  # text-only case is our baseline, no "rescue" to compute
            row_df = sub[sub["alpha"] == alpha_val]
            if row_df.empty:
                continue

            # Some representations might have multiple rows for same alpha
            # (e.g. different classifier seeds). We'll just take the mean here.
            accuracy = row_df["accuracy_mean"].mean()
            rescue_effect = accuracy - text_only_acc

            rows.append(
                {
                    "image_quality": img_type,
                    "alpha": alpha_val,
                    "accuracy": accuracy,
                    "rescue_effect": rescue_effect,
                }
            )

    # Convert to DataFrame and save
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[Saved CSV] -> {OUTPUT_CSV}")

    # ----------------------------------------------------------------------------
    # 4) Plot lines for each image_quality vs. alpha (converted to %)
    # ----------------------------------------------------------------------------
    plt.figure(figsize=(9, 6))

    # Custom colors for clarity. Adjust or expand as you like:
    colors = {
        "Pristine Image": "black",
        "25% Degraded Image": "#1f77b4",
        "50% Degraded Image": "#ff7f0e",
        "75% Degraded Image": "#2ca02c",
        "90% Degraded Image": "#d62728",
    }

    for img_type in all_img_types:
        subset = results_df[results_df["image_quality"] == img_type]
        if subset.empty:
            continue

        # Plot alpha vs. rescue effect
        plt.plot(
            subset["alpha"] * 100,  # convert alpha to percentage
            subset["rescue_effect"],
            marker="o",
            linewidth=2,
            markersize=7,
            label=img_type,
            color=colors.get(img_type, "gray"),
        )

    plt.axhline(y=0, linestyle="--", color="gray", linewidth=1.5)
    plt.grid(True, alpha=0.7)

    plt.xlabel("Alpha - Image Contribution (%)", fontsize=13)
    plt.ylabel("Accuracy Improvement vs. Text-Only Baseline", fontsize=13)
    plt.title(
        "How Adding Images to Sparse Text Affects Classification Accuracy", fontsize=14
    )
    plt.legend(title="Image Quality", fontsize=11, title_fontsize=12, loc="best")
    plt.xticks([0, 25, 50, 75, 100])
    plt.tight_layout()

    outpath = os.path.join(OUTPUT_DIR, "image_affects_degraded_text.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Saved Plot] -> {outpath}")


if __name__ == "__main__":
    main()
