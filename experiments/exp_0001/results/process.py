import os
import pandas as pd
import matplotlib.pyplot as plt

# CSV file containing columns:
#   [dropout_level, representation, alpha, accuracy_mean, accuracy_std]
RESULTS_CSV = "experiments/exp_0001/results/data/combined/multi_dropout_results.csv"
OUTPUT_DIR = "experiments/exp_0001/results/images/combined"
BEST_ALPHA_CSV = "experiments/exp_0001/results/data/combined/best_alpha_summary.csv"

# Create a mapping from the raw 'representation' string to a more readable label
REP_LABELS = {
    "LowImg-HighText": "Degraded Image + Detailed Text",
    "HighImg-LowText": "High-Quality Image + Sparse Text",
    "LowImg-LowText": "Degraded Image + Sparse Text",
    "HighImg-HighText": "High-Quality Image + Detailed Text",
    # Add more if needed
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(RESULTS_CSV):
        print(f"Error: {RESULTS_CSV} does not exist.")
        return

    # 1) Read CSV
    df = pd.read_csv(RESULTS_CSV)

    # Ensure correct dtypes
    df["alpha"] = df["alpha"].astype(float)
    df["accuracy_mean"] = df["accuracy_mean"].astype(float)
    df["accuracy_std"] = df["accuracy_std"].astype(float)

    # -------------------------------------------------------------------------
    # Compute a GLOBAL y-limit range (min, max) across the entire dataframe,
    # so we can standardize the y-axis for all plots:
    # -------------------------------------------------------------------------
    global_min = df["accuracy_mean"].min()
    global_max = df["accuracy_mean"].max()

    # Add a small margin:
    margin = 0.02 * (global_max - global_min) if (global_max - global_min) > 0 else 0.01
    global_ylim = (global_min - margin, global_max + margin)

    # 2) Generate Type A plots:
    #    For each dropout level, lines for each representation vs. alpha
    plot_accuracy_vs_alpha_by_dropout(df, global_ylim)

    # 3) Generate Type B plots:
    #    For each representation, lines for each dropout level vs. alpha
    plot_accuracy_vs_alpha_by_representation(df, global_ylim)

    # 4) Compute best alpha per (dropout_level, representation)
    best_df = find_best_alpha(df)
    best_df.to_csv(BEST_ALPHA_CSV, index=False)
    print(f"\n[Saved best-alpha summary] -> {BEST_ALPHA_CSV}")

    # Print console summary
    print("\n=== Best Alpha Summary (Dropout, Representation) ===")
    for _, row in best_df.iterrows():
        print(
            f"- {row['dropout_level']}, {row['representation']}: "
            f"alpha={row['best_alpha']}, accuracy={row['best_accuracy_mean']} ± {row['best_accuracy_std']}"
        )


def plot_accuracy_vs_alpha_by_dropout(df: pd.DataFrame, ylim: tuple):
    """
    One figure per dropout level.
    X-axis: alpha
    Multiple lines: each representation
    """
    dropout_levels = sorted(df["dropout_level"].unique())
    for dl in dropout_levels:
        sub_df = df[df["dropout_level"] == dl].copy()

        # Convert "dropout_90" -> "90"
        numeric_dropout = dl.replace("dropout_", "")
        # Prepare figure
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        # Plot lines for each representation
        for rep in sorted(sub_df["representation"].unique()):
            rep_df = sub_df[sub_df["representation"] == rep].copy()
            rep_df.sort_values(by="alpha", inplace=True)
            alphas = rep_df["alpha"].values
            means = rep_df["accuracy_mean"].values
            stds = rep_df["accuracy_std"].values

            label_str = REP_LABELS.get(rep, rep)  # fallback to raw if missing
            ax.errorbar(
                alphas, means, yerr=stds, label=label_str, capsize=3, marker="o"
            )

        ax.set_xlabel(r"Alpha (0 = All Text, 1 = All Image)")
        ax.set_ylabel("Mean Accuracy")
        ax.set_ylim(ylim[0], ylim[1])

        # --- Build the math-mode title for dropout ---
        # We want something like: (25\% Pixel Dropout) in bold with spaces preserved.
        # Build the text in pieces so that the % is escaped.
        # Note: In the f-string, double backslashes produce a single backslash in the result.
        title_text = f"{numeric_dropout}\\% Pixel Dropout"
        # Replace literal spaces with an explicit escape for mathtext.
        title_text_escaped = title_text.replace(" ", "\\ ")
        # Create a math string using an f-string.
        math_string = f"$\\mathbf{{({title_text_escaped})}}$"
        ax.set_title("Classification Accuracy vs. α\n" + math_string, fontsize=11)

        ax.legend(loc="best")
        plt.tight_layout()
        outpath = os.path.join(OUTPUT_DIR, f"accuracy_vs_alpha_{dl}.png")
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"[Saved figure for dropout={dl}] -> {outpath}")


def plot_accuracy_vs_alpha_by_representation(df: pd.DataFrame, ylim: tuple):
    """
    One figure per representation.
    X-axis: alpha
    Multiple lines: each dropout level
    """
    representations = sorted(df["representation"].unique())
    for rep in representations:
        sub_df = df[df["representation"] == rep].copy()

        # Convert representation to a nicer label
        friendly_rep = REP_LABELS.get(rep, rep)

        # Prepare figure
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        # Plot each dropout level line
        for dl in sorted(sub_df["dropout_level"].unique()):
            dl_df = sub_df[sub_df["dropout_level"] == dl].copy()
            dl_df.sort_values(by="alpha", inplace=True)
            alphas = dl_df["alpha"].values
            means = dl_df["accuracy_mean"].values
            stds = dl_df["accuracy_std"].values

            # Convert "dropout_75" -> "75% Pixel Dropout" for the label.
            numeric_dropout = dl.replace("dropout_", "")
            label_str = f"{numeric_dropout}% Pixel Dropout"
            ax.errorbar(
                alphas, means, yerr=stds, label=label_str, capsize=3, marker="o"
            )

        ax.set_xlabel(r"Alpha (0 = All Text, 1 = All Image)")
        ax.set_ylabel("Mean Accuracy")
        ax.set_ylim(ylim[0], ylim[1])

        # --- Build the math-mode title for representation ---
        rep_title_text = f"Representation: {friendly_rep}"
        rep_title_text_escaped = rep_title_text.replace(" ", "\\ ")
        rep_math_string = f"$\\mathbf{{{rep_title_text_escaped}}}$"
        ax.set_title(
            "Classification Accuracy vs. α\n(" + rep_math_string + ")", fontsize=11
        )

        ax.legend(loc="best")
        plt.tight_layout()
        outpath = os.path.join(OUTPUT_DIR, f"accuracy_vs_alpha_{rep}.png")
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"[Saved figure for rep={rep}] -> {outpath}")


def find_best_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (dropout_level, representation),
    find the row with the highest accuracy_mean.
    """
    best_rows = []
    dropout_list = sorted(df["dropout_level"].unique())
    for dl in dropout_list:
        dl_df = df[df["dropout_level"] == dl]
        reps = sorted(dl_df["representation"].unique())
        for rep in reps:
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
    return pd.DataFrame(best_rows)


if __name__ == "__main__":
    main()
