"""
Analyze multiple dropout levels in a single run.

For each dropout level (25, 50, 75, 90):
  1) Loads the combined CLIP embeddings from: vector_store/combined_embeddings/dropout_{X}
  2) Classifies them via logistic regression (5-fold CV)
  3) Appends results to: experiments/exp_0003/results/data/combined/multi_dropout_results.csv
     with columns: [dropout_level, representation, alpha, accuracy_mean, accuracy_std]

No charts are generated here; just a single CSV for all dropout levels.
"""

import os
import sys
from pathlib import Path
import csv
import numpy as np

# Allow file importing from parent directory
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from classifiers.logistic_regression import evaluate_classifier

# Create output directory for CSV
os.makedirs("experiments/exp_0003/results/data/combined", exist_ok=True)

CSV_RESULTS_PATH = (
    "experiments/exp_0003/results/data/combined/multi_dropout_results.csv"
)

# Specify all dropout levels you want to test
DROPOUT_LEVELS = [25, 50, 75, 90]

PAIRS = [
    ("low_info_img__high_info_text", "LowImg-HighText"),
    ("high_info_img__low_info_text", "HighImg-LowText"),
    ("low_info_img__low_info_text", "LowImg-LowText"),
    ("high_info_img__high_info_text", "HighImg-HighText"),
]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_vectors_labels(embedding_folder: str):
    vectors, labels = [], []
    for file in Path(embedding_folder).glob("*.npy"):
        vec = np.load(file)
        label = 0 if "cat" in file.name else 1
        vectors.append(vec)
        labels.append(label)
    return np.array(vectors), np.array(labels)


def main(debug=False):
    # Overwrite CSV with header
    with open(CSV_RESULTS_PATH, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "dropout_level",
                "representation",
                "alpha",
                "accuracy_mean",
                "accuracy_std",
            ]
        )

    # Loop over each dropout level
    for level in DROPOUT_LEVELS:
        base_combined_path = (
            Path("vector_store/combined_embeddings") / f"dropout_{level}"
        )

        if not base_combined_path.is_dir():
            print(
                f"Warning: {base_combined_path} not found. Skipping this dropout level."
            )
            continue

        print(f"\n=== Analyzing dropout_{level} ===")

        # For each representation & alpha, classify and append results
        for folder_name, display_name in PAIRS:
            for alpha in ALPHAS:
                alpha_folder = base_combined_path / folder_name / f"alpha_{alpha:.2f}"
                if not alpha_folder.is_dir():
                    print(f"  [Skip] {alpha_folder} not found.")
                    continue

                embeddings, labels = load_vectors_labels(str(alpha_folder))
                mean, std = evaluate_classifier(str(alpha_folder), debug=debug)

                print(
                    f"  [dropout_{level}, {display_name}, alpha={alpha:.2f}] "
                    f"Accuracy: {mean:.3f} ± {std:.3f}"
                )

                # Append a row to the CSV
                with open(CSV_RESULTS_PATH, mode="a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [
                            f"dropout_{level}",
                            display_name,
                            f"{alpha:.2f}",
                            f"{mean:.3f}",
                            f"{std:.3f}",
                        ]
                    )


if __name__ == "__main__":
    main(debug=True)
