import os
import sys
from pathlib import Path
import csv

# Allow file importing from parent directory
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from classifiers.logistic_regression import evaluate_classifier

CSV_RESULTS_PATH = "experiements/exp_0001/results/data/baseline/results.csv"


def load_vectors_labels(embedding_folder: str):
    vectors, labels = [], []
    for file in Path(embedding_folder).glob("*.npy"):
        vec = np.load(file)
        label = 0 if "cat" in file.name else 1
        vectors.append(vec)
        labels.append(label)
    return np.array(vectors), np.array(labels)


def plot_tsne(embeddings, labels, outpath: str):
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    plt.figure()
    plt.scatter(X_tsne[labels == 0, 0], X_tsne[labels == 0, 1], label="cat")
    plt.scatter(X_tsne[labels == 1, 0], X_tsne[labels == 1, 1], label="dog")
    plt.legend()
    plt.title("t-SNE of Embeddings")
    plt.savefig(outpath)
    plt.close()


def save_results_csv(
    writer, representation_name: str, accuracy_mean: float, accuracy_std: float
):
    writer.writerow(
        [representation_name, f"{accuracy_mean:.3f}", f"{accuracy_std:.3f}"]
    )


def run_and_record(writer, name, path, tsne_out, debug=False):
    embeddings, labels = load_vectors_labels(path)
    plot_tsne(embeddings, labels, tsne_out)
    mean, std = evaluate_classifier(path, debug=debug)
    print(f"[{name}] Accuracy: {mean:.3f} ± {std:.3f}")
    save_results_csv(writer, name, mean, std)


if __name__ == "__main__":
    os.makedirs("experiements/exp_0001/results/data/baseline", exist_ok=True)
    os.makedirs("experiements/exp_0001/results/images/baseline", exist_ok=True)

    with open(CSV_RESULTS_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["representation", "accuracy_mean", "accuracy_std"])

        run_and_record(
            writer,
            "image_high_info",
            "vector_store/image_embeddings/high_info",
            "experiements/exp_0001/results/images/baseline/scatter_plots/scatter_plots/image_high_info_tsne.png",
            debug=True,
        )
        run_and_record(
            writer,
            "image_low_info",
            "vector_store/image_embeddings/low_info",
            "experiements/exp_0001/results/images/baseline/scatter_plots/image_low_info_tsne.png",
            debug=True,
        )
        run_and_record(
            writer,
            "text_high_info",
            "vector_store/text_embeddings/high_info",
            "experiements/exp_0001/results/images/baseline/scatter_plots/text_high_info_tsne.png",
            debug=True,
        )
        run_and_record(
            writer,
            "text_low_info",
            "vector_store/text_embeddings/low_info",
            "experiements/exp_0001/results/images/baseline/scatter_plots/text_low_info_tsne.png",
            debug=True,
        )
