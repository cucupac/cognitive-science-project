import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_classifier(
    embedding_folder: str, cv: int = 5, random_state: int = 42, debug: bool = False
) -> Tuple[float, float]:
    """
    Evaluate a logistic regression classifier on embeddings in 'embedding_folder'.
    Returns the mean and std of cross-validation accuracy.
    Optionally prints debug info (label distribution, confusion matrix, etc.).
    """

    def load_vectors_labels(path):
        vectors, labels = [], []
        for file in Path(path).glob("*.npy"):
            vec = np.load(file)
            # 0 = cat, 1 = dog
            label = 0 if "cat" in file.name else 1
            vectors.append(vec)
            labels.append(label)
        return np.array(vectors), np.array(labels)

    # Load embeddings and labels
    embeddings, labels = load_vectors_labels(embedding_folder)

    if debug:
        print(f"\n[DEBUG] Loading embeddings from: {embedding_folder}")
        print(f"[DEBUG] embeddings.shape: {embeddings.shape}")
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"[DEBUG] Label distribution: {dict(zip(unique_labels, counts))}")

    # Set up logistic regression & cross-validation
    clf = LogisticRegression(max_iter=1000)
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Get accuracy across folds
    scores = cross_val_score(clf, embeddings, labels, cv=kf, scoring="accuracy")
    accuracy_mean, accuracy_std = scores.mean(), scores.std()

    if debug:
        # Predict across folds for confusion matrix
        preds = cross_val_predict(clf, embeddings, labels, cv=kf)
        cm = confusion_matrix(labels, preds)
        print("[DEBUG] Confusion Matrix:\n", cm)
        print(
            "[DEBUG] Classification Report:\n",
            classification_report(labels, preds, target_names=["cat", "dog"]),
        )

    return accuracy_mean, accuracy_std
