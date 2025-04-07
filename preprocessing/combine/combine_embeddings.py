"""
This script generates combined multimodal CLIP embeddings for classification experiments.

For each pair of image and text embeddings (e.g., low-info image + high-info text),
it computes a combined vector using the formula:

    combined_vector = alpha * image_vector + (1 - alpha) * text_vector

Alpha (α) controls the weighting between image and text embeddings:
- alpha = 1.0 means only the image vector is used.
- alpha = 0.0 means only the text vector is used.
- alpha = 0.5 gives equal weight to both modalities.

The script saves the resulting combined vectors into a structured directory under:
    vector_store/combined_embeddings/{image_level}_img__{text_level}_text/alpha_{α}/

Run this script before your classification experiments to precompute all necessary embeddings.
"""

import numpy as np
from pathlib import Path

# Define paths
VECTOR_STORE = "vector_store"
IMAGE_EMB = Path(VECTOR_STORE) / "image_embeddings"
TEXT_EMB = Path(VECTOR_STORE) / "text_embeddings"
COMBINED_EMB = Path(VECTOR_STORE) / "combined_embeddings"

# Make sure output directory exists
COMBINED_EMB.mkdir(parents=True, exist_ok=True)

# Define combinations and alphas
pairs = [
    ("low_info", "high_info"),
    ("high_info", "low_info"),
    ("low_info", "low_info"),
    ("high_info", "high_info"),
]

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]


def combine_and_store(image_level, text_level, alpha):
    image_path = IMAGE_EMB / image_level
    text_path = TEXT_EMB / text_level
    combined_path = (
        COMBINED_EMB / f"{image_level}_img__{text_level}_text" / f"alpha_{alpha:.2f}"
    )
    combined_path.mkdir(parents=True, exist_ok=True)

    # Go through embeddings by file name
    for image_file in image_path.glob("*.npy"):
        text_file = text_path / image_file.name

        if not text_file.exists():
            print(f"Warning: No matching text embedding for {image_file.name}")
            continue

        # Load embeddings
        img_vec = np.load(image_file)
        txt_vec = np.load(text_file)

        # Combine embeddings
        combined_vec = alpha * img_vec + (1 - alpha) * txt_vec

        # Save combined embedding
        np.save(combined_path / image_file.name, combined_vec)


if __name__ == "__main__":
    for img_lvl, txt_lvl in pairs:
        for alpha in alphas:
            print(f"Combining {img_lvl} image + {txt_lvl} text at alpha={alpha:.2f}")
            combine_and_store(img_lvl, txt_lvl, alpha)

    print("✅ Combined embeddings successfully generated and stored.")
