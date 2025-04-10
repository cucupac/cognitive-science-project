"""
This script generates combined multimodal CLIP embeddings for classification experiments.

For each pair of image and text embeddings (e.g., low-info image + high-info text),
it computes a combined vector using the formula:

    combined_vector = alpha * image_vector + (1 - alpha) * text_vector

Alpha (α) controls the weighting between image and text embeddings:
- alpha = 1.0 means only the image vector is used.
- alpha = 0.0 means only the text vector is used.
- alpha = 0.5 gives equal weight to both modalities.

Run this script before your classification experiments to precompute all necessary embeddings.
"""

import numpy as np
from pathlib import Path

PIXEL_DROPUT_LEVEL = "dropout_25"

# Define paths
VECTOR_STORE = "vector_store"
IMAGE_EMB = Path(VECTOR_STORE) / "image_embeddings"
TEXT_EMB = Path(VECTOR_STORE) / "text_embeddings"
COMBINED_EMB = Path(VECTOR_STORE) / f"combined_embeddings/{PIXEL_DROPUT_LEVEL}"

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
    # Handle the directory structure for image embeddings
    if image_level == "low_info":
        # Use the dropout directory for low_info images
        image_path = IMAGE_EMB / image_level / PIXEL_DROPUT_LEVEL
    else:
        # Use the regular directory for high_info images
        image_path = IMAGE_EMB / image_level

    text_path = TEXT_EMB / text_level
    combined_path = (
        COMBINED_EMB / f"{image_level}_img__{text_level}_text" / f"alpha_{alpha:.2f}"
    )
    combined_path.mkdir(parents=True, exist_ok=True)

    # Check if paths exist
    if not image_path.exists():
        print(f"Error: Image embedding path {image_path} does not exist")
        return

    if not text_path.exists():
        print(f"Error: Text embedding path {text_path} does not exist")
        return

    # Count files to process
    image_files = list(image_path.glob("*.npy"))
    if not image_files:
        print(f"Warning: No image embeddings found in {image_path}")
        return

    # Clear output format
    print(
        f"\nProcessing combination: {image_level} images + {text_level} text (α={alpha:.2f})"
    )
    print(f"  Source image embeddings: {image_path}")
    print(f"  Source text embeddings: {text_path}")
    print(f"  Saving combined embeddings to: {combined_path}")
    print(f"  Found {len(image_files)} image embeddings to process")

    # Go through embeddings by file name
    processed_count = 0
    missing_count = 0
    for image_file in image_files:
        text_file = text_path / image_file.name

        if not text_file.exists():
            missing_count += 1
            if missing_count <= 3:  # Limit the number of missing file warnings
                print(f"  Missing text: {text_file.name}")
            continue

        # Load embeddings
        img_vec = np.load(image_file)
        txt_vec = np.load(text_file)

        # Combine embeddings
        combined_vec = alpha * img_vec + (1 - alpha) * txt_vec

        # Save combined embedding
        np.save(combined_path / image_file.name, combined_vec)
        processed_count += 1

    if missing_count > 3:
        print(f"  ... and {missing_count - 3} more missing text embeddings")

    print(f"  ✓ Saved {processed_count} combined embeddings")


if __name__ == "__main__":
    print("=== Combined Embeddings Generator ===")
    print(f"Dropout level: {PIXEL_DROPUT_LEVEL}")

    for img_lvl, txt_lvl in pairs:
        for alpha in alphas:
            combine_and_store(img_lvl, txt_lvl, alpha)

    print("\n✅ Combined embeddings generation complete")
    print(f"All embeddings saved to: {COMBINED_EMB}")
