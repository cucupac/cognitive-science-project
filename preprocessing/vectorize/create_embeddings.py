import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load CLIP model once
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Helper functions
def get_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs[0].numpy()


def get_text_embedding(text_path):
    with open(text_path, "r") as file:
        text = file.read().strip()
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return outputs[0].numpy()


# Define directories to process
data_dirs = {
    "image_embeddings/high_info": "sample_sets/photos/high_info",
    "image_embeddings/low_info": "sample_sets/photos/low_info",
    "text_embeddings/high_info": "sample_sets/descriptions/high_info",
    "text_embeddings/low_info": "sample_sets/descriptions/low_info",
}

# Process each directory
for embed_dir, data_dir in data_dirs.items():
    # Ensure embedding directories exist
    embed_full_dir = os.path.join("vector_store", embed_dir)
    os.makedirs(embed_full_dir, exist_ok=True)

    data_type = "image" if "photos" in data_dir else "text"

    # Process each file
    for filename in os.listdir(data_dir):
        base_name, ext = os.path.splitext(filename)
        if data_type == "image" and ext.lower() in [".jpg", ".jpeg", ".png"]:
            data_path = os.path.join(data_dir, filename)
            embedding = get_image_embedding(data_path)

        elif data_type == "text" and ext.lower() == ".txt":
            data_path = os.path.join(data_dir, filename)
            embedding = get_text_embedding(data_path)

        else:
            continue  # Skip any non-matching files

        # Save embedding
        embedding_path = os.path.join(embed_full_dir, f"{base_name}.npy")
        np.save(embedding_path, embedding)
        print(f"Saved embedding to {embedding_path}")
