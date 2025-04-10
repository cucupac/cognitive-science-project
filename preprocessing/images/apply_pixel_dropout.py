import numpy as np
from PIL import Image
import os

# Paths
input_dir = "sample_sets/photos/high_info"
output_dir = "sample_sets/photos/low_info/dropout_25"

# Dropout probability
dropout_probability = 0.25

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Apply dropout to all images
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = Image.open(input_path)
        img_array = np.array(img)

        dropout_mask = np.random.rand(*img_array.shape[:2]) > dropout_probability

        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            dropout_mask = np.stack([dropout_mask] * 3, axis=-1)

        img_dropout = img_array * dropout_mask
        Image.fromarray(img_dropout.astype(np.uint8)).save(output_path)
