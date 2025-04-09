"""
Creates a fresh "high-info" sample set consisting of 500 cat and 500 dog images,
randomly sampled from an unfiltered source directory.

This script:
- Randomly selects 500 cat and 500 dog images from 'unfiltered_photos'
- Copies these images into 'sample_sets/photos/high_info'
- Overwrites existing contents in the target directory

Ensure 'unfiltered_photos' contains enough images (at least 500 cats and 500 dogs).
"""

import os
import shutil
import random

source_dir = "unfiltered_photos"
target_dir = "sample_sets/photos/high_info"

samples_per_class = 500

# Create (or clear) target directory
os.makedirs(target_dir, exist_ok=True)

# Get lists of cat/dog photos
cats = [f for f in os.listdir(source_dir) if f.startswith("cat.")]
dogs = [f for f in os.listdir(source_dir) if f.startswith("dog.")]

# Select random samples
selected_cats = random.sample(cats, samples_per_class)
selected_dogs = random.sample(dogs, samples_per_class)

# Copy selected images to target dir
for filename in selected_cats + selected_dogs:
    shutil.copy2(os.path.join(source_dir, filename), os.path.join(target_dir, filename))

print(
    f"Successfully created new set with {samples_per_class} cats and {samples_per_class} dogs."
)
