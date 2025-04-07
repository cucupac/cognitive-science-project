import os
import shutil
import random

source_dir = "unfiltered_photos"
target_dir = "sample_sets/photos/high_info"

# Count of new samples per class
new_samples_per_class = 200

# Get list of all available cat/dog photos
cats = [f for f in os.listdir(source_dir) if f.startswith("cat.")]
dogs = [f for f in os.listdir(source_dir) if f.startswith("dog.")]

# Already selected photos (existing set)
existing_photos = set(os.listdir(target_dir))

# Filter out existing selections
available_cats = [f for f in cats if f not in existing_photos]
available_dogs = [f for f in dogs if f not in existing_photos]

# Randomly select NEW photos (no duplicates)
selected_cats = random.sample(available_cats, new_samples_per_class)
selected_dogs = random.sample(available_dogs, new_samples_per_class)

# Copy newly selected files
for filename in selected_cats + selected_dogs:
    shutil.copy2(os.path.join(source_dir, filename), os.path.join(target_dir, filename))

print(
    f"Successfully added {len(selected_cats)} new cats and {len(selected_dogs)} new dogs."
)
