"""
Randomly selects exactly ONE new dog image from 'unfiltered_photos' and copies it into
'sample_sets/photos/high_info', ensuring no duplicates with existing images.

Useful for fixing selection errors or missing files.
"""

import os
import shutil
import random

source_dir = "unfiltered_photos"
target_dir = "sample_sets/photos/high_info"

# Get existing filenames in target_dir
existing_photos = set(os.listdir(target_dir))

# Available dog images excluding existing ones
available_dogs = [
    f
    for f in os.listdir(source_dir)
    if f.startswith("dog.") and f not in existing_photos
]

# Randomly select ONE new dog
selected_dog = random.choice(available_dogs)

# Copy it to the target directory
shutil.copy2(
    os.path.join(source_dir, selected_dog), os.path.join(target_dir, selected_dog)
)

print(f"✅ Successfully added new dog image: {selected_dog}")
