import base64
import sys
import os
import re
from openai import OpenAI

# Allow file importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing.descriptions.ai_instructions import INSTRUCTIONS

client = OpenAI()


def encode_image(image_path):
    """Return a Base64-encoded string from a local image file."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def main():
    high_info_dir = "sample_sets/photos/high_info"
    high_info_output_dir = "sample_sets/descriptions/high_info"
    low_info_output_dir = "sample_sets/descriptions/low_info"

    # Ensure output directories exist
    os.makedirs(high_info_output_dir, exist_ok=True)
    os.makedirs(low_info_output_dir, exist_ok=True)

    # Gather all images in the high_info directory
    image_files = [f for f in os.listdir(high_info_dir) if f.lower().endswith(".jpg")]

    # Process each image
    for idx, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(high_info_dir, image_file)
        base64_image = encode_image(image_path)
        data_url = f"data:image/jpeg;base64,{base64_image}"

        # Call your model
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": INSTRUCTIONS},
                        {
                            "type": "input_image",
                            "image_url": data_url,
                            "detail": "high",
                        },
                    ],
                }
            ],
        )

        # Parse the response
        response_text = response.output_text.strip()
        match = re.match(
            r'high_info="(.+?)"\s+low_info="(.+?)"', response_text, re.DOTALL
        )

        if match:
            high_info_desc = match.group(1).strip()
            low_info_desc = match.group(2).strip()

            # Create paths for the new txt files based on the image file name
            base_name = os.path.splitext(image_file)[0]
            high_info_path = os.path.join(high_info_output_dir, base_name + ".txt")
            low_info_path = os.path.join(low_info_output_dir, base_name + ".txt")

            # Write out the descriptions
            with open(high_info_path, "w") as f:
                f.write(high_info_desc)

            with open(low_info_path, "w") as f:
                f.write(low_info_desc)

            print(
                f"Descriptions saved successfully for {image_file}:\n"
                f"  {high_info_path}\n  {low_info_path}"
            )
        else:
            print(
                f"Response format was incorrect for {image_file}. "
                f"Here's the response for debugging:\n{response_text}"
            )

        # Print progress every 10 images
        if idx % 10 == 0:
            print(f"Processed {idx} images so far...")

    # Optional: Print a final summary
    print(f"\nDone! Processed {len(image_files)} images in total.")


if __name__ == "__main__":
    main()
