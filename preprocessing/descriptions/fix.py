import base64
import sys
import os
import re
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing.descriptions.ai_instructions import INSTRUCTIONS

client = OpenAI()

missing_files = ["dog.1287.jpg"]

photo_dir = "sample_sets/photos/high_info"
high_info_output_dir = "sample_sets/descriptions/high_info"
low_info_output_dir = "sample_sets/descriptions/low_info"

os.makedirs(high_info_output_dir, exist_ok=True)
os.makedirs(low_info_output_dir, exist_ok=True)


def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def regenerate_description(image_file):
    image_path = os.path.join(photo_dir, image_file)
    base64_image = encode_image(image_path)
    data_url = f"data:image/jpeg;base64,{base64_image}"

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": INSTRUCTIONS},
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ],
            }
        ],
    )

    response_text = response.output_text.strip()
    match = re.match(r'high_info="(.+?)"\s+low_info="(.+?)"', response_text, re.DOTALL)

    if match:
        high_info_desc = match.group(1).strip()
        low_info_desc = match.group(2).strip()

        base_name = os.path.splitext(image_file)[0]
        high_info_path = os.path.join(high_info_output_dir, base_name + ".txt")
        low_info_path = os.path.join(low_info_output_dir, base_name + ".txt")

        with open(high_info_path, "w") as f:
            f.write(high_info_desc)

        with open(low_info_path, "w") as f:
            f.write(low_info_desc)

        print(f"✅ Descriptions generated for {image_file}")
    else:
        print(f"❌ Incorrect response format for {image_file}:\n{response_text}")


def main():
    for image_file in missing_files:
        regenerate_description(image_file)


if __name__ == "__main__":
    main()
