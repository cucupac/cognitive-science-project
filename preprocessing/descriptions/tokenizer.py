import os
import matplotlib.pyplot as plt
from transformers import CLIPTokenizerFast

# Use the exact tokenizer CLIP uses (GPT-2 BPE tokenizer)
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

desc_dir = "sample_sets/descriptions/high_info"
token_counts = []

# Loop through all description files
for filename in os.listdir(desc_dir):
    if filename.endswith(".txt"):
        path = os.path.join(desc_dir, filename)
        with open(path, "r") as f:
            text = f.read().strip()
            tokens = tokenizer.encode(text)
            num_tokens = len(tokens)
            token_counts.append((filename, num_tokens))

# Extract counts for plotting
counts_only = [count for _, count in token_counts]

# Plot the distribution
plt.hist(counts_only, bins=20, color="skyblue", edgecolor="black")
plt.axvline(x=77, color="red", linestyle="--", label="Token Limit (77)")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.title("Token Count Distribution of High-Info Descriptions (CLIP Tokenizer)")
plt.legend()
plt.show()

# Identify descriptions exceeding 77 tokens
over_limit = [(fname, count) for fname, count in token_counts if count > 77]

if over_limit:
    print("Descriptions exceeding 77 tokens (CLIP tokenizer):")
    for fname, count in over_limit:
        print(f"{fname}: {count} tokens")
else:
    print("✅ No descriptions exceed 77 tokens (CLIP tokenizer).")
