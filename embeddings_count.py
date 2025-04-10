import os

base_dir = "vector_store"
subdirs = [
    "image_embeddings/high_info",
    "image_embeddings/low_info/dropout_50",
    "text_embeddings/high_info",
    "text_embeddings/low_info",
]

expected_count = 1000


def count_embeddings(path):
    return len([f for f in os.listdir(path) if f.endswith(".npy")])


def main():
    all_good = True
    for subdir in subdirs:
        full_path = os.path.join(base_dir, subdir)
        count = count_embeddings(full_path)
        if count != expected_count:
            print(
                f"❌ Issue found in '{subdir}': {count} embeddings (expected {expected_count})"
            )
            all_good = False
        else:
            print(f"✅ '{subdir}' has {count} embeddings (correct)")

    if all_good:
        print("\n🎉 All embedding counts are correct.")
    else:
        print("\n⚠️ Some embedding counts need attention.")


if __name__ == "__main__":
    main()
