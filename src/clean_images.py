import os
from PIL import Image

DATASET_PATH = "dataset"

def clean_dataset(dataset_path):
    removed_count = 0

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                print(f"Removing corrupted file: {file_path}")
                os.remove(file_path)
                removed_count += 1

    print(f"\n✅ Cleaning completed. Removed {removed_count} corrupted images.")

if __name__ == "__main__":
    clean_dataset(DATASET_PATH)