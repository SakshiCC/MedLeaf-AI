import os
from PIL import Image

DATASET_PATH = "dataset"

bad_files = []

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        file_path = os.path.join(root, file)

        try:
            img = Image.open(file_path)
            img = img.convert("RGB")  # Force decode
            img.load()                # Fully load image
        except Exception as e:
            print("❌ Problematic file:", file_path)
            print("Error:", e)
            bad_files.append(file_path)

print("\nDone.")
print("Total bad files:", len(bad_files))
