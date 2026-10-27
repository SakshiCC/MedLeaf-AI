import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

converted = 0
removed = 0

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        file_path = os.path.join(root, file)

        try:
            with Image.open(file_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(file_path)
                converted += 1
        except Exception:
            print("Removing corrupted file:", file_path)
            os.remove(file_path)
            removed += 1

print(f"\n✅ Converted {converted} images")
print(f"🗑 Removed {removed} corrupted images")