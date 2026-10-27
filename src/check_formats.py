import os

DATASET_PATH = "dataset"

valid_extensions = ('.jpg', '.jpeg', '.png')

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if not file.lower().endswith(valid_extensions):
            print("Invalid file:", os.path.join(root, file))
        