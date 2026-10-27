import os
import re

# Path to your dataset folder
DATASET_PATH = r"E:\MedicinalPlantProject\dataset"

def clean_name(name):
    # Convert to lowercase
    name = name.lower()

    # Remove content inside brackets
    name = re.sub(r"\(.*?\)", "", name)

    # Replace hyphens with space
    name = name.replace("-", " ")

    # Replace spaces with underscore
    name = name.replace(" ", "_")

    # Remove anything that is not letter, number, underscore
    name = re.sub(r"[^a-z0-9_]", "", name)

    # Remove multiple underscores
    name = re.sub(r"_+", "_", name)

    # Remove leading/trailing underscore
    name = name.strip("_")

    return name


for folder in os.listdir(DATASET_PATH):
    old_path = os.path.join(DATASET_PATH, folder)

    if os.path.isdir(old_path):
        new_name = clean_name(folder)
        new_path = os.path.join(DATASET_PATH, new_name)

        if old_path != new_path:
            print(f"Renaming: {folder}  -->  {new_name}")
            os.rename(old_path, new_path)

print("✅ Folder renaming completed.")