import os
import pandas as pd

DATASET_PATH = r"E:\MedicinalPlantProject\dataset"
OUTPUT_PATH = r"E:\MedicinalPlantProject\data\plant_info.csv"

data = []

for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)

    if os.path.isdir(folder_path):
        data.append({
            "plant_name": folder,
            "common_name": "",
            "parts_used": "",
            "diseases_treated": "",
            "description": ""
        })

df = pd.DataFrame(data)
df = df.sort_values("plant_name")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("plant_info.csv created successfully.")
