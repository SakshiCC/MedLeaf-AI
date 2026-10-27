import pandas as pd
import os

# =============================
# PATHS
# =============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "dataset", "Medicinalplant_updated_lowercase.xlsx")
OUTPUT_PATH = os.path.join(BASE_DIR, "dataset", "plant_database.csv")

# =============================
# LOAD EXCEL
# =============================

df = pd.read_excel(INPUT_PATH, engine="openpyxl")

# =============================
# SAVE AS CSV
# =============================

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(" plant_database.csv created successfully!")