import pandas as pd
import os

# =============================
# PATHS
# =============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "dataset", "Medicinalplant_updated.xlsx")
OUTPUT_PATH = os.path.join(BASE_DIR, "dataset", "Medicinalplant_updated_lowercase.xlsx")

# =============================
# LOAD FILE
# =============================

df = pd.read_excel(INPUT_PATH, engine="openpyxl")

# =============================
# CONVERT ALL STRING VALUES TO LOWERCASE
# =============================

df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# =============================
# SAVE FILE
# =============================

df.to_excel(OUTPUT_PATH, index=False)

print("All values converted to lowercase successfully!")