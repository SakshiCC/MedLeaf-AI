import pandas as pd
import os
import re

# =============================
# PATHS
# =============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MEDICINAL_PATH = os.path.join(BASE_DIR, "dataset", "Medicinalplant.xlsx")
DATA_PATH = os.path.join(BASE_DIR, "dataset", "data_1.xlsx")

OUTPUT_PATH = os.path.join(BASE_DIR, "dataset", "Medicinalplant_updated.xlsx")

# =============================
# LOAD FILES
# =============================

med_df = pd.read_excel(MEDICINAL_PATH, engine="openpyxl")
data_df = pd.read_excel(DATA_PATH, engine="openpyxl")

# =============================
# CLEAN FUNCTION
# =============================

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[()\-]", " ", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("_")
    return text

# -----------------------------
# Create matching key for medicinal file
# -----------------------------
med_df["match_key"] = med_df.iloc[:, 0].apply(normalize)

# -----------------------------
# Extract scientific + local from data file
# -----------------------------

def create_match_key(name):
    name = str(name)

    # extract scientific part before "-"
    scientific = name.split("-")[0].strip()

    # extract local name inside brackets
    local_match = re.search(r"\((.*?)\)", name)
    local = local_match.group(1).strip() if local_match else ""

    combined = scientific + " " + local
    return normalize(combined)

data_df["match_key"] = data_df.iloc[:, 1].apply(create_match_key)

# =============================
# SELECT REQUIRED COLUMNS
# =============================

data_subset = data_df[[
    "match_key",
    data_df.columns[1],  # scientific name
    data_df.columns[2],  # botanical name
    data_df.columns[3]   # local name
]]

# =============================
# MERGE
# =============================

merged = med_df.merge(
    data_subset,
    on="match_key",
    how="left"
)

# Rename columns
merged.rename(columns={
    data_df.columns[1]: "scientific_name",
    data_df.columns[2]: "botanical_name",
    data_df.columns[3]: "common_name"
}, inplace=True)

# Drop helper column
merged.drop(columns=["match_key"], inplace=True)

# =============================
# SAVE FILE
# =============================

merged.to_excel(OUTPUT_PATH, index=False)

print("✅ Medicinalplant_updated.xlsx created successfully!")