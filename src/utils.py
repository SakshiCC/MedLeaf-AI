import os
import pandas as pd

# ==============================
# 🔹 Configuration
# ==============================

CSV_PATH = "../data/plant_database.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

# Load CSV
df = pd.read_csv(CSV_PATH)

# Clean column names (important!)
df.columns = df.columns.str.strip().str.lower()

# Rename columns to easier names
df.rename(columns={
    "indian medicinal plant": "plant_name",
    "plant part": "plant_part",
    "therapeutic use": "therapeutic_use"
}, inplace=True)

# Make plant_name lowercase for matching
df["plant_name"] = df["plant_name"].str.lower()


# ==============================
# Get Plant Information
# ==============================

def get_plant_info(plant_name):
    """
    Returns plant information including full part-use mapping.
    """

    plant_name = plant_name.lower()

    plant_rows = df[df["plant_name"] == plant_name]

    if plant_rows.empty:
        return None

    # Extract general info (same for all rows)
    scientific_name = plant_rows.iloc[0]["scientific_name"]
    botanical_name = plant_rows.iloc[0]["botanical_name"]
    common_name = plant_rows.iloc[0]["common_name"]

    # Create mapping dataframe (only required columns)
    mapping_df = plant_rows[["plant_part", "therapeutic_use"]].drop_duplicates()

    return {
        "plant_name": plant_name,
        "scientific_name": scientific_name,
        "botanical_name": botanical_name,
        "common_name": common_name,
        "mapping_df": mapping_df
    }

    # Collect unique values
    plant_parts = ", ".join(plant_rows["plant_part"].dropna().unique())
    therapeutic_uses = ", ".join(plant_rows["therapeutic_use"].dropna().unique())

    # These columns are same for all rows of that plant
    scientific_name = plant_rows.iloc[0]["scientific_name"]
    botanical_name = plant_rows.iloc[0]["botanical_name"]
    common_name = plant_rows.iloc[0]["common_name"]

    return {
        "plant_name": plant_name,
        "scientific_name": scientific_name,
        "botanical_name": botanical_name,
        "common_name": common_name,
        "plant_parts": plant_parts,
        "therapeutic_uses": therapeutic_uses
    }


# ==============================
# 🔹 Test Block
# ==============================

if __name__ == "__main__":
    test = get_plant_info("aegle_marmelos_bel_patra")
    print(test)