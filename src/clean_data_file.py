import pandas as pd
import re

# Load the Excel file
file_path = "E:\MedicinalPlantProject\dataset\data.xlsx"   # change path if needed
df = pd.read_excel(file_path)

# Function to clean each cell value
def clean_value(value):
    if isinstance(value, str):
        value = value.lower()                 # convert to lowercase
        value = re.sub(r"[()]", "", value)    # remove brackets only
        value = value.replace(" ", "_")       # replace spaces with underscore
    return value

# Apply cleaning to all cells
df_cleaned = df.map(clean_value)

# Save cleaned file
df_cleaned.to_excel("data_1.xlsx", index=False)

print("File cleaned and saved as data_1.xlsx")