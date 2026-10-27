import pandas as pd
import re

def find_plants_for_disease(df, query):

    query = query.lower()

    # Extract meaningful keywords
    keywords = re.findall(r'\b[a-z]+\b', query)

    matches = pd.DataFrame()

    for word in keywords:
        temp = df[df["therapeutic use"].str.lower().str.contains(word, na=False)]
        matches = pd.concat([matches, temp])

    matches = matches.drop_duplicates()

    if matches.empty:
        return "No medicinal plant found in the database for this condition."

    result = ""

    for _, row in matches.iterrows():

        result += f"""
🌿 Plant: {row['indian medicinal plant']}
🔬 Scientific Name: {row['scientific_name']}
🌱 Common Name: {row['common_name']}
🩺 Plant Part Used: {row['plant part']}
💊 Therapeutic Use: {row['therapeutic use']}

----------------------------------
"""

    return result