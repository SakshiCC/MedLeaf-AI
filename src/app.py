import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from PIL import Image
import os

# =============================
# LOAD MODEL & DATA
# =============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")
DATABASE_PATH = os.path.join(BASE_DIR, "data", "plant_database.csv")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping
idx_to_class = {v: k for k, v in class_indices.items()}

plant_data = pd.read_csv(DATABASE_PATH)

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(page_title="MedLeaf AI", layout="wide")

st.title("🌿 MedLeaf AI")
st.subheader("Intelligent Medicinal Plant Classification & Knowledge System")

# =============================
# IMAGE UPLOAD
# =============================

uploaded_file = st.file_uploader("Upload a Plant Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    plant_name = idx_to_class[predicted_class]

    st.success(f"Predicted Plant: {plant_name}")
    st.info(f"Confidence: {confidence*100:.2f}%")

    # =============================
    # FETCH PLANT DETAILS
    # =============================

    plant_info = plant_data[plant_data["Indian medicinal plant"] == plant_name]

    if not plant_info.empty:

        st.markdown("### 🌱 Plant Information")

        st.write("**Scientific Name:**", plant_info["scientific_name"].iloc[0])
        st.write("**Common Name:**", plant_info["common_name"].iloc[0])
        st.write("**Botanical Name:**", plant_info["botanical_name"].iloc[0])

        st.markdown("### 🌿 Plant Parts Used & Therapeutic Uses")

        for _, row in plant_info.iterrows():
            st.write(f"• **{row['Plant part']}** → {row['Therapeutic use']}")

    else:
        st.warning("Plant information not found in database.")