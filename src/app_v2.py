from ai_module import generate_ai_response
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import os
from PIL import Image

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(
    page_title="MedLeaf AI",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 MedLeaf AI")
st.caption("Intelligent Medicinal Plant Classification & Knowledge System")

# =============================
# PATHS
# =============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")
DATABASE_PATH = os.path.join(BASE_DIR, "data", "plant_database.csv")
CONF_MATRIX_PATH = os.path.join(BASE_DIR, "models", "confusion_matrix.png")
ACCURACY_PLOT_PATH = os.path.join(BASE_DIR, "models", "accuracy_plot.png")

# =============================
# LOAD MODEL & DATA
# =============================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# Load medicinal database
plant_data = pd.read_csv(DATABASE_PATH)
plant_data.columns = plant_data.columns.str.strip().str.lower()

# =============================
# TABS
# =============================

tab1, tab2 = st.tabs(["🔍 Plant Detection", "📊 Admin Analytics"])

# ============================================================
# TAB 1 — USER PREDICTION
# ============================================================

with tab1:

    uploaded_file = st.file_uploader(
        "Upload a Plant Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        col1, col2 = st.columns([1, 1])

        image = Image.open(uploaded_file).convert("RGB")

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # =============================
        # IMAGE PREPROCESSING
        # =============================

        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]

        # =============================
        # TOP 3 PREDICTIONS
        # =============================

        top3_idx = prediction.argsort()[-3:][::-1]
        top3_probs = prediction[top3_idx]

        with col2:
            st.subheader("Top 3 Predictions")

            for i in range(3):
                plant_name = idx_to_class[top3_idx[i]]
                confidence = top3_probs[i] * 100

                st.write(f"**{plant_name}** — {confidence:.2f}%")
                st.progress(float(top3_probs[i]))

        predicted_class = idx_to_class[top3_idx[0]]
        confidence = top3_probs[0] * 100

        st.markdown("---")
        st.success(f"🌱 Predicted Plant: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        # =============================
        # MEDICINAL INFORMATION
        # =============================

        plant_info = plant_data[
            plant_data["indian medicinal plant"].str.lower() == predicted_class.lower()
        ]

        if not plant_info.empty:

            st.markdown("---")
            st.markdown("## 🌿 Medicinal Information")

            colA, colB = st.columns(2)

            with colA:
                st.markdown("### 🔬 Scientific Name")
                st.success(plant_info["scientific_name"].iloc[0])

            with colB:
                st.markdown("### 🌱 Common Name")
                st.success(plant_info["common_name"].iloc[0])

            st.markdown("### 🩺 Plant Parts & Therapeutic Uses")

            mapping_df = plant_info[["plant part", "therapeutic use"]] \
                .drop_duplicates()

            mapping_df.rename(columns={
                "plant part": "Plant Part",
                "therapeutic use": "Therapeutic Use"
            }, inplace=True)

            st.dataframe(mapping_df, use_container_width=True, hide_index=True)

        else:
            st.warning("Plant information not found in database.")

        # =============================
        # AI HERBAL GUIDANCE
        # =============================

        st.divider()
        st.subheader("🧠 AI Herbal Guidance")

        # 🔒 Safety: limit input size
        user_query = st.text_area(
            "Ask how this plant can be used (example: treatment for cough, preparation method)",
            max_chars=200,
            key="user_query"
        )

        if st.button("Generate AI Recommendation"):

            if user_query.strip() == "":
                st.warning("Please enter a question.")

            elif plant_info.empty:
                st.warning("No plant data available for AI response.")

            else:

                # ✅ FIX: Optimized small context (NO TOKEN OVERFLOW)
                plant_context = f"""
Plant Name: {predicted_class}

Scientific Name: {plant_info['scientific_name'].iloc[0]}
Common Name: {plant_info['common_name'].iloc[0]}

Uses:
{', '.join(plant_info['therapeutic use'].dropna().unique()[:5])}

Parts Used:
{', '.join(plant_info['plant part'].dropna().unique()[:5])}
"""

                with st.spinner("Generating AI guidance..."):

                    ai_response = generate_ai_response(plant_context, user_query)

                    if "AI Error" in ai_response:
                        st.error(ai_response)

                    else:
                        st.markdown(
                            f"""
                            <div style="
                                background:#0f172a;
                                padding:20px;
                                border-radius:10px;
                                border-left:5px solid #22c55e;
                                color:white;
                            ">
                            {ai_response}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.info(
                            "⚠️ Educational purpose only. Consult a medical professional."
                        )

# ============================================================
# TAB 2 — ADMIN ANALYTICS
# ============================================================

with tab2:

    st.subheader("📈 Model Performance")

    if os.path.exists(ACCURACY_PLOT_PATH):
        st.image(ACCURACY_PLOT_PATH, caption="Training Accuracy Curve")

    if os.path.exists(CONF_MATRIX_PATH):
        st.image(CONF_MATRIX_PATH, caption="Confusion Matrix")

    st.markdown("---")

    st.subheader("📊 Dataset Statistics")

    total_classes = len(class_indices)
    st.write("Number of Classes:", total_classes)

    total_records = len(plant_data)
    st.write("Total Knowledge Records:", total_records)

    st.success("System Ready for Deployment 🚀")