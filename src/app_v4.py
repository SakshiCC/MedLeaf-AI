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
# SESSION STATE
# =============================

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "last_plant_info" not in st.session_state:
    st.session_state.last_plant_info = None

if "last_predicted_class" not in st.session_state:
    st.session_state.last_predicted_class = None

# =============================
# SIDEBAR (CHATGPT STYLE)
# =============================

st.sidebar.title("💬 Chats")

# ➕ New Chat
if st.sidebar.button("➕ New Chat"):
    chat_id = f"Chat {len(st.session_state.chats) + 1}"
    st.session_state.chats[chat_id] = []
    st.session_state.current_chat = chat_id

# 📋 Show chat titles only
for chat_id in st.session_state.chats.keys():
    if st.sidebar.button(chat_id):
        st.session_state.current_chat = chat_id

# 🗑️ Clear current chat
if st.session_state.current_chat:
    if st.sidebar.button("🗑️ Clear Current Chat"):
        st.session_state.chats[st.session_state.current_chat] = []

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

plant_data = pd.read_csv(DATABASE_PATH)
plant_data.columns = plant_data.columns.str.strip().str.lower()

# =============================
# TABS
# =============================

tab1, tab2, tab3 = st.tabs([
    "🔍 Plant Detection",
    "📊 Admin Analytics",
    "💬 AI Chatbot"
])

# ============================================================
# TAB 1 — PLANT DETECTION
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

        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]

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

        # Save for chatbot
        st.session_state.last_predicted_class = predicted_class

        plant_info = plant_data[
            plant_data["indian medicinal plant"].str.lower() == predicted_class.lower()
        ]

        st.session_state.last_plant_info = plant_info

        st.markdown("---")
        st.success(f"🌱 Predicted Plant: {predicted_class.title().replace('_', ' ')}")

        # =============================
        # MEDICINAL UI (RESTORED)
        # =============================

        if not plant_info.empty:

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
            st.warning("Plant information not found.")

# ============================================================
# TAB 2 — ANALYTICS
# ============================================================

with tab2:

    st.subheader("📈 Model Performance")

    if os.path.exists(ACCURACY_PLOT_PATH):
        st.image(ACCURACY_PLOT_PATH)

    if os.path.exists(CONF_MATRIX_PATH):
        st.image(CONF_MATRIX_PATH)

    st.markdown("---")

    st.write("Total Classes:", len(class_indices))
    st.write("Total Records:", len(plant_data))

# ============================================================
# TAB 3 — CHATBOT (CHATGPT STYLE)
# ============================================================

with tab3:

    st.subheader("💬 AI Herbal Chatbot")

    if st.session_state.last_plant_info is None:
        st.warning("⚠️ Please detect a plant first.")
    else:

        if st.session_state.current_chat is None:
            st.info("👉 Click 'New Chat' in sidebar to start.")
        else:

            messages = st.session_state.chats[st.session_state.current_chat]

            # Display chat
            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Fixed input
            user_query = st.chat_input("Ask about the detected plant...")

            if user_query:

                plant_info = st.session_state.last_plant_info
                predicted_class = st.session_state.last_predicted_class

                messages.append({
                    "role": "user",
                    "content": user_query
                })

                # Context
                plant_context = f"""
Plant Name: {predicted_class}

Scientific Name: {plant_info['scientific_name'].iloc[0]}
Common Name: {plant_info['common_name'].iloc[0]}

Uses:
{', '.join(plant_info['therapeutic use'].dropna().unique()[:5])}

Parts Used:
{', '.join(plant_info['plant part'].dropna().unique()[:5])}
"""

                history = ""
                for m in messages[-6:]:
                    history += f"{m['role']}: {m['content']}\n"

                full_query = f"{history}\nUser: {user_query}"

                with st.spinner("Thinking..."):
                    ai_response = generate_ai_response(
                        plant_context,
                        full_query
                    )

                messages.append({
                    "role": "assistant",
                    "content": ai_response
                })

                st.rerun()