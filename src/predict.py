import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore

# ==============================
# 🔹 Configuration
# ==============================

MODEL_PATH = "../models/best_model.keras"
CLASS_INDICES_PATH = "../models/class_indices.json"
IMAGE_SIZE = (224, 224)


# ==============================
# 🔹 Load Model & Class Mapping
# (Loaded only once)
# ==============================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(CLASS_INDICES_PATH):
    raise FileNotFoundError(f"Class indices file not found: {CLASS_INDICES_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index → class name
index_to_class = {v: k for k, v in class_indices.items()}


# ==============================
# 🔹 Image Preprocessing
# ==============================

def preprocess_image(img_path):
    """
    Loads and preprocesses image for prediction.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)

    # Same normalization used during training
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ==============================
# 🔹 Prediction Function
# ==============================

def predict_plant(img_path):
    """
    Predicts plant name and confidence score.
    
    Returns:
        plant_name (str)
        confidence (float)
    """

    processed_image = preprocess_image(img_path)

    predictions = model.predict(processed_image)

    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    plant_name = index_to_class[predicted_index]

    return plant_name, confidence


# ==============================
# 🔹 Optional Test Block
# ==============================

if __name__ == "__main__":
    test_image_path = "E:\mintimg.jpeg"  # Replace with your test image path

    try:
        plant, conf = predict_plant(test_image_path)
        print(f"🌿 Predicted Plant: {plant}")
        print(f"🔎 Confidence: {conf:.4f}")
    except Exception as e:
        print("Error:", e)