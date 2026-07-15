import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
MODEL_PATH = "pneumonia_model.keras"
IMG_SIZE = (224, 224)
CLASSES = ["NORMAL", "PNEUMONIA"]

st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="centered")


# ---------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize, convert to array, normalize, add batch dimension."""
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr


def predict(model, img_arr: np.ndarray):
    """Model has a single sigmoid output -> probability of class 1 (PNEUMONIA)."""
    prob_pneumonia = float(model.predict(img_arr)[0][0])
    pred_idx = 1 if prob_pneumonia >= 0.5 else 0
    confidence = prob_pneumonia if pred_idx == 1 else 1 - prob_pneumonia
    return CLASSES[pred_idx], confidence, prob_pneumonia


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.write(
    "Upload a chest X-ray image and the CNN model will predict whether it shows "
    "signs of **pneumonia** or is **normal**."
)

st.warning(
    "⚠️ This tool is for educational/demo purposes only and is **not** a medical "
    "diagnostic device. Always consult a qualified radiologist or physician."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    with st.spinner("Analyzing image..."):
        try:
            model = get_model()
            img_arr = preprocess_image(img)
            label, confidence, raw_prob = predict(model, img_arr)
        except FileNotFoundError:
            st.error(
                f"Model file '{MODEL_PATH}' not found. Place it in the same "
                "folder as app.py."
            )
            st.stop()

    st.subheader("Prediction")
    if label == "PNEUMONIA":
        st.error(f"**{label}** detected — confidence: {confidence * 100:.2f}%")
    else:
        st.success(f"**{label}** — confidence: {confidence * 100:.2f}%")

    with st.expander("See raw model output"):
        st.write(f"Raw sigmoid output (P(PNEUMONIA)): `{raw_prob:.4f}`")
else:
    st.info("👆 Upload an image to get a prediction.")