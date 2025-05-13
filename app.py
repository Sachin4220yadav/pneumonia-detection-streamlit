import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("pneumonia_detection_model.h5")
IMG_SIZE = (224, 224)

st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a Chest X-ray image (JPG/PNG)")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
    confidence = pred if pred > 0.5 else 1 - pred

    st.subheader("Prediction:")
    st.markdown(f"**{label}** with **{confidence*100:.2f}% confidence**")
