import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="🫁 Pneumonia Detection", layout="centered")

st.title("🫁 Pneumonia Detection System")

model = tf.keras.models.load_model("pneumonia_model.h5")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=200)

    if st.button("Detect Pneumonia"):
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        confidence = prediction if prediction > 0.5 else 1 - prediction

        if prediction > 0.5:
            st.error(f"❌ Prediction: Pneumonia found")
        else:
            st.success(f"✅ Prediction: Normal (Pneumonia not found)")

        st.info(f"Confidence: {confidence * 100:.2f}%")
