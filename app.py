import streamlit as st
import keras
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

MODEL_REPO = "24f2004275/potato-disease-classifier"
MODEL_FILENAME = "potato-disease-model.keras"
CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
IMAGE_SIZE = 256


@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    model = keras.models.load_model(model_path)
    return model


def predict(image, model):
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = round(100 * float(np.max(predictions[0])), 2)
    return CLASS_NAMES[class_idx], confidence


st.set_page_config(page_title="Potato Disease Classifier", layout="centered")

st.title("Potato Disease Classifier")
st.markdown("Upload a potato leaf image to classify it as **Early Blight**, **Late Blight**, or **Healthy**.")

with st.spinner("Loading model from Hugging Face Hub..."):
    model = load_model()

uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        class_name, confidence = predict(image, model)

    emoji_map = {
        "Potato___Early_blight": "⚠️",
        "Potato___Late_blight": "❗",
        "Potato___healthy": "✅",
    }
    display_name = class_name.replace("Potato___", "").replace("_", " ")
    st.markdown(f"{emoji_map.get(class_name, '')} ## {display_name}")
    st.markdown(f"**Confidence:** {confidence}%")
