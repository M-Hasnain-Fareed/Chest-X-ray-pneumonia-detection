import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Pneumonia ANN Diagnostic", page_icon="🫁", layout="centered"
)

st.title("🫁 Deep ANN Pneumonia Detector")
st.write(
    "Multi-Layer Perceptron (MLP) diagnostic engine. Expects PA/AP thoracic radiographs."
)


@st.cache_resource
def load_model():
    # Grabs the brain you generated in Phase 3
    return tf.keras.models.load_model("deep_ann_pneumonia.keras")


try:
    model = load_model()
except Exception:
    st.error(
        "Brain missing! Run `model_training.ipynb` first to generate the .keras file."
    )
    st.stop()

uploaded_file = st.file_uploader("Upload X-Ray (PNG/JPG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Force the image to Grayscale ('L' stands for Luma/Grayscale in PIL)
    raw_image = Image.open(uploaded_file).convert("L")

    st.image(raw_image, caption="Uploaded Radiograph", width=350)

    if st.button("🔬 Process via Dense Layers", type="primary"):
        with st.spinner("Flattening tensor & calculating neuron weights..."):
            # 2. Resize to the exact 128x128 our ANN expects
            img_resized = raw_image.resize((128, 128))

            # 3. Turn image into a numpy array of numbers
            img_array = np.array(img_resized)

            # 4. Reshape from (128, 128) -> (1, 128, 128, 1)
            # (The first '1' means "Batch of 1 image", the last '1' means "1 Grayscale color channel")
            input_tensor = np.expand_dims(img_array, axis=-1)
            input_tensor = np.expand_dims(input_tensor, axis=0)

            # 5. Get the prediction float (e.g. 0.872)
            prediction = model.predict(input_tensor)[0][0]

            st.divider()

            if prediction > 0.5:
                confidence = prediction * 100
                st.error(f"### ⚠️ POSITIVE FOR PNEUMONIA")
                st.write(f"**Network Confidence:** `{confidence:.1f}%`")
                st.caption(
                    "High neural activation caused by dense white lung opacity."
                )
            else:
                confidence = (1 - prediction) * 100
                st.success(f"### 🟢 NORMAL / CLEAR")
                st.write(f"**Network Confidence:** `{confidence:.1f}%`")
                st.caption(
                    "Low neural activation; standard dark lung volume detected."
                )