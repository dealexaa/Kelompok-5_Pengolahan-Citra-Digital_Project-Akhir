import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_paru.h5")

model = load_model()

class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((224,224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# GRAD-CAM
# =========================
def make_gradcam_heatmap(img_array, model):

    last_conv_layer = "conv5_block3_out"

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)

        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return np.zeros((224,224))

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

# =========================
# OVERLAY
# =========================
def overlay_heatmap(heatmap, image):
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = np.array(image)
    result = heatmap * 0.4 + img

    return result.astype("uint8")

# =========================
# UI
# =========================
st.title("🫁 Deteksi Penyakit Paru AI (ResNet50)")

uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg","png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Gambar Asli")

    img_array = preprocess(image)

    with st.spinner("Analisis AI..."):
        pred = model.predict(img_array)

    confidence = float(np.max(pred))*100
    label = class_names[np.argmax(pred)]

    st.markdown(f"## 🧾 Hasil: **{label}**")
    st.markdown(f"### 🎯 Akurasi: **{confidence:.2f}%**")

    heatmap = make_gradcam_heatmap(img_array, model)
    result = overlay_heatmap(heatmap, image)

    with col2:
        st.image(result, caption="Grad-CAM")