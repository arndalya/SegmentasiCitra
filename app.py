import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from utils import (
    extract_features_from_image,
    show_thresholded_image,
    get_equalized_image,
    get_contour_image
)

st.title("🌿 Aplikasi Klasifikasi Daun Mangrove")
st.write("Upload gambar daun untuk klasifikasi:")
st.markdown("- **Bruguiera**\n- **Sonneratia**\n- **Lumnitzera**")

uploaded_file = st.file_uploader("📤 Upload gambar daun", type=["jpg", "jpeg", "png"])

model = joblib.load("mangrove_model_knn.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

def plot_histogram(img_gray):
    fig, ax = plt.subplots()
    ax.hist(img_gray.ravel(), bins=256, range=(0, 256), color='gray')
    ax.set_title("Histogram Grayscale")
    st.pyplot(fig)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Gambar Diupload", use_column_width=True)

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    st.subheader("📊 Histogram Gambar Grayscale")
    plot_histogram(img_gray)

    st.subheader("🟠 Histogram Equalization")
    equalized = get_equalized_image(image_cv)
    st.image(equalized, caption="Grayscale Setelah Histogram Equalization", use_column_width=True, channels="GRAY")

    st.subheader("🔵 Kontur Daun")
    contoured = get_contour_image(image_cv)
    st.image(contoured, caption="Kontur Terdeteksi", use_column_width=True)

    try:
        features = extract_features_from_image(image_cv).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        label = label_encoder.inverse_transform([prediction])[0]
        st.success(f"✅ Daun tersebut adalah: **{label}**")
    except Exception as e:
        st.error(f"❌ Gagal klasifikasi: {e}")

    if st.checkbox("🧪 Tampilkan hasil Otsu Thresholding"):
        thresholded = show_thresholded_image(image_cv)
        st.image(thresholded, caption="Thresholding Otsu", use_column_width=True)
