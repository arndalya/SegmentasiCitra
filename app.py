import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model, scaler, pca = joblib.load("model_knn.pkl")

# -----------------------------------
# Ekstraksi fitur dan segmentasi
# -----------------------------------
def preprocess_and_extract_features(image):
    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    masked_img = cv2.bitwise_and(image, image, mask=mask)

    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h != 0 else 0
        extent = float(area) / (w * h) if w * h != 0 else 0
    else:
        area = perimeter = aspect_ratio = extent = 0

    mean_color = cv2.mean(image)[:3]
    feature_vector = [area, perimeter, aspect_ratio, extent] + list(mean_color)

    segmented = cv2.bitwise_and(image, image, mask=otsu)
    return feature_vector, otsu, segmented, image, hsv, contours

# -----------------------------------
# Histogram
# -----------------------------------
def plot_rgb_histogram(image):
    plt.figure(figsize=(6, 4))
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
    plt.title("Histogram RGB")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_hsv_histogram(hsv_img):
    h_hist = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])
    plt.figure(figsize=(6, 4))
    plt.plot(h_hist, label='Hue', color='purple')
    plt.plot(s_hist, label='Saturation', color='orange')
    plt.title("Histogram HSV (H & S)")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

# -----------------------------------
# STREAMLIT UI
# -----------------------------------
st.set_page_config(page_title="Klasifikasi Daun Mangrove", layout="wide")
st.title("🌿 Klasifikasi Daun Mangrove")
st.markdown("Upload gambar daun (JPG/PNG/JPEG) untuk memprediksi jenis, segmentasi, histogram, dan fitur.")

uploaded_file = st.file_uploader("📂 Upload Gambar Daun", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Gambar Asli", use_container_width=True)

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    features, otsu_mask, segmented_leaf, original_bgr, hsv_img, contours = preprocess_and_extract_features(image_cv)

    # Prediksi
    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)
    prediction = model.predict(features_pca)[0]

    # Confidence
    distances, _ = model.kneighbors(features_pca)
    confidence_score = np.exp(-np.mean(distances)) * 100
    confidence_score = np.clip(confidence_score, 0, 100)

    st.success(f"🌱 Prediksi Jenis Daun: **{prediction}**")
    st.info(f"🔍 Confidence: **{confidence_score:.2f}%**")

    # Layout dengan 2 kolom (opsional)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔳 Segmentasi Otsu (Biner)")
        st.image(otsu_mask, channels="GRAY", use_container_width=True)

    with col2:
        st.markdown("### 🌿 Segmentasi Daun Berwarna")
        st.image(cv2.cvtColor(segmented_leaf, cv2.COLOR_BGR2RGB), caption="Daun Tersegmentasi", use_container_width=True)

    # Kontur
    overlay = original_bgr.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    st.markdown("### 🧭 Kontur Daun")
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Kontur Daun", use_container_width=True)

    # Histogram
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 📊 Histogram Warna RGB")
        plot_rgb_histogram(original_bgr)

    with col4:
        st.markdown("### 📊 Histogram HSV (H & S)")
        plot_hsv_histogram(hsv_img)

    # Tabel Fitur
    st.markdown("### 📐 Fitur Ekstraksi")
    df_features = pd.DataFrame({
        "Fitur": [
            "Luas (Area)",
            "Keliling (Perimeter)",
            "Aspect Ratio",
            "Extent",
            "Mean B",
            "Mean G",
            "Mean R"
        ],
        "Nilai": [
            round(features[0], 2),
            round(features[1], 2),
            round(features[2], 2),
            round(features[3], 2),
            round(features[4], 2),
            round(features[5], 2),
            round(features[6], 2)
        ]
    })
    st.table(df_features)


