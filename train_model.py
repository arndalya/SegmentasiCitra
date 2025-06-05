import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
from tqdm import tqdm

# -----------------------------------------------
# 1. AKUISISI & PERSIAPAN DATA
# -----------------------------------------------

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        for filename in os.listdir(label_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(label_folder, filename)
                image = cv2.imread(img_path)
                if image is not None:
                    images.append(image)
                    labels.append(label)
    return images, labels

# -----------------------------------------------
# 2-6. PRAPROSES, GRAYSCALE, OTSU, EKSTRAKSI FITUR
# -----------------------------------------------

def preprocess_and_extract_features(image):
    # Resize
    image = cv2.resize(image, (128, 128))

    # HSV masking
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 40, 40])  # rentang hijau
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    masked_img = cv2.bitwise_and(image, image, mask=mask)

    # Grayscale
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

    # Otsu Thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ekstraksi fitur bentuk
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

    # Ekstraksi fitur warna dominan (rata-rata RGB)
    mean_color = cv2.mean(image)[:3]  # ambil R,G,B

    # Gabungkan fitur (bentuk + warna)
    feature_vector = [area, perimeter, aspect_ratio, extent] + list(mean_color)
    return feature_vector

# -----------------------------------------------
# 7. AUGMENTASI (opsional: flip horizontal saja)
# -----------------------------------------------

def augment(image):
    return [image, cv2.flip(image, 1)]

# -----------------------------------------------
# 8. KLASIFIKASI
# -----------------------------------------------

def prepare_data(folder):
    images, labels = load_images_from_folder(folder)
    X, y = [], []

    for img, label in tqdm(zip(images, labels), total=len(images)):
        for augmented in augment(img):
            features = preprocess_and_extract_features(augmented)
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

# MAIN PIPELINE
train_folder = 'dataset/train'
test_folder = 'dataset/test'

# Load dan proses data latih & uji
X_train, y_train = prepare_data(train_folder)
X_test, y_test = prepare_data(test_folder)

# Standarisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reduksi dimensi dengan PCA
pca = PCA(n_components=4)  # Misalnya reduksi jadi 4 dimensi
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Klasifikasi dengan KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_pca, y_train)

# Simpan model
joblib.dump((model, scaler, pca), 'model_knn.pkl')

# Evaluasi
y_pred = model.predict(X_test_pca)
print(classification_report(y_test, y_pred))
