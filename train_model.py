import os
import glob
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def extract_features_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    features = resized.flatten()
    return features

DATASET_DIR = "dataset/mangrove"
LABELS = ["Bruguiera", "Sonneratia", "Lumnitzera"]

X = []
y = []

print("[INFO] Memulai ekstraksi fitur dari dataset...")

for label in LABELS:
    class_dir = os.path.join(DATASET_DIR, label)
    image_paths = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                  glob.glob(os.path.join(class_dir, "*.jpeg")) + \
                  glob.glob(os.path.join(class_dir, "*.png"))
    
    print(f"[INFO] Memproses label: {label}, Jumlah gambar ditemukan: {len(image_paths)}")
    
    if len(image_paths) == 0:
        print(f"[WARNING] Tidak ditemukan gambar untuk kelas {label}, periksa folder dan ekstensi file!")
    
    for img_path in image_paths:
        print(f"[DEBUG] Membaca gambar: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Gagal membaca gambar {img_path}, melewati file ini.")
            continue
        
        features = extract_features_from_image(img)
        X.append(features)
        y.append(label)

if len(X) == 0:
    print("[ERROR] Tidak ada fitur yang berhasil diekstrak, hentikan proses.")
    exit(1)

X = np.array(X)
y = np.array(y)

print(f"[INFO] Total data fitur: {X.shape}, Total label: {y.shape}")

# Label encoding dari string ke angka
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Skala fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print("[INFO] Melatih model KNN...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluasi model
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[INFO] Akurasi pada data test: {acc:.4f}")
print("[INFO] Laporan klasifikasi:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Simpan model, scaler, dan label encoder
joblib.dump(knn, "mangrove_model_knn.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("[INFO] Model, scaler, dan label encoder berhasil disimpan.")
