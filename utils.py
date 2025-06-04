# utils.py
import cv2
import numpy as np
from skimage.feature import hog
from rembg import remove
from PIL import Image
import io

def get_equalized_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

def show_thresholded_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def get_contour_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img_bgr.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    return contour_img

def remove_background(img_bgr):
    """Menghapus background dari citra BGR dan mengembalikan array BGR tanpa latar belakang"""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    img_no_bg = remove(img_pil)
    img_pil_result = Image.open(io.BytesIO(img_no_bg)).convert("RGB")
    img_bgr_result = cv2.cvtColor(np.array(img_pil_result), cv2.COLOR_RGB2BGR)
    return img_bgr_result

def extract_features_from_image(img_bgr):
    # Hapus background
    img_no_bg = remove_background(img_bgr)

    # Konversi ke grayscale
    gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)

    # Otsu Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Cari kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Tidak ada kontur ditemukan setelah penghapusan background.")
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    # Resize untuk fitur HOG
    resized = cv2.resize(gray, (128, 128))
    hog_features, _ = hog(resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)

    return np.hstack([area, perimeter, hog_features])
