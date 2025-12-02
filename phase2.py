import os
import cv2
import numpy as np
import csv
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# ==========================
# تنظیمات کلی
# ==========================

DATA_DIR = r"Dataset\Animals-10"   # ← مسیر دیتاست را اصلاح کن

CLASSES = [
    "butterfly", "cat", "chicken", "cow", "dog",
    "elephant", "horse", "sheep", "spider", "squirrel"
]

IMG_SIZE = (128, 128)  # اندازه تصویر برای یکسان‌سازی

# LBP
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

# Color Histogram
COLOR_BINS = 16

# GLCM – Haralick Features
GLCM_DIST = [1]  # فاصله پیکسل
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 4 زاویه


# ==========================
# توابع فیچرها
# ==========================

def extract_lbp(gray):
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins+1), density=True)
    return hist.astype(np.float32)


def extract_glcm(gray):
    # GLCM نیاز دارد تصویر 0..255 و نوع uint8 باشد
    gray_u8 = gray.astype(np.uint8)
    glcm = graycomatrix(gray_u8, distances=GLCM_DIST, angles=GLCM_ANGLES,
                        levels=256, symmetric=True, normed=True)

    features = []
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

    for p in props:
        feat = graycoprops(glcm, p)
        features.append(feat.mean())  # میانگین روی زاویه‌ها

    return np.array(features, dtype=np.float32)


def extract_color_hist(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist_feats = []

    ranges = [(0, 180), (0, 256), (0, 256)]

    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [COLOR_BINS], ranges[ch])
        h = cv2.normalize(h, h).flatten()
        hist_feats.append(h)

    return np.concatenate(hist_feats).astype(np.float32)


def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image {img_path}")

    # اگر تصویر ۱ کاناله بود → ۳ کاناله‌اش کن
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # اگر ۴ کاناله بود → آلفا را حذف کن
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = cv2.resize(img, IMG_SIZE)
    return img


def extract_features(img_path):
    img = load_image(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbp = extract_lbp(gray)
    glcm = extract_glcm(gray)
    color = extract_color_hist(img)

    return np.concatenate([lbp, glcm, color])


# ==========================
# ورود به CSV (استریمی – بدون مصرف زیاد حافظه)
# ==========================

def build_csv(data_dir, out_csv="features.csv"):
    header_written = False
    n_features = None

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for label_idx, cls in enumerate(CLASSES):
            class_dir = os.path.join(data_dir, cls)
            print(f"[INFO] Processing class: {cls}")

            for fname in os.listdir(class_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue

                path = os.path.join(class_dir, fname)

                try:
                    feat = extract_features(path)
                except Exception as e:
                    print(f"[ERROR] Skipping {path} → {e}")
                    continue

                if not header_written:
                    n_features = len(feat)
                    header = ["path", "class", "label"] + [f"f{i}" for i in range(n_features)]
                    writer.writerow(header)
                    header_written = True
                    print(f"[INFO] Feature length = {n_features}")

                row = [path, cls, label_idx] + feat.tolist()
                writer.writerow(row)

    print(f"[INFO] Saved to {out_csv}")


if __name__ == "__main__":
    build_csv(DATA_DIR, out_csv="features.csv")
