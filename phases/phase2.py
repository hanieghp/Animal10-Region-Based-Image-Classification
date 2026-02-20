import os
import csv
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis


DATA_DIR = r"Dataset\Animals-10"  

CLASSES = [
    "butterfly", "cat", "chicken", "cow", "dog",
    "elephant", "horse", "sheep", "spider", "squirrel",
]

PROCESS_SIZE = (128, 128) 

GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_N_BINS = LBP_POINTS + 2 
LBP_METHOD = "uniform"

COLOR_HIST_BINS = 16

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)


def read_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def compute_basic_dims(img):
    h, w = img.shape[:2]
    aspect_ratio = w / h if h > 0 else 0.0
    return w, h, aspect_ratio


def compute_rgb_stats(img):
    # img: BGR
    b, g, r = cv2.split(img.astype(np.float32))
    mean_red = r.mean()
    mean_green = g.mean()
    mean_blue = b.mean()
    std_red = r.std()
    std_green = g.std()
    std_blue = b.std()
    return mean_red, mean_green, mean_blue, std_red, std_green, std_blue


def compute_hsv_stats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    mean_h = h.mean()
    mean_s = s.mean()
    mean_v = v.mean()
    std_h = h.std()
    std_s = s.std()
    std_v = v.std()
    return mean_h, mean_s, mean_v, std_h, std_s, std_v


def compute_intensity_stats(gray):
    g = gray.astype(np.float32)
    mean_int = g.mean()
    std_int = g.std()
    min_int = g.min()
    max_int = g.max()
    range_int = max_int - min_int

    flat = g.ravel()
    if flat.size > 1:
        flat64 = flat.astype(np.float64, copy=False)
        skewness_val = float(skew(flat64, bias=False))
        kurtosis_val = float(kurtosis(flat64, fisher=True, bias=False))
    else:
        skewness_val = 0.0
        kurtosis_val = 0.0

    entropy_val = float(shannon_entropy(gray))

    return (
        mean_int,
        std_int,
        min_int,
        max_int,
        range_int,
        skewness_val,
        kurtosis_val,
        entropy_val,
    )


def compute_glcm_features(gray):
    gray_u8 = gray.astype(np.uint8)
    glcm = graycomatrix(
        gray_u8,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=256,
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity",
             "energy", "correlation", "ASM"]
    vals = [float(graycoprops(glcm, p).mean()) for p in props]
    return vals  # [contrast, dissimilarity, homogeneity, energy, correlation, ASM]


def compute_edge_features(gray):
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    mask = edges > 0
    if np.any(mask):
        vals = mag[mask]
        edge_mean = float(vals.mean())
        edge_std = float(vals.std())
    else:
        edge_mean = 0.0
        edge_std = 0.0

    return edge_mean, edge_std, edge_density


def compute_contour_features(gray):
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0, 0.0, 0.0

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    if area > 0:
        compactness = (perimeter ** 2) / (4 * np.pi * area)
    else:
        compactness = 0.0

    return area, perimeter, compactness


def compute_lbp_features(gray):
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(lbp, bins=LBP_N_BINS, range=(0, LBP_N_BINS), density=False)
    hist = hist.astype(np.float32)
    hist /= hist.sum() if hist.sum() > 0 else 1.0
    return hist


def compute_color_histograms(img):
    hist_channels = []
    for channel in cv2.split(img):
        hist = cv2.calcHist([channel], [0], None, [COLOR_HIST_BINS], [0, 256])
        hist = hist.flatten().astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        hist_channels.append(hist)
    return np.concatenate(hist_channels)


def compute_hog_features(gray):
    gray_float = gray.astype(np.float32, copy=False) / 255.0
    hog_vec = hog(
        gray_float,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return hog_vec.astype(np.float32)


def process_image(full_path):
    img_orig = read_image(full_path)
    _, _, aspect_ratio = compute_basic_dims(img_orig)

    img = cv2.resize(img_orig, PROCESS_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_r, mean_g, mean_b, std_r, std_g, std_b = compute_rgb_stats(img)

    mean_h, mean_s, mean_v, std_h, std_s, std_v = compute_hsv_stats(img)

    # intensity
    (
        mean_int,
        std_int,
        min_int,
        max_int,
        range_int,
        skewness_val,
        kurtosis_val,
        entropy_val,
    ) = compute_intensity_stats(gray)

    # brightness & contrast_simple
    brightness = mean_v           
    contrast_simple = std_int     

    # GLCM
    contrast, dissimilarity, homogeneity, energy, correlation, ASM = \
        compute_glcm_features(gray)

    # edges
    edge_mean, edge_std, edge_density = compute_edge_features(gray)

    # contour
    contour_area, contour_perimeter, compactness = compute_contour_features(gray)

    # texture via LBP histogram
    lbp_hist = compute_lbp_features(gray)

    # color distribution via normalized histograms per channel
    color_hist = compute_color_histograms(img)

    # gradient structure via HOG descriptor
    hog_feats = compute_hog_features(gray)

    feats = [
        aspect_ratio,
        mean_r,
        mean_g,
        mean_b,
        std_r,
        std_g,
        std_b,
        mean_h,
        mean_s,
        mean_v,
        std_h,
        std_s,
        std_v,
        brightness,
        contrast_simple,
        mean_int,
        std_int,
        min_int,
        max_int,
        range_int,
        skewness_val,
        kurtosis_val,
        entropy_val,
        contrast,
        dissimilarity,
        homogeneity,
        energy,
        correlation,
        ASM,
        edge_mean,
        edge_std,
        edge_density,
        contour_area,
        contour_perimeter,
        compactness,
    ]

    feats.extend(lbp_hist.tolist())
    feats.extend(color_hist.tolist())
    feats.extend(hog_feats.tolist())

    feats = np.array(feats).astype(np.float32, casting="unsafe")
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return feats


def build_feature_csv(data_dir, out_csv="feature2.csv"):
    header = [
        "filename", "class", "label",
        "aspect_ratio",
        "mean_red", "mean_green", "mean_blue",
        "std_red", "std_green", "std_blue",
        "mean_hue", "mean_saturation", "mean_value",
        "std_hue", "std_saturation", "std_value",
        "brightness", "contrast_simple",
        "mean_intensity", "std_intensity",
        "min_intensity", "max_intensity", "range_intensity",
        "skewness", "kurtosis", "entropy",
        "contrast", "dissimilarity", "homogeneity",
        "energy", "correlation", "ASM",
        "edge_mean", "edge_std", "edge_density",
        "contour_area", "contour_perimeter", "compactness",
    ]
    header.extend([f"lbp_hist_{i}" for i in range(LBP_N_BINS)])
    header.extend([f"color_hist_b_{i}" for i in range(COLOR_HIST_BINS)])
    header.extend([f"color_hist_g_{i}" for i in range(COLOR_HIST_BINS)])
    header.extend([f"color_hist_r_{i}" for i in range(COLOR_HIST_BINS)])

    hog_length = compute_hog_features(np.zeros(PROCESS_SIZE, dtype=np.uint8)).size
    header.extend([f"hog_{i}" for i in range(hog_length)])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for label_idx, cls in enumerate(CLASSES):
            class_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(class_dir):
                print(f"[WARNING] class folder not found: {class_dir}")
                continue

            print(f"[INFO] processing class: {cls}")

            for fname in os.listdir(class_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue

                full_path = os.path.join(class_dir, fname)

                try:
                    feats = process_image(full_path)
                except Exception as e:
                    print(f"[ERROR] skipping {full_path}: {e}")
                    continue

                row = [fname, cls, label_idx] + feats.tolist()
                writer.writerow(row)

    print(f"[INFO] features saved to {out_csv}")


if __name__ == "__main__":
    build_feature_csv(DATA_DIR, out_csv="feature2.csv")
