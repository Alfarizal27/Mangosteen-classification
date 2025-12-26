import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import joblib

# ======================================================
# KONFIGURASI
# ======================================================
IMAGE_DIR = "image path"
LABEL_CSV = "csv path"
IMG_SIZE = 128
RANDOM_STATE = 42

# ======================================================
# LOAD LABEL (GROUND TRUTH)
# ======================================================
def load_label(label_path):
    df = pd.read_csv(label_path, header=None, skiprows=1)
    df.columns = ["filename", "l1", "l2", "l3", "l4", "l5", "l6"]

    df["label"] = df[["l1","l2","l3","l4","l5","l6"]].idxmax(axis=1)
    df["label"] = df["label"].str.replace("l", "").astype(int)

    return df[["filename", "label"]]

label_df = load_label(LABEL_CSV)
print("Jumlah data berlabel:", len(label_df))

# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_color_features(rgb_img):
    return [
        np.mean(rgb_img[:,:,0]),
        np.mean(rgb_img[:,:,1]),
        np.mean(rgb_img[:,:,2]),
        np.std(rgb_img[:,:,0]),
        np.std(rgb_img[:,:,1]),
        np.std(rgb_img[:,:,2])
    ]

def extract_texture_features(gray_img):
    gray_uint8 = (gray_img * 255).astype(np.uint8)

    glcm = graycomatrix(
        gray_uint8,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    return [
        graycoprops(glcm, 'contrast')[0,0],
        graycoprops(glcm, 'homogeneity')[0,0],
        graycoprops(glcm, 'energy')[0,0],
        graycoprops(glcm, 'correlation')[0,0]
    ]

def extract_all_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0

    return extract_color_features(rgb) + extract_texture_features(gray)

# ======================================================
# EKSTRAKSI SEMUA DATA
# ======================================================
X, y = [], []

for _, row in label_df.iterrows():
    img_path = os.path.join(IMAGE_DIR, row["filename"])
    if not os.path.exists(img_path):
        continue

    X.append(extract_all_features(img_path))
    y.append(row["label"])

X = np.array(X)
y = np.array(y)

print("Shape fitur:", X.shape)
print("Distribusi label:\n", pd.Series(y).value_counts())

# ======================================================
# KMEANS (EKSPLORASI SAJA, BUKAN LABEL)
# ======================================================
scaler_mm = MinMaxScaler()
X_km = scaler_mm.fit_transform(X)

silhouette_scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    cluster_labels = km.fit_predict(X_km)
    silhouette_scores[k] = silhouette_score(X_km, cluster_labels)

print("Silhouette scores:", silhouette_scores)

best_k = max(silhouette_scores, key=silhouette_scores.get)
print("Best k:", best_k)

# Visualisasi PCA
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
clusters = kmeans.fit_predict(X_km)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_km)

plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="tab10")
plt.title(f"KMeans Visualization (k={best_k})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# ======================================================
# SUPERVISED LEARNING (VALID)
# ======================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# ---------------- RANDOM FOREST ----------------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n=== RANDOM FOREST ===")
print(classification_report(y_test, y_pred_rf, digits=3))
print("Macro-F1 RF:",
    f1_score(y_test, y_pred_rf, average='macro'))

# ---------------- SVM ----------------
svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True
)
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

print("\n=== SVM ===")
print(classification_report(y_test, y_pred_svm, digits=3))
print("Macro-F1 SVM:",
    f1_score(y_test, y_pred_svm, average='macro'))

# ======================================================
# SIMPAN MODEL UNTUK GUI
# ======================================================
joblib.dump(rf, "rf_model.pkl")
joblib.dump(svm, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel & scaler berhasil disimpan.")
