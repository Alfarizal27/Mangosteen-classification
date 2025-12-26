import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
import sys
sys.stdout.reconfigure(encoding='utf-8')


# =========================
# KONFIGURASI
# =========================
USE_SVM = True          # False → Random Forest
IMG_SIZE = 128
CONF_THRESHOLD = 0.50

# =========================
# LOAD MODEL
# =========================
if USE_SVM:
    model = joblib.load("svm_model.pkl")
else:
    model = joblib.load("rf_model.pkl")

scaler = joblib.load("scaler.pkl")

GRADE_MAP = {
    1: "Grade C",
    2: "Grade C",
    3: "Grade B",
    4: "Grade B",
    5: "Grade A",
    6: "Grade A"
}

# =========================
# FEATURE EXTRACTION
# =========================
def extract_color_features(rgb):
    return [
        np.mean(rgb[:,:,0]),
        np.mean(rgb[:,:,1]),
        np.mean(rgb[:,:,2]),
        np.std(rgb[:,:,0]),
        np.std(rgb[:,:,1]),
        np.std(rgb[:,:,2])
    ]

def extract_texture_features(gray):
    gray_uint8 = (gray * 255).astype(np.uint8)

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

def extract_all_features(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0

    feats = extract_color_features(rgb) + extract_texture_features(gray)
    feats = np.array(feats).reshape(1, -1)

    return scaler.transform(feats)

# =========================
# WEBCAM LOOP
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam tidak bisa dibuka")
    exit()

print("✅ Webcam aktif - tekan Q untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        features = extract_all_features(frame)

        pred = int(model.predict(features)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba))

        grade = GRADE_MAP.get(pred, "Unknown")

        label_text = grade
        if confidence is not None:
            label_text += f" ({confidence:.2f})"

        # warna berdasarkan grade
        color = (0,255,0) if grade == "Grade A" else \
                (0,255,255) if grade == "Grade B" else \
                (0,0,255)

        cv2.putText(
            frame,
            label_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    except Exception as e:
        cv2.putText(
            frame,
            "Processing error",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

    cv2.imshow("Manggis Webcam Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
