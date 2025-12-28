import tkinter as tk
from tkinter import filedialog
import cv2
import joblib
import numpy as np
import paho.mqtt.client as mqtt

from PIL import Image, ImageTk
from skimage.feature import graycomatrix, graycoprops

# ======================================================
# MQTT
# ======================================================
MQTT_BROKER = "broker.hivemq.com"
MQTT_TOPIC  = "manggis/grade"

mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, 1883, 60)
mqtt_client.loop_start()

# ======================================================
# PILIH MODEL
# ======================================================
USE_SVM = True   # True = SVM | False = Random Forest

if USE_SVM:
    model = joblib.load("svm_model.pkl")
else:
    model = joblib.load("rf_model.pkl")

scaler = joblib.load("scaler.pkl")

IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.50

# ======================================================
# LABEL MAPPING
# ======================================================
GRADE_MAP = {
    1: "Grade C",
    2: "Grade C",
    3: "Grade B",
    4: "Grade B",
    5: "Grade A",
    6: "Grade A"
}

GRADE_TO_CMD = {
    "Grade A": "A",
    "Grade B": "B",
    "Grade C": "C"
}

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

    feats = extract_color_features(rgb) + extract_texture_features(gray)
    feats = np.array(feats).reshape(1, -1)

    return scaler.transform(feats)

# ======================================================
# PREDIKSI
# ======================================================
def predict_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    features = extract_all_features(file_path)

    pred_label = int(model.predict(features)[0])
    grade = GRADE_MAP.get(pred_label, "Tidak diketahui")

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = float(np.max(proba))

        if confidence < CONFIDENCE_THRESHOLD:
            result_label.config(
                text=f"Prediksi tidak yakin ({confidence:.2f})"
            )
            return

    # ===== KIRIM MQTT =====
    if grade in GRADE_TO_CMD:
        cmd = GRADE_TO_CMD[grade]
        mqtt_client.publish(MQTT_TOPIC, cmd)
        print("Publish MQTT:", cmd)

    # ===== UPDATE GUI =====
    if confidence:
        result_label.config(
            text=f"Hasil Prediksi: {grade} ({confidence:.2f})"
        )
    else:
        result_label.config(
            text=f"Hasil Prediksi: {grade}"
        )

    img_show = Image.open(file_path).resize((250, 250))
    img_tk = ImageTk.PhotoImage(img_show)
    panel.config(image=img_tk)
    panel.image = img_tk

# ======================================================
# GUI
# ======================================================
root = tk.Tk()
root.title("Klasifikasi Kualitas Manggis (MQTT)")

btn = tk.Button(
    root,
    text="Upload Gambar",
    command=predict_image,
    font=("Arial", 12)
)
btn.pack(pady=10)

result_label = tk.Label(
    root,
    text="Hasil Prediksi:",
    font=("Arial", 12)
)
result_label.pack(pady=5)

panel = tk.Label(root)
panel.pack(pady=10)

root.mainloop()
