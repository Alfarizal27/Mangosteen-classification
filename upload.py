# ============================================================
# predict_image.py
# ============================================================
import cv2
import numpy as np
import joblib

model = joblib.load("decision_tree_manggis.pkl")

def predict_image(path):
    img = cv2.imread(path)
    if img is None:
        print("Gambar tidak ditemukan:", path)
        return

    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten().reshape(1, -1)

    pred = model.predict(img)[0]
    print("Prediksi kelas:", pred)


if __name__ == "__main__":
    path = input("Masukkan path gambar: ")
    predict_image(path)
