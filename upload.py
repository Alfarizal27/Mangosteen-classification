import tkinter as tk
from tkinter import filedialog
import cv2
import joblib
import numpy as np
from PIL import Image, ImageTk

model = joblib.load("decision_tree_manggis.pkl")

def predict_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if not file_path:
        return

    img = cv2.imread(file_path)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.flatten().reshape(1, -1)


    pred = model.predict(img)[0]
    result_label.config(text=f"Hasil Prediksi: Manggis {pred}")

    show_img = Image.open(file_path)
    show_img = show_img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(show_img)
    panel.config(image=img_tk)
    panel.image = img_tk

# GUI
root = tk.Tk()
root.title("Klasifikasi Manggis")

btn = tk.Button(root, text="Upload Gambar", command=predict_image)
btn.pack(pady=10)

result_label = tk.Label(root, text="Hasil Prediksi:")
result_label.pack()

panel = tk.Label(root)
panel.pack()

root.mainloop()
