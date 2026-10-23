import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('FP_fake_image_detector_modelV2.h5')

# Preprocessing and classification
def classify_uploaded_image(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction)

    if confidence < 0.5:
        label = "AI-Generated"
        confidence = (1 - confidence) * 100
    else:
        label = "Real"
        confidence = confidence * 100

    return label, round(confidence, 2)

# Upload handler
def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Show image
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

    # Predict and display result
    label, confidence = classify_uploaded_image(file_path)

    # Determine result color based on certainty
    if 45 <= confidence <= 55:
        color = "orange"
        message = f"Prediction: {label} ({confidence:.2f}%)"
    elif label == "AI-Generated":
        color = "red"
        message = f"Prediction: {label} ({confidence:.2f}%)"
    else:
        color = "green"
        message = f"Prediction: {label} ({confidence:.2f}%)"

    result_label.config(text=message, fg=color)

    # Main window setup
root = tk.Tk()
root.title("Fake Image Detector")
root.geometry("420x550")
root.configure(bg="#f0f0f0")
root.resizable(False, False)  # Make window non-resizable

# Title label
title_label = tk.Label(root, text="Fake Image Detector", font=("Helvetica", 18, "bold"), bg="#f0f0f0", fg="#333")
title_label.pack(pady=20)

# Frame for image preview
image_frame = tk.Frame(root, width=260, height=260, bg="white", highlightbackground="gray", highlightthickness=1)
image_frame.pack(pady=10)
image_label = tk.Label(image_frame, bg="white")
image_label.pack(padx=5, pady=5)

# Upload button
upload_btn = tk.Button(root, text="Upload Image", command=upload_image,
                       font=("Helvetica", 12), bg="#4a90e2", fg="white", padx=20, pady=8, bd=0)
upload_btn.pack(pady=15)

# Result display
result_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"), bg="#f0f0f0")
result_label.pack(pady=20)

# Run the GUI loop
root.mainloop()
