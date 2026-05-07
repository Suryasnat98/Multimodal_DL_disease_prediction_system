# diagnostic.py — run this standalone, outside Flask
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("model/skin_cancer_cnn.h5")

def run_diagnostic(image_path, label):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # test with RGB
    img_bgr = img  # test with BGR too
    
    for name, arr in [("BGR (current)", img_bgr), ("RGB (fix)", img_rgb)]:
        resized = cv2.resize(arr, (128, 128))
        tensor = np.expand_dims(resized / 255.0, axis=0)
        pred = model.predict(tensor, verbose=0)[0][0]
        decision = "Malignant" if pred > 0.5 else "Benign"
        print(f"  [{name}] Raw sigmoid: {pred:.4f} → {decision} (True: {label})")

# Test with a few known images
print("=== Diagnostic Results ===")
run_diagnostic(r"C:\Users\LENOVO\OneDrive\Desktop\Gradcam Working\Hackathon\project\begnin.jpg", "Benign")
run_diagnostic(r"C:\Users\LENOVO\OneDrive\Desktop\Gradcam Working\Hackathon\project\Malignant.jpg", "Malignant")

# Also test with a solid gray image — model should be uncertain (~0.5)
gray = np.full((128, 128, 3), 128, dtype=np.uint8)
tensor = np.expand_dims(gray / 255.0, axis=0)
pred = model.predict(tensor, verbose=0)[0][0]
print(f"\n[Gray image] Raw sigmoid: {pred:.4f} (expected ~0.5 if unbiased)")

# Test with pure black and pure white
for name, val in [("Black", 0), ("White", 255)]:
    img = np.full((128, 128, 3), val, dtype=np.uint8)
    tensor = np.expand_dims(img / 255.0, axis=0)
    pred = model.predict(tensor, verbose=0)[0][0]
    print(f"[{name} image] Raw sigmoid: {pred:.4f}")

