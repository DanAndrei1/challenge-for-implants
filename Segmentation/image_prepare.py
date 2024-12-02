from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk
import cv2
from CCA_Analysis import *

def get_pre_image(image):
    # image = PIL.Image
    # Process the first image
    img = image.convert('L')  # Convert to grayscale
    size = img.size
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    images = np.expand_dims(np.asarray(img), axis=0)  # Add batch dimension
    images = np.expand_dims(images, axis=-1)
    return images, size


def select_file():
    root = tk.Tk()
    root.withdraw()  # Ascunde fereastra principală
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("All Files", "*.*")]
    )
    root.destroy()  # Închide complet fereastra tkinter
    return file_path

def convert_pil_to_opencv(pil_image):
    """Convertește o imagine PIL în format OpenCV fără alte modificări."""
    return np.array(pil_image)


def convert_to_rgb(predicted_mask):
    """
    Convertește o matrice 2D normalizată (valori între 0 și 1) într-o imagine RGB.
    """
    # Scalează valorile la intervalul [0, 255]
    scaled = (predicted_mask * 255).astype(np.uint8)

    # Extinde matricea pentru a crea 3 canale (R, G, B)
    rgb_image = cv2.merge((scaled, scaled, scaled))  # R, G, B au aceeași valoare

    return rgb_image

if __name__ == "__main__":
    model_path = "./model.keras"
    model = load_model(model_path)
    # Creează fereastra pentru dialogul de fișiere
    root = tk.Tk()
    root.withdraw()  # Ascunde fereastra principală
    # Deschide dialogul de fișiere
    file_path = select_file()

    if not file_path:
        print("No file selected.")
        exit()

    print(f"Selected file: {file_path}")

    image = Image.open(file_path)
    pre_img, img_size = get_pre_image(image)
    X = np.float32(pre_img / 255)

    print(X)
    print(img_size)

    pred_img = model.predict(X)
    print("Pred img")
    print(pred_img)

    predict = pred_img[0, :, :, 0]

    print("Predict")
    print(predict)

    plt.figure(figsize=(20, 10))
    plt.title("Predict Mask", fontsize=40)
    plt.imshow(predict)
    plt.show()


    # Plotting - RESULT Example with CCA_Analysis
    img = convert_pil_to_opencv(image)  # original img 107.png

    # load image (mask was saved by matplotlib.pyplot)
    #predicted = cv2.imread("./predict_2.png")
    predicted = convert_to_rgb(predict)

    print("Predicted**")
    print(predicted)

    # Redimensionăm masca prezisă la dimensiunea imaginii originale
    predicted = cv2.resize(predicted, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    #predicted = (predicted * 255).astype(np.uint8)  # Convertim masca în format 8-bit pentru OpenCV

    cca_result, teeth_count = CCA_Analysis(img, predicted, 3, 2)

    print(teeth_count)

    cv2.imshow('CCA Result', cca_result)
    cv2.waitKey(0)