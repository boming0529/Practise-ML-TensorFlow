import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image


def recognize_digit(img):
    print("Type of img:", type(img))
    print("img keys:", list(img.keys()))
    model = load_model('model/inv_mnist_model_2025_04_04.h5')

    if not isinstance(img, dict) or 'composite' not in img:
        raise KeyError("Expected 'composite' key in sketchpad input, but not found.")
    
    img_array = np.array(img['composite'])
    print("Composite image shape:", img_array.shape)

    if img_array.shape[-1] == 4:
        alpha_channel = img_array[..., 3]
        rgb_channels = img_array[..., :3]  # extract RGB channel
        transparent_mask = alpha_channel < 10 
        rgb_channels[transparent_mask] = 0
    else:
        rgb_channels = img_array
    gray_array = np.mean(rgb_channels, axis=2)

    gray_pil = Image.fromarray(gray_array.astype(np.uint8))
    gray_pil = gray_pil.resize((28, 28), Image.Resampling.LANCZOS)
    gray_array = np.array(gray_pil).astype(np.float32) / 255.0  

    black_background = np.zeros((28, 28), dtype=np.float32)
    final_array = black_background.copy()
    gray_array = 1.0 - gray_array
    final_array = final_array+gray_array

    final_array = final_array.reshape(1, 784)
    
    prediction = model.predict(final_array).flatten()
    labels = list('0123456789')
    return { labels[i]: float(prediction[i]) for i in range(10) }

iface = gr.Interface(
    fn=recognize_digit, 
    title="My MNIST AI",
    description="plz input a number, i will recognize",
    inputs="sketchpad",
    outputs="label")

iface.launch()