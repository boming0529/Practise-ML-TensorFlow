import gradio as gr
import numpy as np
import time
from PIL import Image
from tensorflow.keras.models import load_model  # type: ignore

def sleep(im):
    time.sleep(5)
    return [im["background"], im["layers"][0], im["layers"][1], im["composite"]]

def predict(img):
    img_array = np.array(img['composite'])
    # extract alpha channel
    if img_array.shape[-1] == 4:
        alpha_channel = img_array[..., 3]  # (height, width)
    else:
        alpha_channel = None  # if not alpha channel

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
    gray_array = 1.0 - gray_array

    if alpha_channel is not None:
        alpha_pil = Image.fromarray(alpha_channel)
        alpha_pil = alpha_pil.resize((28, 28), Image.Resampling.LANCZOS)
        alpha_array = np.array(alpha_pil).astype(np.float32) / 255.0  
    else:
        final_array = np.ones((28, 28), dtype=np.float32)

    black_background = np.zeros((28, 28), dtype=np.float32)
    final_array = black_background.copy()
    # gray_array = 1.0 - gray_array
    final_array = final_array + gray_array
    
    rgba_array = np.zeros((28, 28, 4), dtype=np.float32)
    print("Gray shape:", final_array.shape)
    

    rgba_array[..., 0] = final_array  # R channel
    rgba_array[..., 1] = final_array  # G channel
    rgba_array[..., 2] = final_array  # B channel
    rgba_array[..., 3] = alpha_array  # Alpha channel

    print("Output image shape:", rgba_array.shape)
    print("Output pixel value range:", rgba_array.min(), rgba_array.max())

    final_array = final_array.reshape(1, 784)
    model = load_model('./model/inv_mnist_model_2025_04_04.h5')
    prediction = model.predict(final_array).flatten()
    labels = list('0123456789')
    print({ labels[i]: float(prediction[i]) for i in range(10) }) 

    return rgba_array


with gr.Blocks() as demo:
    with gr.Row():
        im = gr.ImageEditor(
            type="numpy",
            crop_size="1:1",
        )
        im_preview = gr.Image()
    n_upload = gr.Number(0, label="Number of upload events", step=1)
    n_change = gr.Number(0, label="Number of change events", step=1)
    n_input = gr.Number(0, label="Number of input events", step=1)

    im.upload(lambda x: x + 1, outputs=n_upload, inputs=n_upload)
    im.change(lambda x: x + 1, outputs=n_change, inputs=n_change)
    im.input(lambda x: x + 1, outputs=n_input, inputs=n_input)
    im.change(predict, outputs=im_preview, inputs=im, show_progress="hidden")

if __name__ == "__main__":
    demo.launch()