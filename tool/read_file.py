import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_image(filepath, target_size=(224, 224)):
    try:
        image = Image.open(filepath).convert('RGB')
        image = image.resize(target_size)
        return np.array(image)
    except FileNotFoundError:
        print(f"Error: {filepath} not found")
        return None
    except Exception as e:
        print(f"Error loading image {filepath}: {str(e)}")
        return None
