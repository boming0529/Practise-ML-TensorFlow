# import stranded library
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
# Datasets
from tensorflow.keras.utils import to_categorical # one-hot encoding
from tensorflow.keras.datasets import fashion_mnist

# load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
model = load_model('./model/fashion_mnist_model_2025_04_04.h5')
x_test = x_test.reshape(10000, 28*28).astype('float32') / 255
y_test = to_categorical(y_test, 10)

# predict
n = 1200
inp = x_test[n].reshape(1, 784)
predict_y = model.predict(inp)
print(y_test[n])
print(np.argmax(predict_y, axis=-1))
plt.imshow(x_test[n].reshape(28, 28), cmap='Greys')
plt.show()