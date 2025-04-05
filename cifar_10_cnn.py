# standard library
import matplotlib.pyplot as plt
import numpy as np
# tf.keras
from tensorflow.keras.datasets import cifar10 # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten  # type: ignore
from tensorflow.keras.optimizers import SGD, Adam # type: ignore

# Data Engineering
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# check and show how many training data
print(f'number of train data shape: {x_train.shape}')
classification = np.unique(y_train)
print(f'number of classification : {classification}')

check_class = 6 # frog
x_cc = x_train[np.where(y_train == 6)[0]]
print(f'number of frog data shape: {x_cc.shape}')

plt.subplot(2, 1, 1)
plt.imshow(x_cc[0])
plt.axis('off')

plt.subplot(2, 1, 2)
plt.imshow(x_cc[1])
plt.axis('off')
plt.show()

frog_img = x_cc[0]
red_channel = frog_img[:, :, 0]
green_channel = frog_img[:, :, 1]
blue_channel = frog_img[:, :, 2]

plt.subplot(1, 4, 1)
plt.imshow(frog_img)
plt.axis('off')

plt.subplot(1, 4, 2)
red_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
red_rgb[:, :, 0] = red_channel
plt.imshow(red_rgb)
plt.axis('off')

plt.subplot(1, 4, 3)
green_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
green_rgb[:, :, 1] = green_channel
plt.imshow(green_rgb)
plt.axis('off')

plt.subplot(1, 4, 4)
blue_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
blue_rgb[:, :, 2] = blue_channel
plt.imshow(blue_rgb)
plt.axis('off')
plt.show()

# Feature Engineering
x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

y_train = to_categorical(y_train, len(classification))

# check and show how many training data
print(f'number of train data shape: {x_train.shape}')

# Training Model
model = Sequential()
model.add(Conv2D(16, (5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.summary()
model.add(MaxPooling2D(pool_size=(2, 2)))
model.summary()
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.summary()
model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.summary()
model.add(Flatten())
model.summary()
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=100, epochs=20)

