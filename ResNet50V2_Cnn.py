import os
import numpy as np
import matplotlib.pyplot as plt

from keras._tf_keras.keras.applications import ResNet50V2
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Flatten
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array


# wget --no-check-certificate https://github.com/yenlung/Deep-Learning-Basics/raw/master/images/myna.zip -O myna.zip

base_dir = './mynah_bird/myna/'
mynah_folders = ['common_myna', 'crested_myna', 'javan_myna']

this_dir = base_dir + mynah_folders[0]
print(os.listdir(this_dir))

data = []
target = []
for i in range(3):
    this_dir = base_dir + mynah_folders[i]
    myna_fnames = os.listdir(this_dir)
    for myna in myna_fnames:
        img_path = this_dir + '/' + myna
        img = load_img(img_path, target_size=(256, 256))
        x = np.array(img)

        data.append(x)
        target.append(i)

data = np.array(data)
print(data.shape)

n = 22
plt.imshow(data[n]/255)
plt.axis('off')
plt.show()


x_train = preprocess_input(data)
print(f"x_train min/max: {x_train[n].min():.4f}, {x_train[n].max():.4f}")
plt.imshow(x_train[n])
plt.axis('off')
plt.show()

# one-hot encoding
y_train = to_categorical(target, 3)
print(y_train[22])

# remove last layer, and using global average pooling.
resnet = ResNet50V2(include_top=False, pooling='avg')

model = Sequential()
model.add(resnet)
model.add(Dense(3, activation='softmax'))
