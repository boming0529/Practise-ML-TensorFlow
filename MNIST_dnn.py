# import stranded library
import numpy as np
import matplotlib.pyplot as plt
# import TensorFlow
from tensorflow.keras.utils import to_categorical # one-hot encoding
from tensorflow.keras.models import Sequential # model
from tensorflow.keras.layers import Dense      # dense neural network hidden layer
from tensorflow.keras.optimizers import SGD    # gradient descent
# Datasets
from tensorflow.keras.datasets import mnist

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

n = 5000
# x_train[n]
print(y_train[n])
plt.imshow(x_train[n], cmap='Greys')
plt.show()

# prepare data and normalization
x_train = x_train.reshape(60000, 28*28).astype('float32') / 255
x_test = x_test.reshape(10000, 28*28).astype('float32') / 255
# convert to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# check
print(y_train[n])

model = Sequential() # container
# design hyperparameter, number of DNN neuron size, number of layer 
# first hidden layer 
# model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(128, activation='relu', input_dim=784))
# second hidden layer
model.add(Dense(128, activation='relu'))
# third hidden layer
model.add(Dense(64, activation='relu'))
# output layer 
model.add(Dense(10, activation='softmax'))

# compile
# using mse
# model.compile(loss='mse', 
#               optimizer=SGD(learning_rate=0.05),
#               metrics=['accuracy'])
# using categorical cross entropy
model.compile(optimizer=SGD(learning_rate=0.01), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# summary
model.summary()

# training model
history = model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2)

# evaluation model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"test accuracy: {test_accuracy:.4f}")

# plot
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()

# save model
model.save('./model/mnist_model_2025_04_04.h5')

# predict
n = 2000
inp = x_test[n].reshape(1, 784)
predict_y = model.predict(inp)
print(y_test[n])
print(np.argmax(predict_y, axis=-1))
plt.imshow(x_test[n].reshape(28, 28), cmap='Greys')
plt.show()