import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

# print(iris)
print(iris.keys())
print(iris['DESCR'])

x_data = iris.data
y_target = iris.target

print('x shape, y shape (iris data): ', x_data.shape, y_target.shape)
print(x_data[0], y_target[0])

x_petal_data = x_data[:, 2:4]
print(x_petal_data)
print('x shape, y shape (only petal data): ',
      x_petal_data.shape, y_target.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x_petal_data,
    y_target,
    test_size=0.2,
    random_state=10
)

print('x shape, y shape (after split): ', x_train.shape, y_train.shape)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.show()
