import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # support vector

iris = load_iris()

x_data = iris.data
y_target = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_target,
    test_size=0.2,
    random_state=10
)

print('x shape, y shape (after split): ', x_train.shape, y_train.shape)

# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
# plt.show()

clf = SVC()
clf.fit(x_train, y_train)  # x_train shape (120, 4)

# Pair Plot
df = pd.DataFrame(
    x_data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
df['Target'] = y_target

sns.pairplot(df, hue='Target', vars=[
             'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
plt.show()


def show_feature(x_data, y_target, x_train, x_test, y_train, y_predict, clf, type):
    if type == 'Petal':
        x_subset = x_data[:, 2:4]  # only show petal length, width
        x_train_subset = x_train[:, 2:4]
        x_test_subset = x_test[:, 2:4]
        x_range = np.arange(0, 7, 0.02)
        y_range = np.arange(0, 3, 0.02)
    else:
        x_subset = x_data[:, 0:2]  # only show sepal length, width
        x_train_subset = x_train[:, 0:2]
        x_test_subset = x_test[:, 0:2]
        x_range = np.arange(3, 9, 0.02)
        y_range = np.arange(1, 5, 0.02)

    # plot training data
    plt.scatter(x_train_subset[:, 0], x_train_subset[:, 1], c=y_train)
    plt.xlabel(f'{type} Length')
    plt.ylabel(f'{type} Width')
    plt.title(f'Training Data ({type} Features)')
    plt.show()

    # plot testing data
    plt.scatter(x_test_subset[:, 0], x_test_subset[:, 1], c=y_predict)
    plt.xlabel(f'{type} Length')
    plt.ylabel(f'{type} Width')
    plt.title(f'Predicted Test Data ({type} Features)')
    plt.show()

    # plot boundary
    x1, x2 = np.meshgrid(x_range, y_range)
    # all feature mean value
    mean_features = np.mean(x_data, axis=0)
    if type == 'Petal':
        grid_points = np.c_[
            np.ones(x1.ravel().shape[0]) * mean_features[0],  # sepal length
            np.ones(x1.ravel().shape[0]) * mean_features[1],  # sepal width
            x1.ravel(),  # petal length
            x2.ravel()   # petal width
        ]
    else:
        grid_points = np.c_[
            x1.ravel(),  # sepal length
            x2.ravel(),  # sepal width
            np.ones(x1.ravel().shape[0]) * mean_features[2],  # petal length
            np.ones(x1.ravel().shape[0]) * mean_features[3]   # petal width
        ]

    z = clf.predict(grid_points)
    z = z.reshape(x1.shape)

    plt.contourf(x1, x2, z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(x_subset[:, 0], x_subset[:, 1], c=y_target, edgecolors='k')
    plt.xlabel(f'{type} Length')
    plt.ylabel(f'{type} Width')
    plt.title(f'Decision Boundary ({type} Features)')
    plt.show()


y_predict = clf.predict(x_test)
show_feature(x_data, y_target, x_train, x_test,
             y_train, y_predict, clf, 'Petal')
show_feature(x_data, y_target, x_train, x_test,
             y_train, y_predict, clf, 'Sepal')

# Error Plot
correct = y_test == y_predict
plt.scatter(x_test[~correct, 2], x_test[~correct, 3],
            c='red', label='Incorrect', marker='x')
plt.scatter(x_test[correct, 2], x_test[correct, 3],
            c='green', label='Correct', marker='o')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Prediction Results')
plt.legend()
plt.show()


plt.scatter(x_test[~correct, 0], x_test[~correct, 1],
            c='red', label='Incorrect', marker='x')
plt.scatter(x_test[correct, 0], x_test[correct, 1],
            c='green', label='Correct', marker='o')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Prediction Results')
plt.legend()
plt.show()
