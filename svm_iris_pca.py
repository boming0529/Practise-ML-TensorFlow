import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA
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

# dimensionality reduction 2D
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_data)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
print('x shape, y shape (after pca): ', x_train_pca.shape, y_train.shape)

# training model
clf_pca = SVC()
clf_pca.fit(x_train_pca, y_train)

# plot training data
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Training Data (PCA)')
plt.show()
# sns.scatterplot(x=x_train_pca[:, 0], y=x_train_pca[:, 1], hue=y_train)

# plot testing data
y_predict_pca = clf_pca.predict(x_test_pca)
plt.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=y_predict_pca)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Predicted Test Data (PCA)')
plt.show()

# plot boundary
x_min, x_max = x_pca[:, 0].min() - 1, x_pca[:, 0].max() + 1
y_min, y_max = x_pca[:, 1].min() - 1, x_pca[:, 1].max() + 1
x1, x2 = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
z = clf_pca.predict(np.c_[x1.ravel(), x2.ravel()])
z = z.reshape(x1.shape)

plt.contourf(x1, x2, z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y_target, edgecolors='k')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Decision Boundary (PCA)')
plt.show()
