import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

best_rf = grid_search.best_estimator_
print(f"Best Model Score: {best_rf.score(X_test, y_test)}")

# 可視化結果
results = grid_search.cv_results_

# 創建參數組合的得分矩陣
n_estimators = param_grid['n_estimators']
max_depth = [
    str(d) if d is not None else 'None' for d in param_grid['max_depth']]
min_samples = param_grid['min_samples_split']

# 對每個 min_samples_split 值創建一個熱圖
for min_split in min_samples:
    scores = np.zeros((len(max_depth), len(n_estimators)))
    for i, depth in enumerate(param_grid['max_depth']):
        for j, n_est in enumerate(n_estimators):
            mask = ((results['param_max_depth'] == depth) &
                    (results['param_n_estimators'] == n_est) &
                    (results['param_min_samples_split'] == min_split))
            scores[i, j] = results['mean_test_score'][mask][0]

    plt.figure(figsize=(10, 6))
    sns.heatmap(scores, annot=True, fmt='.3f',
                xticklabels=n_estimators,
                yticklabels=max_depth,
                cmap='YlOrRd')
    plt.title(f'GridSearch Results (min_samples_split={min_split})')
    plt.xlabel('n_estimators')
    plt.ylabel('max_depth')
    plt.show()
