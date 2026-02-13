import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = np.sort(15 * np.random.rand(500, 1), axis=0) - 5
y = (0.5 * X.ravel()**2 - 0.15 * X.ravel()**3 + 2 * X.ravel() +
     np.random.normal(0, 3, X.shape[0]))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def visualize_predictions(tree, X_train, X_test, y_train, y_test, min_samples_split):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    y_plot = tree.predict(X_plot)
    ax.scatter(X_train, y_train, c="blue", marker="o", s=30, alpha=0.3,
               label="Обучающие данные")
    ax.scatter(X_test, y_test, c="red", marker="^", s=50, alpha=0.8,
               edgecolors="black", linewidth=1, label="Тестовые данные")
    ax.plot(X_plot, y_plot, c="green", linewidth=3, label="Предсказания модели")
    ax.set_xlabel("Признак X")
    ax.set_ylabel("Целевая переменная y")
    ax.set_title(f"Предсказания модели (min_samples_split={min_samples_split})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
 
min_samples_split = 20

tree = DecisionTreeRegressor(min_samples_split=min_samples_split, random_state=42)
tree.fit(X_train, y_train)

y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"min_samples_split: {min_samples_split}")
print(f"MSE на обучении: {train_mse:.4f}")
print(f"MSE на тесте: {test_mse:.4f}")
print(f"Разность MSE (переобучение): {train_mse - test_mse:.4f}")
print(f"Количество узлов: {tree.tree_.node_count}")
print(f"Количество листьев: {tree.tree_.n_leaves}")
print(f"Максимальная глубина: {tree.tree_.max_depth}")

visualize_predictions(tree, X_train, X_test, y_train, y_test, min_samples_split)

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
plot_tree(tree, ax=ax, filled=True, rounded=True,
          feature_names=["Признак X"], fontsize=9)
ax.set_title(f"Структура дерева (min_samples_split={min_samples_split})")
plt.tight_layout()
plt.show()