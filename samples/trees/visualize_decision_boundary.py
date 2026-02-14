import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    flip_y=0.2,
    class_sep=1.2,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=412,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def visualize_decision_boundary(tree, X_train, X_test, y_train, y_test, max_leaf_nodes):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="RdYlBu")
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
               c="red", marker="o", s=30, alpha=0.7, label="Класс 0 (обучение)")
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
               c="blue", marker="o", s=30, alpha=0.7, label="Класс 1 (обучение)")
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
               c="red", marker="^", s=60, alpha=0.9, edgecolors="black",
               linewidth=1, label="Класс 0 (тест)")
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
               c="blue", marker="v", s=60, alpha=0.9, edgecolors="black",
               linewidth=1, label="Класс 1 (тест)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Граница решения (max_leaf_nodes={max_leaf_nodes})")
    ax.legend()
    ax.set_xlabel("Признак 1")
    ax.set_ylabel("Признак 2")
    plt.tight_layout()
    plt.show()
 
max_leaf_nodes = 2

tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=42)
tree.fit(X_train, y_train)

y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Максимальное количество листьев: {max_leaf_nodes}")
print(f"Фактическое количество листьев: {tree.tree_.n_leaves}")
print(f"Глубина дерева: {tree.tree_.max_depth}")
print(f"Точность на обучении: {train_accuracy:.3f}")
print(f"Точность на тесте: {test_accuracy:.3f}")
print(f"Разность (переобучение): {train_accuracy - test_accuracy:.3f}")
print(f"Количество узлов: {tree.tree_.node_count}")

visualize_decision_boundary(tree, X_train, X_test, y_train, y_test, max_leaf_nodes)

fig, ax = plt.subplots(1, 1, figsize=(20, 16))
plot_tree(tree, ax=ax, filled=True, rounded=True,
          feature_names=["Признак 1", "Признак 2"],
          class_names=["Класс 0", "Класс 1"], fontsize=8)
ax.set_title(f"Структура дерева (листьев: {tree.tree_.n_leaves})")
plt.tight_layout()
plt.show()