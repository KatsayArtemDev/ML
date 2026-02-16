import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X, y = make_classification(n_samples=15000, n_features=25, n_classes=2, random_state=42, n_informative=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

n_trees = 23
max_depth = 5
print(X_train.shape)
print(X_test.shape)

trees = []
for _ in range(n_trees):
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_bootstrap = X_train[indices]
    y_bootstrap = y_train[indices]
    
    tree = DecisionTreeClassifier(max_depth = 5, random_state = 42)
    tree.fit(X_bootstrap, y_bootstrap)
    trees.append(tree)

print(trees)

def predict_ensemble(X):
    predictions = np.array([tree.predict(X) for tree in trees])
    return np.round(predictions.mean(axis=0)).astype(int) # Для двух классов усреднение и округление работает как голосование (для многоклассовых задач нужно использовать argmax)

y_pred = predict_ensemble(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность ансамбля: {accuracy:.4f}")

report = classification_report(y_test, y_pred)
print(report)

solo_tree = DecisionTreeClassifier(max_depth = 5, random_state = 42)
solo_tree.fit(X_train, y_train)

y_pred_solo_tree = solo_tree.predict(X_test)

solo_tree_accuracy = accuracy_score(y_test, y_pred_solo_tree)
print(f"Точность одного дерева: {solo_tree_accuracy:.4f}")

solo_tree_report = classification_report(y_test, y_pred_solo_tree)
print(solo_tree_report)