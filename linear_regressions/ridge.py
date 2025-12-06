# Импорт библиотек
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

def scale_data(X_train, X_val, X_test, method='standard'):
    if method == 'standard':
        mean = X_train.mean()
        std = X_train.std()
        X_train_scaled = (X_train - mean) / std
        X_val_scaled = (X_val - mean) / std
        X_test_scaled = (X_test - mean) / std
    elif method == 'minmax':
        min_val = X_train.min()
        max_val = X_train.max()
        X_train_scaled = (X_train - min_val) / (max_val - min_val)
        X_val_scaled = (X_val - min_val) / (max_val - min_val)
        X_test_scaled = (X_test - min_val) / (max_val - min_val)
    else:
        raise ValueError("Неверный метод масштабирования. Используйте 'standard' или 'minmax'.")
    return X_train_scaled, X_val_scaled, X_test_scaled

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'RMSE': round(rmse, 2),
        'MAPE': round(mape * 100, 2),
        'R2': round(r2, 3)
    }

def compare_metrics(train_metrics, val_metrics):
    for metric in train_metrics:
        change = (val_metrics[metric] - train_metrics[metric]) / train_metrics[metric] * 100
        direction = "изменилась" if metric != 'R2' else "изменился"
        print(f"{metric} на val {direction} по сравнению с {metric} на train на {change:.2f}%")

X, y = make_regression(n_samples=200,
                       n_features=5,
                       noise=15,
                       random_state=42)

X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y = pd.Series(y, name='target')

X.insert(0, 'record_id', range(len(X)))

df = pd.concat([X, y], axis=1)

df = df.drop(columns='record_id')

X = df.drop(columns='target')
y = df['target']

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25,
    shuffle=True,
    random_state=42
)

X_train_scaled, X_val_scaled, X_test_scaled = scale_data(X_train, X_val, X_test, method='standard')

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

y_train_pred_ridge = ridge.predict(X_train_scaled)
y_val_pred_ridge = ridge.predict(X_val_scaled)
y_test_pred_ridge = ridge.predict(X_test_scaled)

ridge_train_metrics = calculate_metrics(y_train, y_train_pred_ridge)
ridge_val_metrics = calculate_metrics(y_val, y_val_pred_ridge)
ridge_test_metrics = calculate_metrics(y_test, y_test_pred_ridge)

print("Метрики на train:", ridge_train_metrics)
print("Метрики Ridge на val:", ridge_val_metrics)
print("Метрики на test:", ridge_test_metrics)

compare_metrics(ridge_train_metrics, ridge_val_metrics)