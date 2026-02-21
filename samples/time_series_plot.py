# Временные данные представлены в X, y

# --- 1. Настройка TimeSeriesSplit ---
tscv = TimeSeriesSplit(n_splits=5)
print(f'Инициализирована кросс-валидация TimeSeriesSplit с n_splits={tscv.n_splits}\n')

# --- 2. Итерация по фолдам со cлучайным лесом ---
mse_scores = []
fig, axes = plt.subplots(tscv.n_splits + 1, 1, figsize=(14, 5 * (tscv.n_splits + 1)))

# Построение всего ряда для контекста
axes[0].plot(df.index, df['Value'], label='Весь временной ряд', color='gray', alpha=0.6)
axes[0].set_title('Общий вид временного ряда')
axes[0].set_ylabel('Значение')
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.5)

for fold, (train_index, valid_index) in enumerate(tscv.split(X)):
    
    # 2.1. Разделение данных для текущего фолда
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    train_dates = df.index[train_index]
    valid_dates = df.index[valid_index]

    # 2.2. Обучение модели: использование RandomForestRegressor
    # n_estimators=100 - количество деревьев, random_state для воспроизводимости
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=100)
    model.fit(X_train, y_train)

    # 2.3. Прогнозирование
    y_pred = model.predict(X_valid)

    # 2.4. Оценка (MSE)
    mse = mean_squared_error(y_valid, y_pred)
    mse_scores.append(mse)

    print(f'Фолд {fold+1}: Размер Train = {len(X_train)}, MSE (Random Forest) = {mse:.4f}')

    # 2.5. Визуализация текущего фолда
    ax = axes[fold + 1]
    ax.plot(train_dates, y_train, label='Обучающая выборка (Train)', color='#1f77b4')
    ax.plot(valid_dates, y_valid, label='Valid выборка (Valid/Actual)', color='#ff7f0e')
    ax.plot(valid_dates, y_pred, label='Прогноз (Random Forest)', color='green', linestyle='--')
    
    ax.axvline(x=valid_dates[0], color='red', linestyle=':', alpha=0.7, label='Точка разбиения')
    ax.set_title(f'Фолд {fold+1}: Прогноз на период {valid_dates[0].date()} - {valid_dates[-1].date()} (MSE: {mse:.4f})')
    ax.set_ylabel('Значение')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.5)

axes[-1].set_xlabel('Дата')
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()

# --- 3. Визуализация ошибок MSE по фолдам ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(mse_scores)+1), mse_scores, marker='o', color='darkgreen', linewidth=2)
plt.title('Изменение ошибки MSE по фолдам (Random Forest)', fontsize=14)
plt.xlabel('Номер фолда (прогресс во времени)', fontsize=12)
plt.ylabel('Среднеквадратичная ошибка (MSE)', fontsize=12)
plt.xticks(range(1, len(mse_scores)+1))
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print(f'\nСредняя MSE Random Forest по всем фолдам: {np.mean(mse_scores):.4f}')