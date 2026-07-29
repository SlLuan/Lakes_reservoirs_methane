import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from itertools import combinations

# 加载数据
file_path = 'data.csv' 
data = pd.read_csv(file_path)


selected_features = ['pH', 'DO', 'PI', 'TN', 'TP', 'Cond', 'Tur', 'DOC']
X = data[selected_features]
y = data['CH4_evasio']
fixed_features = ['DO', 'DOC', 'pH']
remaining_features = [feature for feature in selected_features if feature not in fixed_features]

metrics_combinations = {}

params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 5,
    'alpha': 0.1,
    'lambda': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'seed': 42
}
num_boost_round = 800

# 定义训练和预测函数
def train_and_predict(i, X, y, params, num_boost_round):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    test_indices = X_test.index.tolist()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    y_pred = model.predict(dtest)
    return test_indices, y_pred.tolist()

# 测试每种特征组合
for comb_len in range(1, len(remaining_features) + 1):  # 从1到剩余特征的数量
    for comb in combinations(remaining_features, comb_len):
        current_features = fixed_features + list(comb)
        X_current = X[current_features]
        
        # 并行训练并预测
        results = Parallel(n_jobs=30)(
            delayed(train_and_predict)(i, X_current, y, params, num_boost_round) for i in range(100)
        )
        
        # 处理结果，累积每个数据点的预测值
        predictions = {}
        for test_indices, y_pred in results:
            for idx, pred in zip(test_indices, y_pred):
                if idx in predictions:
                    predictions[idx].append(pred)
                else:
                    predictions[idx] = [pred]

        # 计算每个数据点预测值的平均值
        average_predictions = np.array([np.mean(predictions[idx]) if idx in predictions else np.nan for idx in range(len(y))])
        true_values = y.to_numpy()

        # 过滤掉未预测的数据点（如果有）
        valid_indices = ~np.isnan(average_predictions)
        average_predictions = average_predictions[valid_indices]
        true_values = true_values[valid_indices]

        # 计算 R², RMSE, MAE
        r2 = r2_score(true_values, average_predictions)
        rmse = np.sqrt(mean_squared_error(true_values, average_predictions))
        mae = mean_absolute_error(true_values, average_predictions)
        
        # 存储每个组合的评估指标
        metrics_combinations[tuple(comb)] = {'R²': r2, 'RMSE': rmse, 'MAE': mae}

# 打印每个组合的评估指标
for comb, metrics in metrics_combinations.items():
    print(f"Combination: {comb}")
    print(f"  R²: {metrics['R²']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}")
    print()

# 找到最佳的特征组合
best_combination = max(metrics_combinations, key=lambda x: metrics_combinations[x]['R²'])
best_metrics = metrics_combinations[best_combination]

print(f"Best feature combination: {best_combination} with R² = {best_metrics['R²']:.2f}, RMSE = {best_metrics['RMSE']:.2f}, MAE = {best_metrics['MAE']:.2f}")
