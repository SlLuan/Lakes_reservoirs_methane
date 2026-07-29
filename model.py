import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed


file_path = 'data.csv'  
data = pd.read_csv(file_path)

selected_features = ['DO', 'Cond', 'PI', 'DOC', 'pH']  
X = data[selected_features]
y = data['CH4_evasio']

predictions = {}

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

def train_and_predict(i, X, y, params, num_boost_round):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    test_indices = X_test.index.tolist()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    y_pred = model.predict(dtest)
    return test_indices, y_pred.tolist()

results = Parallel(n_jobs=30)(
    delayed(train_and_predict)(i, X, y, params, num_boost_round) for i in range(100)
)

for test_indices, y_pred in results:
    for idx, pred in zip(test_indices, y_pred):
        if idx in predictions:
            predictions[idx].append(pred)
        else:
            predictions[idx] = [pred]

average_predictions = np.array([np.mean(predictions[idx]) if idx in predictions else np.nan for idx in range(len(y))])
true_values = y.to_numpy()

valid_indices = ~np.isnan(average_predictions)
average_predictions = average_predictions[valid_indices]
true_values = true_values[valid_indices]

final_r2 = r2_score(true_values, average_predictions)
final_mse = mean_squared_error(true_values, average_predictions)
final_rmse = np.sqrt(final_mse)

print(f"Final R²: {final_r2:.2f}")
print(f"Final MSE: {final_mse:.2f}")
print(f"Final RMSE: {final_rmse:.2f}")

regression_model = LinearRegression().fit(true_values.reshape(-1, 1), average_predictions)
fit_line_x = np.array([true_values.min(), true_values.max()]).reshape(-1, 1)
fit_line_y = regression_model.predict(fit_line_x)

slope = regression_model.coef_[0]
intercept = regression_model.intercept_
fit_equation = f'y = {slope:.2f}x + {intercept:.2f}'

plt.show()

def train_and_predict(i, X, y, params, num_boost_round):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    test_indices = X_test.index.tolist()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    y_pred = model.predict(dtest)
    return model, test_indices, y_pred.tolist()

results = Parallel(n_jobs=30)(
    delayed(train_and_predict)(i, X, y, params, num_boost_round) for i in range(100)
)

for model, test_indices, y_pred in results:
    for idx, pred in zip(test_indices, y_pred):
        if idx in predictions:
            predictions[idx].append(pred)
        else:
            predictions[idx] = [pred]

average_predictions = np.array([np.mean(predictions[idx]) if idx in predictions else np.nan for idx in range(len(y))])
true_values = y.to_numpy()

valid_indices = ~np.isnan(average_predictions)
average_predictions = average_predictions[valid_indices]
true_values = true_values[valid_indices]

final_r2 = r2_score(true_values, average_predictions)
final_mse = mean_squared_error(true_values, average_predictions)
final_rmse = np.sqrt(final_mse)

print(f"Final R²: {final_r2:.2f}")
print(f"Final MSE: {final_mse:.2f}")
print(f"Final RMSE: {final_rmse:.2f}")


