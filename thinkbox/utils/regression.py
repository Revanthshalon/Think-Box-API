import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFE


def linear_regression(df, significant_cols, target, cat_cols, num_cols):
    ss = StandardScaler()
    ohe = OneHotEncoder(drop='first', sparse=False)
    X = df[significant_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    train_data = np.c_[ohe.fit_transform(X_train[cat_cols]), ss.fit_transform(X_train[num_cols])]
    test_data = np.c_[ohe.transform(X_test[cat_cols]), ss.transform(X_test[num_cols])]
    print(f"train:{type(train_data)}")
    print(f"test:{type(test_data)}")
    estimator = LinearRegression(n_jobs=-1)
    estimator.fit(train_data, y_train)
    y_pred = estimator.predict(test_data)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    params = estimator.get_params()
    return r2, rmse, params


def decision_tree(df, significant_cols, target, cat_cols, num_cols):
    return ("r2score", "rmse", "params")


def random_forest(df, significant_cols, target, cat_cols, num_cols):
    return ("r2score", "rmse", "params")


def knn_regression(df, significant_cols, target, cat_cols, num_cols):
    return ("r2score", "rmse", "params")


def ada_boost(df, significant_cols, target, cat_cols, num_cols):
    return ("r2score", "rmse", "params")


def gradient_boost(df, significant_cols, target, cat_cols, num_cols):
    return ("r2score", "rmse", "params")


def polynomial_regression(df, significant_cols, target, cat_cols, num_cols):
    return ("r2score", "rmse", "params")


def elastic_net(df, significant_cols, target, cat_cols, num_cols):
    return ("r2score", "rmse", "params")


def ridge_regression(df, significant_cols, target, cat_cols, num_cols):
    return ("r2score", "rmse", "params")


def lasso_regression(df, significant_cols, target, cat_cols, num_cols):
    return ("r2score", "rmse", "params")


def lightgbm(df, significant_cols, target, cat_cols, num_cols):
    return ("r2score", "rmse", "params")


def xgb_regression(df, significant_cols, target, cat_cols, num_cols):
    return ("r2score", "rmse", "params")
