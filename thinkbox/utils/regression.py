import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFE


def linear_regression(df, significant_cols, target, cat_cols, num_cols):
    ss = StandardScaler()
    X = df[significant_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    estimator = LinearRegression(n_jobs=-1)
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    return "Linear Regression"


def decision_tree(df, significant_cols, target, cat_cols, num_cols):
    return ('qoq', "Decision Tree")


def random_forest(df, significant_cols, target, cat_cols, num_cols):
    return "Random Forest"


def knn_regression(df, significant_cols, target, cat_cols, num_cols):
    return "KNN Regression"


def ada_boost(df, significant_cols, target, cat_cols, num_cols):
    return "Ada Boost Regression"


def gradient_boost(df, significant_cols, target, cat_cols, num_cols):
    return "Gradient Boost Regression"


def polynomial_regression(df, significant_cols, target, cat_cols, num_cols):
    return "Polynomial Regression"


def elastic_net(df, significant_cols, target, cat_cols, num_cols):
    return "Elastic Net"


def ridge_regression(df, significant_cols, target, cat_cols, num_cols):
    return "Ridge Regression"


def lasso_regression(df, significant_cols, target, cat_cols, num_cols):
    return "Lasso Regression"


def lightgbm(df, significant_cols, target, cat_cols, num_cols):
    return "LightGBM Regression"


def xgb_regression(df, significant_cols, target, cat_cols, num_cols):
    return "XGBRF Regression"
