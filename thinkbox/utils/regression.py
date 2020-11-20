import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE


def linear_regression(df, significant_cols, target, cat_cols, num_cols):
    ss = StandardScaler()
    ohe = OneHotEncoder(drop='first', sparse=False)
    X = df[significant_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train_cat = ohe.fit_transform(X_train[cat_cols])
    X_train_num = ss.fit_transform(X_train[num_cols])
    X_test_cat = ohe.transform(X_test[cat_cols])
    X_test_num = ss.transform(X_test[num_cols])
    train_data = np.c_[X_train_cat, X_train_num]
    test_data = np.c_[X_test_cat, X_test_num]
    estimator = LinearRegression(n_jobs=-1)
    r2_cv_scores = cross_val_score(estimator, train_data, y_train, scoring='r2', cv=3, n_jobs=-1)
    rmse_cv_scores = cross_val_score(estimator, train_data, y_train, scoring='neg_root_mean_squared_error', cv=3,
                                     n_jobs=-1)
    r2 = np.mean(r2_cv_scores)
    rmse = np.abs(np.mean(rmse_cv_scores))
    r2_variance = np.var(r2_cv_scores, ddof=1)
    rmse_variance = np.abs(np.var(rmse_cv_scores, ddof=1))
    estimator.fit(train_data, y_train)
    y_pred = estimator.predict(test_data)
    r2_validation = r2_score(y_test, y_pred)
    rmse_validation = np.sqrt(mean_squared_error(y_test, y_pred))
    params = estimator.get_params()
    return r2, rmse, r2_variance, rmse_variance, r2_validation, rmse_validation, params


def decision_tree(df, significant_cols, target, cat_cols, num_cols):
    ss = StandardScaler()
    ohe = OneHotEncoder(drop='first', sparse=False)
    X = df[significant_cols]
    y = df[target]
    estimator = DecisionTreeRegressor(random_state=0, splitter='best')
    params = {
        'criterion': ['mse', 'friedman_mse', 'mae'],
        'max_features': [None, 'log2', 'sqrt'],
        'ccp_alpha': np.arange(0, 1.1, 0.1)
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train_cat = ohe.fit_transform(X_train[cat_cols])
    X_train_num = ss.fit_transform(X_train[num_cols])
    X_test_cat = ohe.transform(X_test[cat_cols])
    X_test_num = ss.transform(X_test[num_cols])
    train_data = np.c_[X_train_cat, X_train_num]
    test_data = np.c_[X_test_cat, X_test_num]
    gs = GridSearchCV(estimator, params, scoring='r2', cv=3)
    gs.fit(train_data, y_train)
    estimator = gs.best_estimator_
    r2_cv_scores = cross_val_score(estimator, train_data, y_train, scoring='r2', cv=3, n_jobs=-1)
    rmse_cv_scores = cross_val_score(estimator, train_data, y_train, scoring='neg_root_mean_squared_error', cv=3,
                                     n_jobs=-1)
    params = estimator.get_params()
    r2 = np.mean(r2_cv_scores)
    rmse = np.abs(np.mean(rmse_cv_scores))
    r2_variance = np.var(r2_cv_scores, ddof=1)
    rmse_variance = np.abs(np.var(rmse_cv_scores, ddof=1))
    estimator.fit(train_data, y_train)
    y_pred = estimator.predict(test_data)
    r2_validation = r2_score(y_test, y_pred)
    rmse_validation = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse, r2_variance, rmse_variance, r2_validation, rmse_validation, params


def random_forest(df, significant_cols, target, cat_cols, num_cols):
    return "r2", "rmse", "r2_variance", "rmse_variance", "r2_validation", "rmse_validation", "params"


def knn_regression(df, significant_cols, target, cat_cols, num_cols):
    ss = StandardScaler()
    ohe = OneHotEncoder(drop='first', sparse=False)
    X = df[significant_cols]
    y = df[target]
    estimator = KNeighborsRegressor(n_jobs=-1)
    params = {
        'n_neighbors': np.arange(5, int(X.shape[0] * 0.1)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train_cat = ohe.fit_transform(X_train[cat_cols])
    X_train_num = ss.fit_transform(X_train[num_cols])
    X_test_cat = ohe.transform(X_test[cat_cols])
    X_test_num = ss.transform(X_test[num_cols])
    train_data = np.c_[X_train_cat, X_train_num]
    test_data = np.c_[X_test_cat, X_test_num]
    gs = GridSearchCV(estimator, params, scoring='r2', cv=3)
    gs.fit(train_data, y_train)
    estimator = gs.best_estimator_
    r2_cv_scores = cross_val_score(estimator, train_data, y_train, scoring='r2', cv=3, n_jobs=-1)
    rmse_cv_scores = cross_val_score(estimator, train_data, y_train, scoring='neg_root_mean_squared_error', cv=3,
                                     n_jobs=-1)
    params = estimator.get_params()
    r2 = np.mean(r2_cv_scores)
    rmse = np.abs(np.mean(rmse_cv_scores))
    r2_variance = np.var(r2_cv_scores, ddof=1)
    rmse_variance = np.abs(np.var(rmse_cv_scores, ddof=1))
    estimator.fit(train_data, y_train)
    y_pred = estimator.predict(test_data)
    r2_validation = r2_score(y_test, y_pred)
    rmse_validation = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse, r2_variance, rmse_variance, r2_validation, rmse_validation, params


def ada_boost(df, significant_cols, target, cat_cols, num_cols):
    ss = StandardScaler()
    ohe = OneHotEncoder(drop='first', sparse=False)
    X = df[significant_cols]
    y = df[target]
    base = DecisionTreeRegressor(max_depth=3, random_state=0)
    estimator = AdaBoostRegressor(base_estimator=base, random_state=0)
    params = {
        'n_estimators': np.arange(5, int(X.shape[0] * 0.1)),
        'learning_rate': np.arange(0.1, 1.1, 0.1),
        'loss': ['linear', 'square', 'exponential'],
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train_cat = ohe.fit_transform(X_train[cat_cols])
    X_train_num = ss.fit_transform(X_train[num_cols])
    X_test_cat = ohe.transform(X_test[cat_cols])
    X_test_num = ss.transform(X_test[num_cols])
    train_data = np.c_[X_train_cat, X_train_num]
    test_data = np.c_[X_test_cat, X_test_num]
    gs = GridSearchCV(estimator, params, scoring='r2', cv=3)
    gs.fit(train_data, y_train)
    estimator = gs.best_estimator_
    r2_cv_scores = cross_val_score(estimator, train_data, y_train, scoring='r2', cv=3, n_jobs=-1)
    rmse_cv_scores = cross_val_score(estimator, train_data, y_train, scoring='neg_root_mean_squared_error', cv=3,
                                     n_jobs=-1)
    params = estimator.get_params()
    r2 = np.mean(r2_cv_scores)
    rmse = np.abs(np.mean(rmse_cv_scores))
    r2_variance = np.var(r2_cv_scores, ddof=1)
    rmse_variance = np.abs(np.var(rmse_cv_scores, ddof=1))
    estimator.fit(train_data, y_train)
    y_pred = estimator.predict(test_data)
    r2_validation = r2_score(y_test, y_pred)
    rmse_validation = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse, r2_variance, rmse_variance, r2_validation, rmse_validation, params


def gradient_boost(df, significant_cols, target, cat_cols, num_cols):
    return "r2", "rmse", "r2_variance", "rmse_variance", "r2_validation", "rmse_validation", "params"


def polynomial_regression(df, significant_cols, target, cat_cols, num_cols):
    return "r2", "rmse", "r2_variance", "rmse_variance", "r2_validation", "rmse_validation", "params"


def elastic_net(df, significant_cols, target, cat_cols, num_cols):
    return "r2", "rmse", "r2_variance", "rmse_variance", "r2_validation", "rmse_validation", "params"


def ridge_regression(df, significant_cols, target, cat_cols, num_cols):
    return "r2", "rmse", "r2_variance", "rmse_variance", "r2_validation", "rmse_validation", "params"


def lasso_regression(df, significant_cols, target, cat_cols, num_cols):
    return "r2", "rmse", "r2_variance", "rmse_variance", "r2_validation", "rmse_validation", "params"


def lightgbm(df, significant_cols, target, cat_cols, num_cols):
    return "r2", "rmse", "r2_variance", "rmse_variance", "r2_validation", "rmse_validation", "params"


def xgb_regression(df, significant_cols, target, cat_cols, num_cols):
    return "r2", "rmse", "r2_variance", "rmse_variance", "r2_validation", "rmse_validation", "params"


regression_models = {
    'Linear Regression': linear_regression,
    'Decision Tree': decision_tree,
    'Random Forest': random_forest,
    'KNN Regression': knn_regression,
    'Ada Boost': ada_boost,
    'Gradient Boost': gradient_boost,
    'Polynomial Regression': polynomial_regression,
    'Elastic Net': elastic_net,
    'Ridge Regression': ridge_regression,
    'Lasso Regression': lasso_regression,
    'Light GBM': lightgbm,
    'XGB Regression': xgb_regression,
}
