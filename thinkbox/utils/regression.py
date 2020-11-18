import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFE


def linear_regression(data, significant_cols, target, degree=5, encoding="default", style="cut", bins=3,
                      labels=None):
    """
    :param data:
    :param significant_cols:
    :param target:
    :param degree:
    :param encoding:
    :param style:
    :param bins:
    :param labels:
    :return:
    """
    df, X, y = data, data[significant_cols], data[target]
    labels = ['Low','Mid','High']
    for col in significant_cols:
        if X[col].dtype in (np.object, np.bool, np.int64):
            if X[col].dtype == np.int64 and X[col].nunique() <= 20:
                X[col] = X[col].astype('category')
            elif X[col].dtype == np.int64 and 20 < X[col].nunique():
                X[col] = pd.cut(X[col], bins=bins, labels=labels)
            elif X[col].dtype in (np.object, np.bool):
                X[col] = X[col].astype('category')

        elif X[col].dtype in (np.float64,):
            pass
