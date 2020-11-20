import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE


def logistic_regression(df, significant_cols, target, cat_cols, num_cols):
    ss = StandardScaler()
    ohe = OneHotEncoder(drop='first', sparse=False)
    le = LabelEncoder()
    X = df[significant_cols]
    y = df[target]
    X_cat = ohe.fit_transform(X[cat_cols])
    X_num = ss.fit_transform(X[num_cols])
    X = np.c_[X_cat, X_num]
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
    estimator = LogisticRegression(random_state=0, max_iter=50000)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, roc_auc


def decision_tree(df, significant_cols, target, cat_cols, num_cols):
    ss = StandardScaler()
    ohe = OneHotEncoder(drop='first', sparse=False)
    le = LabelEncoder()
    X = df[significant_cols]
    y = df[target]
    X_cat = ohe.fit_transform(X[cat_cols])
    X_num = ss.fit_transform(X[num_cols])
    X = np.c_[X_cat, X_num]
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
    estimator = DecisionTreeClassifier(random_state=0)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, roc_auc, estimator.get_params()


classification_models = {
    'Logistic Regression': logistic_regression,
    'Decision Tree': decision_tree,
}
