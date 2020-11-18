import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Column Datatype update
def datatype_update(df, updates):
    for col in updates.keys():
        df.loc[:, col] = df[col].astype(updates[col])
    print(df.dtypes)
    return df


# Preprocess the data
def preprocess_data(df, target, drop=None, style='cut', bins=None, labels=None):
    """
    :param df: dataframe
    :param target: target column name
    :param drop: Droppable features
    :param style: ['cut','qcut']
    :param bins: total number of bins
    :param labels: labels
    :return: dataframe
    """

    if drop:
        if isinstance(drop, list):
            for i in drop:
                df.drop(i, 1, inplace=True)
        elif isinstance(drop, str):
            df.drop(drop, 1, inplace=True)
    X = df.drop(target, 1)
    if not labels and not bins:
        bins = 3
        labels = ['Low', 'Mid', 'High']
    cols = X.columns.to_list()
    for col in cols:
        # Checking if the datatype of the column is boolean, object or integer
        if X[col].dtype in (np.bool, np.object, np.int64):
            if X[col].dtype == np.int64 and X[col].nunique() <= 20:
                """
                If the column datatype is integer and the unique value counts
                 of it is less than 20, then just change it as category
                 """
                df.loc[:, col] = X[col].astype('category')
            elif X[col].dtype == np.int64 and 20 < X[col].nunique():
                """
                If the column datatype is integer and the unique value count is more than 20
                then we will be performing a binning operation
                """
                if style == 'cut':
                    df.loc[:, col] = pd.cut(X[col], bins=bins, labels=labels)
                elif style == 'qcut':
                    df.loc[:, col] = pd.qcut(X[col], q=bins, labels=labels, duplicates='drop')
            elif X[col].dtype in (np.bool, np.object):
                """
                If the column datatype is a object or a bool, the convert the datatype in
                category
                """
                df.loc[:, col] = X[col].astype('category')
        elif X[col].dtype == np.float64:
            """
            If the column datatype is float64 then, leave it as it is
            since, we don't need to perform any operation on numerical data
            """
            pass
    return df
