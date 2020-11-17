from scipy import stats
import pandas as pd
import numpy as np


def conduct_test(df, target, drop):
    data = df.drop(drop, 1)
    cols = data.columns.to_list()
    datatypes = {}
    for col in cols:
        datatypes[col] = data[col].dtype
    y = data[target]
    X = data.drop(target, 1)
    data_cols = X.columns.to_list()
    test = {}
    # if our target is a float then regression
    if datatypes[target] == np.float64:
        for col in data_cols:
            test[col] = {}
            test[col]['corr_coef'] = stats.pearsonr(X[col], y)[0]
            test[col]['p_value'] = stats.pearsonr(X[col], y)[1]
            if test[col]['p_value'] < 0.05:
                test[col]['significance'] = "significant"
            else:
                test[col]['significance'] = "insignificant"
        test_results = pd.DataFrame(test).T
        return test_results
