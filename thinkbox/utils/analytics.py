from scipy import stats
import pandas as pd
import numpy as np


# Defining our ANOVA
def anova(X, y, df):
    """
    :param X: X is the Column data
    :param y: Target or predictor
    :param df: dataframe
    :return:
    """
    # Checking if the datatype of our X parameter is object
    if df[y].dtype == np.float64:
        groups = {}  # For storing our grouped data
        classes = df[X].unique()  # Getting the different classes
        mystr = ''  # Initialize an Empty string
        for c in classes:
            groups[c] = df[df[X] == c][y]
        for k, v in groups.items():
            if not isinstance(k, str):
                mystr += 'groups[' + str(k) + '],'
            else:
                mystr += 'groups["' + k + '"],'
        stat_test = 'stats.f_oneway(' + mystr + ')'
        return eval(stat_test)


def conduct_test(df, target, drop=None):
    """
    :param target: target feature
    :param drop: unwanted feature(i.e, index)
    :param df: dataframe
    :return:
    """
    if drop:
        df = df.drop(drop, 1)
    X = df.drop(target, 1)
    y = df[target]
    cols = X.columns.to_list()
    test_results = {}  # Dictionary to store test results
    for col in cols:
        test_results[col] = {}  # initializing the key as column name
        if X[col].dtype == np.bool:
            X[col] = X[col].astype(object)
        if y.dtype == np.bool:
            y = y.astype(object)

        if X[
            col].dtype == np.float64 and y.dtype == np.float64:  # Condition when both the predictor and the target is a qualitative variable
            test_results[col][
                'statistical test conducted'] = 'pearson correlation coefficient'  # Statistical Test Conducted
            test_results[col]['test statistic'] = stats.pearsonr(X[col], y)[0]  # Correlation Coefficient
            test_results[col]['p value'] = stats.pearsonr(X[col], y)[1]  # p value
            if test_results[col]['p value'] < 0.05:  # Significance of the features
                test_results[col]['test decision'] = 'significant'
            else:
                test_results[col]['test decision'] = 'insignificant'

        elif X[col].dtype == np.object or X[
            col].dtype == np.int64 and y.dtype == np.object:  # Categorical vs Categorical i.e, chi-square test
            if X[col].dtype == np.int64:
                X[col] = pd.cut(X[col], bins=3, labels=['Low', 'Medium', 'High'])
            test_results[col]['statistical test conducted'] = 'chi-square test'
            ct = pd.crosstab(X[col], y)  # Creating a cross-tab of the data since both are categorical
            test_results[col]['test statistic'] = stats.chi2_contingency(ct)[0]  # Test Statistic
            test_results[col]['p value'] = stats.chi2_contingency(ct)[1]  # p value for chi-square
            if test_results[col]['p value'] < 0.05:  # Significance of the features
                test_results[col]['test decision'] = 'significant'
            else:
                test_results[col]['test decision'] = 'insignificant'

        elif ((X[col].dtype == np.object or X[col].dtype == np.int64) and y.dtype == np.float64 and X[
            col].nunique() == 2) or (X[col].dtype == np.float64 and (
                y.dtype == np.object or y.dtype == np.int64) and y.nunique() == 2):  # To conduct t test of independance
            test_results[col]['statistical test conducted'] = 't test independant'
            if X[col].dtype == np.object or X[col].dtype == np.int64:
                classes = X[col].unique()  # Finding the unique classes
                groups = {}
                for c in classes:
                    groups[c] = df[df[col] == c][target]
                test_results[col]['test statistic'] = stats.ttest_ind(groups[classes[0]], groups[classes[1]])[
                    0]  # Test Statistic
                test_results[col]['p value'] = stats.ttest_ind(groups[classes[0]], groups[classes[1]])[
                    1]  # p value for t test of independance
                if test_results[col]['p value'] < 0.05:  # Significance of the features
                    test_results[col]['test decision'] = 'significant'
                else:
                    test_results[col]['test decision'] = 'insignificant'
            else:
                classes = y.unique()  # Finding the unique classes
                groups = {}
                for c in classes:
                    groups[c] = df[df[target] == c][col]
                test_results[col]['test statistic'] = stats.ttest_ind(groups[classes[0]], groups[classes[1]])[
                    0]  # Test Statistic
                test_results[col]['p value'] = stats.ttest_ind(groups[classes[0]], groups[classes[1]])[
                    1]  # p value for t test of independance
                if test_results[col]['p value'] < 0.05:  # Significance of the features
                    test_results[col]['test decision'] = 'significant'
                else:
                    test_results[col]['test decision'] = 'insignificant'

        elif ((X[col].dtype == np.object or X[col].dtype == np.int64) and y.dtype == np.float64 and X[
            col].nunique() > 2) or (
                X[col].dtype == np.float64 and (y.dtype == np.object or y.dtype == np.int64) and y.nunique() > 2):
            test_results[col]['statistical test conducted'] = 'ANOVA'
            if X[col].dtype == np.object or X[col].dtype == np.int64:
                test_statistic, pvalue = anova(col, y.name, df)
            else:
                test_statistic, pvalue = anova(y.name, col, df)
            test_results[col]['test statistic'] = test_statistic  # Test Statistic
            test_results[col]['p value'] = pvalue  # p value for t test of independance
            if test_results[col]['p value'] < 0.05:  # Significance of the features
                test_results[col]['test decision'] = 'significant'
            else:
                test_results[col]['test decision'] = 'insignificant'

    return pd.DataFrame(test_results)
