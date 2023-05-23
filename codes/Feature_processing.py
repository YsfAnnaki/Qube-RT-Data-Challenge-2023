#############################################
# 10 - 03 - 2023
# @ Youssef ANNAKI
#############################################

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.preprocessing import StandardScaler



""" Features selection tools. """

class feature_selection(BaseEstimator, TransformerMixin):
    """
        Global feature selection filter (filter(s) to be specified in __init__).
         - X_train: train set.
         - Y_train: target variable (related to X_train).
         - n_features: number of features to keep after the filtring process (default == 1).
    """

    def __init__(self, n_features = 1):
        self.n_features = n_features
        self.filter = SelectKBest(score_func = mutual_info_regression, k = n_features) ##

        return

    def fit(self, X, y = None,):
        self.filter.fit(X, y,)

        return self

    def transform(self, X, y = None, ):
        self.X_filtred = self.filter.transform(X)

        return self.X_filtred

    def fit_transform(self, X, y = None, **fit_params):
        return self.filter.fit(X, y).transform(X)

    def get_features(self,):
        return self.filter.get_support()


""" Outliers detection tools. """

def if_outliers_handling(X_train, Y_train):
    """
        Isolation Forest.
         - n_jobs is set to -1 in order to fit the forest. To be changed if the processor resources need to be managed.
    """

    isf = IsolationForest(n_jobs = -1, random_state = 1)
    isf.fit(X_train, Y_train)

    temp = isf.predict(X_train)

    index_X = X_train.index

    index_to_drop = []

    count_droped = 0

    for i in range(len(temp)):
        if temp[i] == -1:
            count_droped += 1
            index_to_drop.append(index_X[i])

    X_train_clean = X_train.drop(index_to_drop)
    Y_train_clean = Y_train.drop(index_to_drop)

    """Logs. """

    print(f"Number of Dropped rows: {count_droped}")

    return {"filter": isf, "X_train": X_train_clean, "Y_train": Y_train_clean}


def zs_outliers_handling(X_train, Y_train, threshold):
    """
        Outliers handling based on extreme values after Z-scoring.
    """

    outliers_index = []

    print(f"Initial size: {len(X_train)}")

    for feature in X_train.columns:
        mean_ = X_train[feature].mean()
        std_ = X_train[feature].std()

        X_train[feature] = (X_train[feature] - mean_) / std_

        outliers_index += X_train[abs(X_train[feature]) > threshold].index

        X_train.drop(outliers_index, axis = 0, inplace = True)
        Y_train.drop(outliers_index, axis=0, inplace = True)

    print(f"Final size: {len(X_train)}")

    return {"X_train": X_train, "Y_train": Y_train}




""" Features transformation. """

def feature_transf(X_train):
    """ Feature scaling using StandardScaler. """

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    return {"scaler": scaler, "X_train": X_train_scaled}


