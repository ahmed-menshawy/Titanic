# -*- coding: UTF-8 -*-
"""
Created on 2017-06-29

@author: zzx
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from more.feature_enginnering import feature_enginnering
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics


def build_model():
    df_train = pd.read_csv("../data/train.csv")
    df_train = feature_enginnering(df_train)
    # df_train.info()
    np_train = df_train.as_matrix()
    X = np_train[:, 2:]
    y = np_train[:, 1]

    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    # clf = RandomForestClassifier(n_estimators=100, oob_score=True)
    #                              , min_samples_split=1, max_depth=9, max_features=31, oob_score=False)
    # clf.fit(X, y)
    # print clf.oob_score_
    d = {}
    i = -2
    for a in df_train:
        if i >= 0:
            d[a] = (clf.coef_[0][i])
        i += 1
    pd.DataFrame(d.values(), d.keys()).plot(kind='barh')
    plt.show()

    return clf


def predict(clf):
    df_test = pd.read_csv("../data/test.csv")
    df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test.Fare.mean()
    df_test = feature_enginnering(df_test)
    df_test['CabinClass_T'] = 0
    # df_test.info()
    np_test = df_test.as_matrix()
    predictions = clf.predict(np_test[:, 1:])

    result = pd.DataFrame(
        {'PassengerId': df_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv("../data/logistic_regression_predictions.csv", index=False)


if __name__ == '__main__':
    clf = build_model()
    predict(clf)