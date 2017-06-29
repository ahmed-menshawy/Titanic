# -*- coding: UTF-8 -*-
"""
Created on 2017-06-26

@author: zzx
"""
import numpy as np
import pandas as pd

from feature_engineering import engineering
from sklearn import linear_model


def build_model():
    # 数据清洗、特征工程
    train_df = engineering(pd.read_csv("../data/train.csv"))

    # 用 正则表达式 取出我们要的属性值
    train_np = train_df.filter(regex='Survived|Age.*|SibSp|Parch|Fare.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*').as_matrix()

    X = train_np[:, 1:]
    y = train_np[:, 0]

    # 模型
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    print pd.DataFrame({"columns": list(train_df.columns)[2:], "coef": list(clf.coef_.T)})

    # 预测
    test_df = pd.read_csv("../data/test.csv")
    test_df.loc[test_df.Fare.isnull(), 'Fare'] = 0
    test_df = engineering(test_df)

    test_X = test_df.filter(regex='Age.*|SibSp|Parch|Fare.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*').as_matrix()
    predictions = clf.predict(test_X)

    result = pd.DataFrame(
        {'PassengerId': test_df['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv("../data/logistic_regression_predictions.csv", index=False)

if __name__ == '__main__':
    build_model()