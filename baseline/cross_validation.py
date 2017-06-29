# -*- coding: UTF-8 -*-
"""
Created on 2017-06-27

@author: zzx
"""

from sklearn import cross_validation, linear_model

from baseline.feature_engineering import engineering
import pandas as pd


df = engineering(pd.read_csv("../data/train.csv"))
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)


# k折交叉验证
train_data = df.as_matrix()
X = train_data[:, 2:]
y = train_data[:, 1]
print '5折交叉验证：', cross_validation.cross_val_score(clf, X, y, cv=5)

# 7:3 分割训练集, 查看 误分类点
split_train, split_cv = cross_validation.train_test_split(df, test_size=0.3, random_state=0)
clf.fit(split_train[:, 2:], split_train[:, 1])

predictions = clf.predict(split_cv[:, 2:])
origin_df = pd.read_csv("../data/train.csv")
error_case = origin_df.loc[df.PassengerId.isin(split_cv[:, 0][predictions != split_cv[:, 1]])]
print error_case
