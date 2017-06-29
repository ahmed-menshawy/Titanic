# -*- coding: UTF-8 -*-
"""
Created on 2017-06-26

@author: zzx
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing

# cabin 根据有无记录设置为{'yes', 'no'}
def set_cabin_type(data):
    data.loc[data.Cabin.notnull(), 'Cabin'] = 'Yes'
    data.loc[data.Cabin.isnull(), 'Cabin'] = 'No'
    return data


# 使用RandomForest 预测缺失的 Age 值
def set_missing_ages(data):
    age_data = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_data[age_data.Age.notnull()].as_matrix()
    unknown_age = age_data[age_data.Age.isnull()].as_matrix()

    X = known_age[:, 1:]
    y = known_age[:, 0]

    rf = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rf.fit(X, y)

    predicted = rf.predict(unknown_age[:, 1:])
    data.loc[data.Age.isnull(), 'Age'] = predicted

    return data


# 类目特征 因子化： cabin: {Yes, No}  -->  cabin_Yes: {1, 0}
#                                       cabin_No: {1, 0}
def category_features_factorization(data_train):

    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df


# rescale Age and Fare
def rescale_features(data):

    scaler = sklearn.preprocessing.StandardScaler()

    age_scale_param = scaler.fit(data['Age'])
    # data['Age_scaled'] = scaler.fit_transform(data['Age'], age_scale_param)
    scaler.fit_transform(data['Age'], age_scale_param)

    fare_scale_param = scaler.fit(data['Fare'])
    # data['Fare_scaled'] = scaler.fit_transform(data['Fare'], fare_scale_param)
    scaler.fit_transform(data['Fare'], fare_scale_param)

    return data


def engineering(dataframe):

    set_cabin_type(dataframe)
    set_missing_ages(dataframe)
    df = category_features_factorization(dataframe)
    return rescale_features(df)

if __name__ == '__main__':
    df = pd.read_csv("../data/train.csv")
    print engineering(df).info()