# -*- coding: UTF-8 -*-
"""
Created on 2017-06-29

@author: zzx
"""
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import sklearn.preprocessing
import numpy as np

def Name(df):
    # Mr, Mrs, Miss, Master, Other

    import re

    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            title = title_search.group(1)
            if title not in ['Mr', 'Mrs', 'Miss', 'Master']:
                title = 'Other'
            return title
        return ""

    for index, series in df.iterrows():
        df.loc[df.PassengerId == series['PassengerId'], 'Title'] = get_title(series['Name'])
    return df

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

def Age(df):
    # 离散化
    df.loc[df.Age<7, 'AgeLevel'] = 'child'
    df.loc[(df.Age>=7) & (df.Age<19), 'AgeLevel'] = 'teenager'
    df.loc[(df.Age>=19) & (df.Age<46), 'AgeLevel'] = 'middleAge'
    df.loc[df.Age>=46, 'AgeLevel'] = 'aged'

    return df


def FamilySize(df):
    # 家庭成员个数为 2,3,4 的，生还概率 高于 0.5
    # 家庭成员为 1 个，或 >=5 的，生还概率 低于 0.5

    df_fs = pd.DataFrame(data=(df.SibSp + df.Parch + 1), columns=['FamilySize'])

    df_fs.loc[(df_fs.FamilySize > 4), 'FamilySize'] = 'L'
    df_fs.loc[(df_fs.FamilySize > 1) & (df_fs.FamilySize < 5), 'FamilySize'] = 'M'
    df_fs.loc[(df_fs.FamilySize == 1), 'FamilySize'] = 'S'

    return pd.concat([df, df_fs], axis=1)


def Cabin(df):
    # A -- G, T, N (NaN 值，没有记录)
    from types import StringType

    def get_Cabin_Class(cabin):
        if type(cabin) == StringType:
            return cabin[0]
        return 'N'

    for index, series in df.iterrows():
        df.loc[df.PassengerId == series['PassengerId'], 'CabinClass'] = get_Cabin_Class(series.Cabin)

    return df

    # df.loc[df.Cabin.notnull(), 'CabinClass'] = 'Yes'
    # df.loc[df.Cabin.isnull(), 'CabinClass'] = 'No'
    # return df

def Embarked(df):
    df.loc[df.Embarked.isnull(), 'Embarked'] = 'S'

def category_features_factorization(df):

    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
    dummies_Sex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dummies_Title = pd.get_dummies(df['Title'], prefix='Title')
    dummies_AgeLevel = pd.get_dummies(df['AgeLevel'], prefix='AgeLevel')
    dummies_FamilySize = pd.get_dummies(df['FamilySize'], prefix='FamilySize')
    dummies_CabinClass = pd.get_dummies(df['CabinClass'], prefix='CabinClass')
    dummies_Child = pd.get_dummies(df['Child'], prefix='Child')

    df = pd.concat([df, dummies_Pclass, dummies_Sex, dummies_Embarked, dummies_Title,
                    dummies_AgeLevel, dummies_FamilySize, dummies_CabinClass, dummies_Child], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title', 'AgeLevel',
              'FamilySize', 'CabinClass', 'Child'], axis=1, inplace=True)
    return df


def rescale_features(data):

    scaler = sklearn.preprocessing.StandardScaler()

    age_scale_param = scaler.fit(data['Age'])
    scaler.fit_transform(data['Age'], age_scale_param)

    fare_scale_param = scaler.fit(data['Fare'])
    scaler.fit_transform(data['Fare'], fare_scale_param)

    return data


def Child(df):
    df.loc[df.Age < 13, 'Child'] = 'Yes'
    df.loc[df.Age > 12, 'Child'] = 'No'

def feature_enginnering(df):

    df = set_missing_ages(df)
    Name(df)
    Child(df)
    Age(df)
    df = FamilySize(df)
    df = Cabin(df)
    Embarked(df)
    df = category_features_factorization(df)
    df = rescale_features(df)

    return df

if __name__ == '__main__':
    df = pd.read_csv("../data/train.csv")
    df = feature_enginnering(df)
