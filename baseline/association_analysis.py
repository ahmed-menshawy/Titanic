# -*- coding: UTF-8 -*-
"""
Created on 2017-06-26

@author: zzx
"""
import re

import matplotlib.pyplot as plt
import pandas as pd

"""
    分析各个特征和y值的关系
"""


def Pclass(data_train):
    # 一等舱 生还概率 更高

    Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'survived':Survived_1, u'unsurvived':Survived_0})
    df.plot(kind='bar', stacked=True)   # stacked = True 表示 累计柱状图
    plt.show()


def Sex(data_train):
    # 女性 生还概率 更高， 远高于男性

    Survived_0 = data_train.Sex[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'survived':Survived_1, u'unsurvived':Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.show()


def Embarked(df):
    # 没看到明显 相关性

    s0 = df.Embarked[df.Survived == 0].value_counts()
    s1 = df.Embarked[df.Survived == 1].value_counts()
    pd.DataFrame({'so': s0, 's1': s1}).plot(kind='bar', stacked=True)
    plt.show()


def FamilySize(df):
    # 家庭成员个数为 2,3,4 的，生还概率 高于 0.5
    # 家庭成员为 1 个，或 >=5 的，生还概率 低于 0.5

    df_ = pd.DataFrame(data=(df.SibSp + df.Parch + 1), columns=['Family_Size'])
    df = pd.concat([df, df_], axis=1)
    s0 = df.Family_Size[df.Survived == 0].value_counts()
    s1 = df.Family_Size[df.Survived ==1].value_counts()
    pd.DataFrame({'s0': s0, 's1': s1}).plot(kind='bar', stacked=False)
    plt.show()


def Age(df):
    # 0-6, 46-100 年龄段获救概率 >0.5

    df.Age = df['Age'][df.Age.notnull()].astype(int)
    s0 = df.Age[df.Survived == 0].value_counts()
    s1 = df.Age[df.Survived == 1].value_counts()
    pd.DataFrame({'so': s0, 's1': s1}).plot(kind='bar', stacked=True)
    pd.DataFrame({'s1/s0': s1/s0}).plot(kind='bar')
    plt.show()


def Name(df):
    # Mr, Mrs, Miss, Master, Other

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

    s0 = df.Title[df.Survived == 0].value_counts()
    s1 = df.Title[df.Survived == 1].value_counts()
    # pd.DataFrame({'s0': s0, 's1': s1}).plot(kind='bar', stacked=True)
    pd.DataFrame({'s1/s0': s1/s0}).plot(kind='bar', stacked=True)
    plt.show()

    return df


def Cabin(df):
    # A -- G, N (NaN 值，没有记录)

    from types import StringType
    def get_Cabin_Class(cabin):
        if type(cabin) == StringType:
            return cabin[0]
        return 'N'

    for index, series in df.iterrows():
        df.loc[df.PassengerId == series['PassengerId'], 'Cabin_Class'] = get_Cabin_Class(series.Cabin)

    s0 = df.Cabin_Class[df.Survived == 0].value_counts()
    s1 = df.Cabin_Class[df.Survived == 1].value_counts()
    # pd.DataFrame({'s0': s0, 's1': s1}).plot(kind='bar', stacked=True)
    pd.DataFrame({'s1/s0': s1/s0}).plot(kind='bar', stacked=True)
    plt.show()

    return df

if __name__ == '__main__':

    data_train = pd.read_csv("../data/train.csv", header=0)
    print data_train.info()

    # Pclass(data_train)
    # Name(data_train)
    # Sex(data_train)
    Age(data_train)
    # FamilySize(data_train)
    # Embarked(data_train)
    # Cabin(data_train)
    print data_train







