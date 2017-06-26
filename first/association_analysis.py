# -*- coding: UTF-8 -*-
"""
Created on 2017-06-26

@author: zzx
"""

import matplotlib.pyplot as plt
import pandas as pd

"""
    分析各个特征和y值的关系
"""


def Pclass(data_train):
    Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'survived':Survived_1, u'unsurvived':Survived_0})
    df.plot(kind='bar', stacked=True)   # stacked = True 表示 累计柱状图
    plt.show()


def Sex(data_train):
    Survived_0 = data_train.Sex[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
    df = pd.DataFrame({u'survived':Survived_1, u'unsurvived':Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.show()


if __name__ == '__main__':

    data_train = pd.read_csv("../data/train.csv", header=0)
    # Pclass(data_train)
    Sex(data_train)
