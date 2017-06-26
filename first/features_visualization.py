# -*- coding: UTF-8 -*-
"""
Created on 2017-06-21

@author: zzx
"""

import pandas as pd
import matplotlib.pyplot as plt

"""
    乘客各特征的分布
"""

train_data = pd.read_csv("../data/train.csv", header=0)     # <class 'pandas.core.frame.DataFrame'>
# train_data.info()
# print train_data.describe()

fig = plt.figure()

plt.subplot2grid(shape=(2, 3), loc=(0, 0))
train_data.Survived.value_counts().plot(kind='bar')     # 柱状图
plt.title(u"Survived")

plt.subplot2grid(shape=(2, 3), loc=(0, 1))
train_data.Pclass.value_counts().plot(kind='bar')
plt.title(u"Pclass")

plt.subplot2grid(shape=(2, 3), loc=(0, 2))
plt.scatter(train_data.Survived, train_data.Age)
plt.title(u"Age distribution")

plt.subplot2grid((2, 3),(1, 0), colspan=2)
train_data.Age[train_data.Pclass == 1].plot(kind='kde')     # 核密度估计
train_data.Age[train_data.Pclass == 2].plot(kind='kde')
train_data.Age[train_data.Pclass == 3].plot(kind='kde')
plt.xlabel(u"Age")
plt.ylabel(u"Probability Density")
plt.title(u"Age Distribution")
plt.legend((u'1', u'2',u'3'),loc='best')

plt.subplot2grid((2,3),(1,2))
train_data.Embarked.value_counts().plot(kind='bar')
plt.title(u"Embarked")

plt.show()
