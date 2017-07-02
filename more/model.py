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
import tensorflow as tf


def build_model_nn(N_INTER, hidden_units_num):
    # 训练集合
    df_train = pd.read_csv("../data/train.csv")
    df_train = feature_enginnering(df_train)
    np_train = df_train.as_matrix()
    X_data = np_train[:, 2:]
    y_data = np_train[:, 1]

    # 测试集合
    df_test = pd.read_csv("../data/test.csv")
    df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test.Fare.mean()
    df_test = feature_enginnering(df_test)
    df_test['CabinClass_T'] = 0
    np_test = df_test.as_matrix()[:, 1:]

    # 三层神经网络模型
    x = tf.placeholder(dtype=tf.float32, shape=[None, 35])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    # 三层神经网络
    # 输入层 - 隐藏层
    w1_2 = tf.Variable(tf.truncated_normal(shape=[35, hidden_units_num], stddev=0.1))
    b1_2 = tf.Variable(tf.truncated_normal(shape=[hidden_units_num], stddev=0.1))

    h2 = tf.nn.relu(tf.matmul(x, w1_2) + b1_2)

    # 隐藏层 - 输出层
    w2_3 = tf.Variable(tf.truncated_normal(shape=[hidden_units_num, 2], stddev=0.1))
    b2_3 = tf.Variable(tf.truncated_normal(shape=[2], stddev=0.1))

    h3 = tf.nn.relu(tf.matmul(h2, w2_3) + b2_3)

    # 评估
    sess = tf.InteractiveSession()
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h3))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(h3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 训练
    sess.run(tf.global_variables_initializer())
    y_data_ = np.array(list(sess.run(tf.one_hot(y_data, 2))))
    for i in range(N_INTER):
        # if i % 100 == 0:
        #     train_accuracy = accuracy.eval(feed_dict={x: X_data[0:624, :], y: y_data_[0:624, :]})
        #     print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: X_data, y: y_data_})
    # print(N_INTER, hidden_units_num, "test accuracy %g" % accuracy.eval(feed_dict={x: X_data[624:, :], y: y_data_[624:, :]}))

    # 预测
    predictions = np.argmax(sess.run(h3, feed_dict={x: np_test}),axis=1)
    result = pd.DataFrame(
        {'PassengerId': df_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv("../data/logistic_regression_predictions.csv", index=False)


def build_model():
    df_train = pd.read_csv("../data/train.csv")
    df_train = feature_enginnering(df_train)
    df_train.info()
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
    # build_model_nn(2000, 30)