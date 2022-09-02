# -*-coding:utf-8-*-
import numpy as np
from sklearn import datasets
from collections import Counter  # 为了做投票
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self):
        pass

    def load_data(self):
        # 导入iris数据
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2022)
        return X_train, X_test, y_train, y_test

    def euclideanDistinct(self, trainX, testX):
        return np.sqrt(sum((trainX - testX) ** 2))

    def knn(self, trainX, trainY, testX, K):
        # 计算样本所有特征和已有label样本的距离
        distinct = [self.euclideanDistinct(X, testX) for X in trainX]
        # 将距离排序，选择最小的K个距离样本的索引
        kneighbors = np.argsort(distinct)[:K]

        # 分类方式
        # 分类方式一：投票
        counter = Counter(trainY[kneighbors])
        # 分类方式二：加入邻居距离的权重
        # wight = []
        # for idx, dist in zip(np.argsort(distinct)[:K], sorted(distinct)[:K]):
        #     for k in range(K):
        #
        #         wight[k]
        return counter.most_common()[0][0]
        

def knn_sklearn(X_train, X_test, y_train, K):
    clf = KNeighborsClassifier(K)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


if __name__ == "__main__":
    knn = KNN()
    X_train, X_test, y_train, y_test = knn.load_data()
    y_pred = [knn.knn(X_train, y_train, test, 3) for test in X_test]
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('MSE:', mean_squared_error(y_test, y_pred))

    y_pred = knn_sklearn(X_train, X_test, y_train, 3)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('MSE:', mean_squared_error(y_test, y_pred))





