# -*-coding:utf-8-*-

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class LRegression:
    def __init__(self):
        self.iterations = 5000
        self.learn_rate = 0.01
        self.threshold = 0.5
        self.batch_size = 32
        
    # 激活函数
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # 权重, 偏差初始化
    def initialize_wight_zero(self, dim):
        w = 0.001 * np.random.randn(1, dim).reshape((-1, 1))
        b = 0
        assert w.shape == (dim, 1)
        assert (isinstance(b, float) or isinstance(b, int))
        return w, b
    
    # 梯度上升
    def grad_ascent(self, dataset, labels, w, b):
        m, n = np.shape(dataset)
        
        # 转换为numpy矩阵
        # 构建特征矩阵X
        x = np.array(dataset)
        # 构建标签矩阵y
        y = np.array(labels).reshape(-1, 1)

        dw, db, loss = 0, 0, 0
        for i in range(self.iterations):
            h_x = self.sigmoid(np.sum(np.dot(x, w) + b))  # 逻辑回归数据公式
            loss = - 1 / m * np.sum(np.log(h_x) * y + (1 - y) * np.log(1 - h_x))  # 损失函数：交叉熵
            
            # 更新w，b
            w = w - 1 / m * np.dot(x.T, (h_x - y)) * self.learn_rate
            b = b - 1 / m * np.sum(h_x - y)
        return w, b, loss
    
    # 随机梯度上升
    ## 随机选择部分样本更新回归系数w，b，可以减少周期性的波动，加快收敛
    def stoc_grad_ascent(self, dataset, labels, w, b):
        m, n = np.shape(dataset)
    
        # 转换为numpy矩阵
        x = np.array(dataset)
        # 构建标签矩阵y
        y = np.array(labels).reshape(-1, 1)
    
        dw, db, loss = 0, 0, 0
        for i in range(self.iterations):
            idx = np.arange(n)
            # learn_rate = 4 / (1.0 + i) + 0.01    # 学习率根据迭代次数发生变化
            learn_rate = self.learn_rate   # 固定学习率
            rand_idx = [int(np.random.uniform(0, len(idx))) for _ in range(self.batch_size)]  # 随机选择部分样本
            
            h_x = self.sigmoid(np.sum(np.dot(x[rand_idx], w) + b))  # 逻辑回归数据公式
            loss = - 1 / m * np.sum(np.log(h_x) * y[rand_idx] + (1 - y[rand_idx]) * np.log(1 - h_x))  # 损失函数：交叉熵

            # 更新w，b
            w = w - 1 / m * np.dot(np.mat(x[rand_idx]).T, np.mat(h_x - y[rand_idx])) * learn_rate
            b = b - 1 / m * np.sum(h_x - y[rand_idx])
            np.delete(x, [rand_idx])
        return w, b, loss
    
    # 预测函数
    def predict(self, w, b, x):
        y_pred = []
        h_x = self.sigmoid(np.dot(x, w) + b)
        for i in range(h_x.shape[0]):
            if h_x[i, 0] > self.threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_pred
    
    # 模型评估: 混淆矩阵
    def assessment(self, y_pred, labels):
        tp, tn, fp, fn = 0, 0, 0, 0
        for y_, y in zip(labels, y_pred):
            if y == 1 and y_ == 1:
                tp += 1
            elif y == 0 and y_ == 0:
                tn += 1
            elif y == 1 and y_ == 0:
                fp += 1
            elif y == 0 and y_ == 1:
                fn += 1
        acc = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        persist = tp / (tp + fp)
        print("准确率：", acc, "召回率：", recall, "精准率：", persist)
        
# 加载数据
def load_data():
    dataset = datasets.load_breast_cancer()
    train_x, train_y = dataset['data'][0:400], dataset['target'][0:400]
    test_x, test_y = dataset['data'][400:-1], dataset['target'][400:-1]
    train_x = train_x.reshape(train_x.shape[0], -1)
    test_x = test_x.reshape(test_x.shape[0], -1)
    
    return train_x, train_y, test_x, test_y

# sklearn 实现逻辑回归
def sklearn_lr():
    train_x, train_y, test_x, test_y = load_data()
    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    y_pred = lr.predict(test_x)
    LRegression().assessment(y_pred, test_y)
    return lr


# 画出决策边界
## 仅画出二维图形
def plt_bast_fit(train_x, train_y):
    lr = LogisticRegression()
    lr.fit(train_x[:, [0, 1]], train_y)    # 选择两个特征画出决策边界
    
    # 获取两个特征的最大值，最小值
    x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 1
    y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 1
    x, y = np.meshgrid(np.arange(x_min, x_max, 0.1),
                       np.arange(y_min, y_max, 0.1))
    
    Z = lr.predict(np.c_[x.ravel(), y.ravel()])
    Z = Z.reshape(x.shape)
    
    plt.plot()
    plt.contourf(x, y, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=plt.cm.brg)
    plt.title("Logistic Regression")
    plt.xlabel("Petal.Length")
    plt.ylabel("Petal.Width")
    plt.show()

if __name__ == "__main__":
    lr = LRegression()
    
    # 加载数据
    train_x, train_y, test_x, test_y = load_data()
    print("***********************************简单版lr*************************************")
    # 权重初始化
    w, b = lr.initialize_wight_zero(train_x.shape[1])
    # 梯度下降优化参数
    dw, db, loss = lr.grad_ascent(train_x, train_y, w, b)
    # lr预测结果
    y_pred = lr.predict(dw, db, test_x)
    # 模型准确率评估
    lr.assessment(y_pred, test_y)

    print("***********************************优化版lr*************************************")
    w, b = lr.initialize_wight_zero(train_x.shape[1])
    # 梯度下降优化参数
    dw, db, loss = lr.stoc_grad_ascent(train_x, train_y, w, b)
    # lr预测结果
    y_pred = lr.predict(dw, db, test_x)
    # 模型准确率评估
    lr.assessment(y_pred, test_y)

    print("**********************************sklearn版lr**********************************")
    sklearn_lr()

    plt_bast_fit(train_x, train_y)
    

    
        
    
            
            
            
            
        