# -*-coding:utf-8-*-
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

def knn_model(X, y):
    # 候选k值
    k_values = [1, 2, 3, 4]
    # k折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=2022)
    
    # 初始化k和准确率acc
    bast_k = k_values[0]
    bast_acc = 0
    
    # 网格搜索+交叉验证
    for i, k in enumerate(k_values):
        kf_socre = 0
        for train_idx, valid_idx in kf.split(X):
            # 实例化KNN模型
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            # 训练
            knn.fit(X[train_idx], y[train_idx])
            # 计算准确率
            kf_socre += knn.score(X[valid_idx], y[valid_idx])
        # 5折交叉验证的平均准确率
        agv_kf_score = kf_socre / 5
        # 更新bast_k,bast_acc
        if agv_kf_score > bast_acc:
            bast_acc, bast_k = agv_kf_score, k
        
        print(f"第{i}轮搜索：bast_k：{bast_k}, 准确率：{bast_acc}")


iris = datasets.load_iris()
# X 为特征集，y 为标签集
X = iris['data']
y = iris['target']

knn_model(X, y)
