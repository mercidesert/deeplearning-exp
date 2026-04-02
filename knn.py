import numpy as np
class knn:
    def __init__(self):
        pass 
    #初始化构造函数，无操作
    def train(self,X,Y):
        self.X_train=X
        self.Y_train=Y
    #输入矩阵数据以及标准答案数据构建训练集

    def compute_distance(self,X_test):
        num_test=X_test.shape[0]
        num_train=self.X_train.shape[0]

        dists=np.zeros((num_test,num_train))
        for i in range(num_test):
            diff=self.X_train-X_test[i]
            #计算欧氏距离的平方
            squared_diff=diff**2
            sum=np.sum(squared_diff,axis=1)
            distance=np.sqrt(sum)
            dists[i,:]=distance
        return dists
    #训练函数 距离计算
    def predict_labels(self,dists,k=1):
        num_test=dists.shape[0]
        pre=np.zeros(num_test)
        for i in range(num_test):
            closest=np.argsort(dists[i])[:k]
            closest_label=self.Y_train[closest]
            values,counts=np.unique(closest_label,return_counts=True)
            best_guess=values[np.argmax(counts)]
            pre[i]=best_guess
        return pre
    def predict(self,X_test,k=1):
        dists=self.compute_distance(X_test)
        return self.predict_labels(dists,k=k)
