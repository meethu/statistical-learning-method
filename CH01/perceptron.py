# coding=utf-8
import numpy as np


def perceptron(trainData, trainLabel, iter=50):
    dataMat = np.mat(trainData)  # 将训练数据转化为矩阵
    labelMat = np.mat(trainLabel).T
    m = dataMat.shape[0]  # 获取样本数量
    w = [0, 0]
    b = 0  # 初始化偏置b为0
    h = 1  # 初始化步长，也就是梯度下降过程中的n，控制梯度下降速率

    for k in range(iter):
        for i in range(m):
            # 获取当前样本的向量
            xi = dataMat[i]
            # 获取当前样本所对应的标签
            yi = labelMat[i]
            # 判断是否是误分类样本
            # 误分类样本特诊为： -yi(w*xi+b)>=0
            # 在书的公式中写的是>0，实际上如果=0，说明该点在超平面上，也是不正确的
            if yi * (w * xi.T + b) <= 0:
                # 对于误分类样本，进行梯度下降，更新w和b
                w = w + h * yi * xi
                b = b + h * yi

        # 打印训练进度
        print('第 %d 轮:共 %d 轮' % (k, iter))
    # 返回训练完的w，b
    return w, b


if __name__ == '__main__':
    trainSet = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]  # 数据集
    trainData, trainLabel = [], []
    for item in trainSet:
        trainData.append(item[0])
        trainLabel.append(item[1])
    # 训练获得权重
    w, b = perceptron(trainData, trainLabel, iter=50)
    # 显示最终 w，b 值
    print('w, b :', w, b)
