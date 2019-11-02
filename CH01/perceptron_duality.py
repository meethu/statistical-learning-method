# coding=utf-8
import numpy as np


def calGram(trainData):  # 计算 Gram 矩阵
    m = len(trainData)  # 获取样本数量
    gram = np.zeros((m, m))  # m x m 大小的 零矩阵
    for i in range(m):
        for j in range(m):
            gram[i][j] = np.dot(trainData[i], trainData[j])
    return gram


def perceptron(gram, trainLabel, iter=50):
    a = np.array([0, 0, 0])
    b = 0  # 初始化偏置b为0
    h = 1  # 初始化步长，也就是梯度下降过程中的n，控制梯度下降速率
    m = len(trainLabel)
    y = trainLabel
    Gram = np.mat(gram)
    for k in range(iter):
        for i in range(m):
            # 判断是否是误分类样本
            if y[i] * (np.dot(a * y, Gram[i].T) + b) <= 0:
                # 对于误分类样本，进行梯度下降，更新a[i]和b
                a[i] += h  # h 为步长，此处 h 为 1
                b = b + h * y[i]
        # 打印训练进度
        print('第 %d 轮:共 %d 轮' % (k, iter))
    # 返回训练完的w，b
    return a, b


if __name__ == '__main__':
    trainSet = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]  # 数据集
    trainData, trainLabel = [], []
    for item in trainSet:
        trainData.append(item[0])
        trainLabel.append(item[1])
    # 训练获得权重
    gram = calGram(trainData)
    a, b = perceptron(gram, trainLabel, iter=50)
    # 获取当前时间，作为结束时间
    end = time.time()
    # 显示获得权重
    print('a, b :', a, b)
