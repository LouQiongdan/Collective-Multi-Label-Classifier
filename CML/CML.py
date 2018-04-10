from __future__ import print_function
import scipy.io as sio
from sklearn import preprocessing
from scipy.special import comb  # 排列组合中的组合公式
from sklearn.metrics import hamming_loss
from sklearn.decomposition import PCA
import numpy as np
import math

import random


def f_k(dataSet, Labels, d, q):
    """

    :param dataSet: 某一个样本的特征集
    :param Labels: 某一个样本的标签集
    :param d: 样本的维度，即一个样本含有的特征数
    :param q: 标签的维度，即标签集中标签的个数
    :return: 返回的是fk(x,y)
    """
    F_k = []
    for l in range(d):
        for j in range(q):
            if Labels[j] == 1:
                F_k.append(dataSet[l])
            else:
                F_k.append(0)

    for j1 in range(q - 1):
        for j2 in range(j1 + 1, q):
            y_j1 = Labels[j1]
            y_j2 = Labels[j2]
            if y_j1 == 1 and y_j2 == 1:
                F_k.append(1)
            else:
                F_k.append(0)
            if y_j1 == 1 and y_j2 == 0:
                F_k.append(1)
            else:
                F_k.append(0)
            if y_j1 == 0 and y_j2 == 1:
                F_k.append(1)
            else:
                F_k.append(0)
            if y_j1 == 0 and y_j2 == 0:
                F_k.append(1)
            else:
                F_k.append(0)

    return F_k


def rand_labels():
    """
    #关于这个函数的for循环的嵌套次数，Y标签集中，有几个标签就嵌套几层。（y1,y2,...,yq）
    :return: 返回的是q维的标签集的所有组合情况
    """
    """
    randLabels=[]
    for i in range(2):
        randLabels.append([i])
    return randLabels
    """
    randLabels = []
    for i1 in range(2):
        for i2 in range(2):
            for i3 in range(2):
                 randLabels.append([i1, i2, i3])
    randLabels = np.array(randLabels)
    return randLabels



def Z(dataSet, d, q, Lambda):  # 对于某一个样本的Z
    """

    :param dataSet: 某一个样本的特征集
    :param d: 样本的维度，即特征的个数
    :param q: 标签集的个数
    :param Lambda: Lambda是一个1*K维向量
    :return: 归一化范数，以及所有标签集组合的对应f_k
    """
    randLabels = rand_labels()
    Lambda = np.array(Lambda)
    # F_k=[]
    Z = 0.0
    for i in range(len(randLabels)):
        temp_sum = 0.0
        fk = f_k(dataSet, randLabels[i], d, q)
        fk = np.array(fk)
        #  F_k.append(fk)
        temp_sum = math.exp((Lambda * fk).sum())
        Z += temp_sum
    # F_k=np.array(F_k)
    # return Z,F_k
    return Z


# 求目标函数l(Lambda|D)
def obj_func(DataSets, Labels, thegma, Lambda):
    """

    :param q:标签集的维度
    :param DataSets:所有训练样本的特征集
    :param Labels:所有训练样本的标签集
    :param thegma:自己给定的参数值，2**-6,2**-5,2**-4,2**-3,2**-2,2**-1,2**1,2**2,2**3,2**4,2**5,2**6逐个取值，参数寻优
    :return:目标函数，以及待定参数Lambda
    """
    samples = len(DataSets)
    d = len(DataSets[0])
    q = len(Labels[0])
    temp_sum = 0.0
    for i in range(samples):
        fk = f_k(DataSets[i], Labels[i], d, q)
        fk = np.array(fk)
        z = Z(DataSets[i], d, q, Lambda)

        z=math.fabs(z)###########新添的,因为z可能为负数
        temp_sum += sum(Lambda * fk) - math.log2(z)
        temp_div = sum((np.array(Lambda)) ** 2 / (2 * thegma ** 2))  # temp_div=sum(Lambda**2/(2*thegma**2))
    l = temp_sum - temp_div
    return -l  # 求解l的最大值，可以转化为求-l的最小值问题

def Train(K,thegma):
    """

    :param K: 标记空间的总个数
    :param thegma: 是经验参数，自己设定
    :return: 返回的是训练好的参数lambda
    """
    N = 100  # 迭代次数
    step = 0.5  # 初始步长
    epsilon = 0.00001
    variables = int(K)  # 变量数目
    init_lam = np.ones([1, int(K)]).tolist()[0]  # 初始点
    walk_num = 1  # 初始化随机游走次数
    #print("迭代次数：", N)
    #print("初始步长：", step)
    #print("epsilon:", epsilon)
    #print("变量数目：", variables)
    #print("初始点：", init_lam)
    while (step > epsilon):
        k = 1  # 初始化计数器
        while (k < N):
            u = [random.uniform(-1, 1) for i in range(variables)]  # 随机向量
            # u1为标准化之后的随机向量
            u1 = [u[i] / math.sqrt(sum([u[i] ** 2 for i in range(variables)])) for i in range(variables)]
            init_lam1 = [init_lam[i] + step * u1[i] for i in range(variables)]
            if (obj_func(train_data, train_target, thegma, init_lam1) < obj_func(train_data, train_target, thegma,
                                                                                 init_lam)):
                # 如果找到了更优点
                k = 1
                init_lam = init_lam1
                print(init_lam)
            else:
                k += 1
        step = step / 2
        #print("第%d次随机游走完成。" % walk_num)
        walk_num += 1
    #print("随机游走次数：", walk_num - 1)
    print("最终最优点：", init_lam)
    return init_lam

def Pred(test_data,Lambda,d,q):
    randLabels=rand_labels()
    bestLabels=None
    z=Z(test_data, d, q, Lambda)
    bestP=-1.0
    for i in range(len(randLabels)):
        fk=f_k(test_data,randLabels[i],d,q)
        fk=np.array(fk)
        temp_P=math.exp((Lambda*fk).sum())/z
        if temp_P>bestP:
            bestP=temp_P
            bestLabels=randLabels[i]
    return bestLabels

if __name__ == '__main__':
    """
    dataSet=[0.5,0.1,0.3]
    Labels=[1,0]
    f=f_k(dataSet,Labels,3,2)
    print(f)
    """
    #训练集的处理
    data_path = 'ML_data/train_300'
    data = sio.loadmat(data_path)
    Train_data = data['tr_300'][:,1:73]
    #train_target = data['trl_300'][:,1:7]
    train_target = data['trl_300'][:, 1:4]
    min_max_scaler = preprocessing.MinMaxScaler()
    Train_data = min_max_scaler.fit_transform(Train_data)

    #主成分降维
    pca=PCA(n_components=5)#保留4个主成分
    train_data=pca.fit_transform(Train_data)

    d = len(train_data[0])
    q = len(train_target[0])
    K = d * q + 4 * comb(q, 2)
    thegma = 2 ** (1)        #参数寻优，-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6
    #训练数据集
    lam=Train(K,thegma)
    #测试出来的thegma=2**0的情况的最优解为：lam= [-0.7719195402696462, 0.015068074559837278, 0.7792862651462468, -0.30323081524481627, -0.1650971099932004, -0.025357213718749813, -33.38015572522746, -33.34392047803654, -33.346302897399156, -32.73809505325492, -33.72741920696339, -32.99667535984283, -32.81677250923353, -33.267345579098595, -33.258058952484575, -33.46815816066302, -33.28604205672134, -32.795830232030255]

    #测试集的处理
    data_path1='ML_data/test_100'
    data1=sio.loadmat(data_path1)

    Test_data=data1['te_100'][:,1:73]
    #test_target=data1['tel_100'][:,1:7]
    test_target = data1['tel_100'][:, 1:4]
    test_data=pca.fit_transform(Test_data)
    preLabels=[]
    for j in range(len(test_data)):
        preLabels.append(Pred(test_data[j],lam,d,q))
    preLabels=np.array(preLabels)
    acc=hamming_loss(test_target,preLabels)#汉明损失，越低越好
    print('acc=',acc)











