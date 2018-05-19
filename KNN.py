# -*- coding: utf-8 -*-
"""
Created on Thu May  3 13:43:57 2018

@author: Yezhipeng
"""
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import operator

def file2matrix(filename):
    """
    读取数据
    filename：文件路径
    """
    fr = open(filename)
    numberOfLine = len(fr.readlines())
    returnMat = zeros((numberOfLine, 3)) # 创建空白矩阵
    classLabelVector = [] # 创建空白向量
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip() # 生成新字符串
        listFromLine = line.split('\t') # 切割
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
    线性归一化
    消除量级不同的影响
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals # 极差
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def classify0(inX, dataSet, labels, k):
    """
    计算距离
    欧式距离
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def datingClassTest(datingDataMat, datingLabels):
    """
    约会网站测试
    """
    hoRatio = 0.1 #测试集比例
    normMat, ranges, minVals = autiNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    print ('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print ("the classifier came back with: %d, the real answer is; %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
        else:
            pass
    print ("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print (errorCount)

def classifyPerson():
    """
    约会网站测试函数
    """
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels, 3)
    print ("You will probably like this person: ", resultList[classifierResult - 1])

datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
fig = plt.figure() # 画图
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

classifyPerson()

