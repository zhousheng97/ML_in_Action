#!usr/bin/python
# -*- coding: utf-8 -*-
from math import log
import operator
import matplotlib.pyplot as plt
import treeplot

'''
学习从一堆原始数据构造决策树：
1.从数据集构造决策树
2.度量算法成功率（信息增益）
3.使用递归建立分类器
4.绘制决策树

3.1决策树的构造：
3.1.1 如何使用信息论划分数据集
3.1.2 编写代码将理论应用于实际数据集
3.1.3 编写代码构建决策树

问题：当前数据集上哪个特征在划分数据集时起决定作用？
解决办法：为了找到决定性的特征，划分出最好的结果，必须评估每个特征

评估方法：
划分数据集的大原则是：将无序的数据变得更加有序。
本章采用ID3算法划分数据集
ID3算法将信息增益做贪心算法来划分算法，在划分数据集的前后信息发生的变化称为信息增益，总是挑选的信息增益最大的特征来划分数据，使得数据更加有序

'''

# 创建数据集
def createDataset():
    dataset = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataset,labels

# 计算信息增益

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  #计算数据集中实例的总数
    labelCounts = {}   #创建数据字典存储 键：类别，值：类别的个数
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:  #使用所有类标签的发生频率计算类别出现的概率
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)   #计算所有类别的香农熵之和
    return shannonEnt

# 划分数据集：
# 度量划分数据集的熵，以便判断当前是否正确地划分了数据集
def splitDataSet(dataSet, axis, value):
    #程序清单3-2的代码使用了三个输入参数：待划分的数据集、划分数据集的特征、需要返回的特征的值。
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            '''
            axis=1，value=0时，
            以第1个特征作为判断，若第1个特征值==value，
            则输出当前数据样本的其他特征的值存入reducedFeatVec
            '''
            #当按照某个特征划分数据集时，需要将所有符合要求的元素抽取出来。
            reducedFeatVec = featVec[:axis]
            # extend函数是将featVec的其他特征的值存入列表reducedFeatVec中
            reducedFeatVec.extend(featVec[axis+1:])
            # append函数是将元素reducedFeatVec作为列表插入列表retDataSet中
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 选择最好的数据集划分方式
# 函数实现：选取特征值，划分数据集，计算得到最 好的划分数据集的特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  #计算每个数据样本共有多少个特征
    baseEntropy = calcShannonEnt(dataSet)  #计算原始数据集的熵
    bestInfoGain = 0.0; bestFeature = -1   #初始化最好的信息增益、最好的用于划分数据集的特征下表
    for i in range(numFeatures): #遍历全部特征，计算每个特征的信息增益并选择最好的
        '''
        featList = [example[i] for example in dataSet]
        该语句是将列表featList取一列元素数据的方法
        先提取一行dataSet的数据，然后取该行数据的第“ i ”位元素
        然后遍历每一行，最后获得一整列数据，变为一个列表list
        如：
        dataSet=[[1,1,'yes'],
         [1,1,'yes'],
         [1,0,'no'],
         [0,1,'no'],
         [0,1,'no']]
        featList1=[example[0] for example in dataSet]
        print('featList1 = \n',featList1)
        print(type(featList1))
        >>> featList1 = 
        >>> [1, 1, 1, 0, 0]
        >>> <class 'list'>
        
        featList2=[example[-1] for example in dataSet]
        print('featList2 = \n',featList2)
        >>> featList2 = 
        >>> ['yes', 'yes', 'no', 'no', 'no']
        '''
        featList = [example[i] for example in dataSet]  #创建对应特征的属性值列表
        uniqueVals = set(featList)  #得到当前特征下的不同属性值集合
        newEntropy = 0.0   #初始化熵

        #计算对样本集进行划分所获得的信息增益
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))  #选择当前属性对样本集进行划分的概率
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        #获取信息增益最大的特征
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 递归构建决策树
# 分类叶子节点：多数表决的方法
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),\
                              key=operator.itemgetter((1),reverse=True))

    return sortedClassCount[0][0]

# 创建树
'''
基本原理是：
给定原始数据集，基于最好的特征划分数据集，根节点为该特征，
分支为对应的不同的属性值，有多少种属性值，就有多少个分支。
不同分支指向的节点是去除分支上属性值后的数据子集。
节点上的数据子集可以再次依照相同的方式被划分,所以此处使用递归的思想。
递归结束的条件：程序遍历完所有可用于划分的特征,或者每个分支下所有实例属于相同的分类。
'''
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #count函数:统计列表classList中classList[0]元素出现的次数
    #print classList
    #>>>['yes', 'yes', 'no', 'no', 'no']
    #print classList[0]
    #>>>yes

    #递归结束条件1：数据集类别属于同一类
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #递归结束条件2：遍历完所有特征时，返回出现次数最多的特征
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #选择最好的特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    #使用最好的特征初始化决策树
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)   #最好特征下的属性集合
    for value in uniqueVals:
        subLabels = labels[:]
        # 使用划分后的数据子集 和 剩下的特征列表  递归构建字典
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

# 绘制决策树图(略)

#测试算法：
# 使用决策树执行分类,即：建好了树之后，拿一条新的数据进行分类
# 使用决策树的分类函数:每次使用分类器时，必须重新构造决策树
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]  #当前绘制的决策树图中树的根节点的特征名称
    secondDict = inputTree[firstStr]  #根节点的所有子节点
    featIndex = featLabels.index(firstStr)  #找到根节点特征对应的下标
    key = testVec[featIndex]  #找出待测数据的特征值
    valueOfFeat = secondDict[key]  #拿这个特征值在根节点的子节点中查找，判断是不是叶节点

    # 如果不是叶节点，递归到下一层节点； 如果是叶节点：确定待测数据的分类
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

#决策树的存储：
# 不希望每次拿新数据进行分类前，都构造一遍决策树。所以得想个办法把它存到硬盘上
# 用pickle这个模块，用来将对象持久化
# 就是把建好的决策树（dict类型）存进或者取出文件
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

if __name__ == "__main__":
    myDat,labels = createDataset()
    # print myDat
    # print splitDataSet(myDat,0,0)
    # print chooseBestFeatureToSplit(myDat)
    myTree = treeplot.retrieveTree(0)
    # print myTree
    # print classify(myTree,labels,[1,0])

    storeTree(myTree, 'storage.txt')
    # print grabTree('storage.txt')

    fp = open('lenses.txt')
    lensesDataSet = [line.strip().split('\t') for line in fp.readlines()]
    lensesFeatNames = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lensesDataSet, lensesFeatNames)
    print lensesTree
    treeplot.createPlot(lensesTree)
