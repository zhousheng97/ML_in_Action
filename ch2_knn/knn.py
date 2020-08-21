#!usr/bin/python
# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.font_manager import FontProperties
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import operator

##创建数据集
def createDataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#实施knn算法
def classify0(inX,dataset,labels,k):
#inX 用于分类的输入向量，array类型1*2
#dataset 输入的训练样本集group，array类型4*2
#labels 标签向量，array类型1*4
#k 表示用于选择最近邻居的数目
    datasetSize = dataset.shape[0]  #shape函数，dataset的行数
    # 标签向量的矩阵数目和dataset的行数相同
    # tile函数表示在XY轴方向进行复制，（）中第一个表示Y轴复制次数，第二个表示X轴复制次数
    diffMat = tile(inX,(datasetSize,1)) - dataset
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances **0.5
    sortedDistIndicies = distances.argsort()  #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y。
    #现在我们可以看看argsort()函数的具体功能是什么：
    #x=np.array([1,4,3,-1,6,9])
    #x.argsort()
    #输出定义为y=array([3,0,2,1,4,5])


    # print datasetSize
    # print diffMat
    # print sqDiffMat
    # print sqDistances
    # print distances
    # print sortedDistIndicies

    classCount = {}  #把字典分解成列表
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #print voteIlabel
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        #print classCount
        sortedClassCount = sorted(classCount.iteritems(),
                                  key=operator.itemgetter(1),
                                  reverse=True)
        return sortedClassCount[0][0]
#sorted() 函数:https://www.runoob.com/python3/python3-func-sorted.html


#准备数据：将文本记录转换为Numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    #读取文本
    arrayOLines = fr.readlines()
    #首先要知道文本文件包含多少行
    numberOfLines = len(arrayOLines)
    #返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    #创建以零填充的矩阵，将矩阵的另一维度设置为固定值3
    returnMat = zeros((numberOfLines,3))
    #返回的分类标签向量
    classLabelVector = []
    #行的索引值
    index = 0
    #循环处理文件中的每行数据
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        #根据文本中标记喜欢的程序进行分类,,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat,classLabelVector

#可视化数据
def showdatas(datingDataMat,datingLabels):
    fig = plt.figure()
    #“111”表示“1×1网格，第一子图”
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2],
    15.0*array(datingLabels),15.0*array(datingLabels))
    plt.show()

#归一化特征值
def autoNorm(dataset):
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataset))
    m = dataset.shape[0]
    normDataSet = dataset - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#测试算法：作为完整程序验证分类器
def datingClassTest():
    filename = 'datingTestSet.txt'
    # 将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat,datingLabels = file2matrix(filename)
    # 取所有数据的百分之十
    hoRatio = 0.03
    # 数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat,ranges,minVals = autoNorm(datingDataMat)
    # 获得normMat的行数
    m = normMat.shape[0]
    # 百分之十的测试数据的个数
    numTestVecs = int(m*hoRatio)
    #分类错误计数
    errorCount = 0.0

    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResults = classify0(normMat[i,:],normMat[numTestVecs:m,:],
                                      datingLabels[numTestVecs:m],4)
        print("分类结果：%d\t真实类别：%d" % (classifierResults,datingLabels[i]))
        if classifierResults != datingLabels[i]:
            errorCount += 1.0
    print("错误率：%f%%" % (errorCount/float(numTestVecs)*100))

#使用算法：通过输入一个人的三维特征,进行分类输出
def classifyPerson():
    #输出结果
    resultList = ['讨厌','有些喜欢','非常喜欢']
    #三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    #打开的文件名
    filename = "datingTestSet.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #生成NumPy数组,测试集
    inArr = array([precentTats, ffMiles, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    #打印结果
    print("你可能%s这个人" % (resultList[classifierResult-1]))

if __name__ == "__main__":
    #创建数据集
    group, labels = createDataset()
    a = classify0([0,0], group, labels, 3)
    print a

    #读取文本文件数据，并解析文件
    filename = 'datingTestSet.txt'
    datingDataMat,datingLabels = file2matrix(filename)
    print datingDataMat
    print datingLabels

    #可视化数据
    #showdatas(datingDataMat,datingLabels)

    #归一化数据
    normDataSet,ranges,minVals = autoNorm(datingDataMat)
    print normDataSet
    print ranges
    print minVals

    #测试数据：计算错误率
    datingClassTest()

    #使用算法：判断结果
    classifyPerson()