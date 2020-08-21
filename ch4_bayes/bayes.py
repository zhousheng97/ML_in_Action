#!usr/bin/python
# -*- coding: utf-8 -*-
from numpy import *

'''
贝叶斯和朴素贝叶斯的最大区别就是：朴素，即条件独立，算法引入朴素贝叶斯的目的是为了降低计算量
'''

'''
优点：在数据较少的情况下仍然有效，可以处理多类别问题。
缺点：对于输入数据的准备方式较为敏感。
适用数据类型：标称型数据。
'''

'''
示例：过滤网站的恶意留言
'''
# 1.准备数据：词表转换为词向量
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

# 创建词表
def createVocabList(dataSet):
    vocabSet = set([])   #创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)   #对两个集合取并集，找出全部非重复词的集合
    return list(vocabSet)

# 词集模型|函数功能：词表转换为词向量（该模型中每个单词只算作出现一次）
def setOfWord2Vec(vocabList,inputSet):
    # 输入参数为词表和某个文档
    # 输出为文档向量，向量的每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现
    returnVec = [0]*len(vocabList)   #创建一个和词表等长的向量
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1。
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return  returnVec
# 词袋模型|函数功能：词表转换为词向量（该模型中每个单词可算出现多次）
# 如果一个词在文档中出现不止一次，这可能意味着包含该词是否出现在文档中所不能表达的某种信息
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)   #创建一个和词表等长的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return  returnVec

#2.训练算法：从词向量计算概率

#函数功能：朴素贝叶斯分类器   p(ci|w) = p(w|ci)*p(ci)/p(w)
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)   # 得到当前数据样本的文档数目
    numWords = len(trainMatrix[0])  # 得到文档矩阵中每个文档的单词个数
    pAbusive = sum(trainCategory)/float(numTrainDocs) # 计算辱骂性文档出现的概率，即p(c1)，p(c0)=1-p(c1)：文档矩阵中一共有6个文档，其中3个为辱骂性文档，概率为0.5
    #初始化p0和p1的分子分母
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)
    # p0Denom = 0
    # p1Denom = 0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 当前为辱骂性文档
            p1Num += trainMatrix[i]  # 同一侮辱性单词对应累加
            p1Denom += sum(trainMatrix[i])   # 侮辱性的总词数累加
        else:                      # 当前为正常性文档
            p0Num += trainMatrix[i] # 同一侮辱性单词对应累加
            p0Denom += sum(trainMatrix[i]) # 侮辱性的总词数累加
    # 侮辱性文档中出现侮辱性词汇的概率
    p1Vect = log(p1Num / p1Denom)
    # p1Vect = p1Num/p1Denom #p(w|c1) 其中w是每个文档的单词向量
    # # 正常性文档中出现侮辱性词汇的概率
    p0Vect = log(p0Num / p0Denom)
    # p0Vect = p0Num/p0Denom # p(w|c0)
    return p0Vect,p1Vect,pAbusive

# 3.测试算法：根据现实情况修改分类器
'''
利用贝叶斯分类器对文档进行分类时，
要计算多个概率的乘积以获得文档属于某个类别的概率，
即计算p(w0|c1)p(w1|c1)p(w2|c1)。
如果其中一个概率值为0，那么最后的乘积也为0。
为降低这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2。
'''

'''
另一个遇到的问题是下溢出，这是由于太多很小的数相乘造成的。
当计算乘积 p(w0|ci)p(w1|ci)p(w2|ci)...p(wN|ci)时，
由于大部分因子都非常小，所以程序会下溢出或者得到不正确的答案。一种解决办法是对乘积取自然对数。
'''

'''  
p(ci|w) = p(w|ci)*p(ci)/p(w)：

计算p(w|ci)，用到朴素贝叶斯假设,
将w展开为一个个独立的特征，概率计算可简化为：
p(w0,w1,...,wn|ci) = p(w0|ci)p(w1|ci)...p(wn|ci)

解释下列classifyNB函数：
如vec2classify为要分类的向量，设为[1,0,0,1,0,0,0,0,1....], p1Vec为前面计算出来的p(w|c1)
侮辱性文档中出现侮辱性词汇的概率即p(w|ci)，如[0.0526 0.05263158 0.05263158 0.05263158 0.05263158 0.05263158...]，
现在vec2classify * p1_vec就是特征单词出现的概率，求sum是因为前面计算p1_vec是log，相加就是相乘，最后在+log(pclass1),即：乘以p(ci)
也是相乘，本来还需要同时除以p(w)的，但是因为所有的计算都除，而且只是比较大小，因此可以都不除这个分母
'''

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec) + log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#函数功能：朴素贝叶斯分类函数
def testingNB():
    listOPosts,listClasses = loadDataSet()
    # 创建词表
    myVocabList = createVocabList(listOPosts)
    # 词表转换为词向量
    trainMat = []
    for poatinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, poatinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)

    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWord2Vec(myVocabList,testEntry))
    print testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)

    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWord2Vec(myVocabList,testEntry))
    print testEntry, 'classified as:', classifyNB(thisDoc,p0V,p1V,pAb)

if __name__ == "__main__":
    testingNB()

