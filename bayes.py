import random
import pandas as pd
import math

# 拆分训练数据集
def split_train_dataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    testSet = list(dataset)
    while len(trainSet) < trainSize :
        index = random.randrange(len(testSet))
        trainSet.append(testSet.pop(index))
    return [trainSet,testSet]

# 对于各类别分开数据 label：0/1
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

# 连续值属性计算概率密度函数 平均值/方差
def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers) - 1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(feature), stdev(feature)) for feature in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByclass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

# 计算高斯概率密度函数
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

# 计算所有连续值属性对应类别的概率 inputVector测试数值
def CalculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

# 朴素贝叶斯分类结果
def predict(summaries, inputVector):
    probabilities = CalculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb :
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = (int)(predict(summaries, testSet[i]))
        predictions.append((result))
    return predictions

# 计算正确率
def Accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1;
    return (correct/float(len(testSet)))*100.0

# 训练测试
def Bayes():
    trainDataset = pd.read_csv('../bayes_data/train_important_feature.csv', encoding='gb2312')

    train_values = trainDataset.values

    splitRatio = 0.7

    train, test = split_train_dataset(train_values, splitRatio)
    summaries = summarizeByclass(train)

    predictions = getPredictions(summaries, test)
    print(predictions)
    acc = Accuracy(test, predictions)
    print('Accuracy: ' + str(acc) + '%'+'\n')

# 朴素贝叶斯分类器
def Bayes_Classifier():
    trainDataset = pd.read_csv('../bayes_data/train_important_feature.csv', encoding='gb2312')
    testDataset = pd.read_csv('../bayes_data/test_important_feature.csv', encoding='gb2312')

    summaries = summarizeByclass(trainDataset.values)

    predictions = getPredictions(summaries, testDataset.values)
    predictionss = pd.DataFrame(predictions)
    predictionss.to_csv('../bayes_data/f_answer.csv', encoding='gb2312')
    print(predictions)

Bayes()
Bayes_Classifier()