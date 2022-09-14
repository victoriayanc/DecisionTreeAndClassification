from math import log
import ID3
from math import inf
from sklearn.model_selection import KFold
from numpy import array
from pandas import read_csv
import random


class Samples:
    def __init__(self, samplesList):
        self.list = samplesList
        self.count = len(samplesList)
        if self.count > 0:
            self.num_features = len(samplesList[0]) - 1
        self.m = 0
        self.b = 0
        for x in samplesList:
            if x[0] == 'M':
                self.m += 1
            if x[0] == 'B':
                self.b += 1

        if self.m >= self.b:
            self.majorityTag = 'M'
            self.majorityCount = self.m
        else:
            self.majorityTag = 'B'
            self.majorityCount = self.b


def maxIG(samples, num_features):
    listIG = []
    for i in range(num_features):
        fis = ID3.partFeature(i + 1, samples)
        max_ig = 0
        max_fi = fis[0]
        for fi in fis:
            ig = calcIG(fi, samples)
            if ig > max_ig:
                max_ig = ig
                max_fi = fi

        listIG.append([max_ig, max_fi])
    maxFeature = max(listIG, key=lambda k: (k[0], k[1][0]))
    return maxFeature[1]


def calcIG(feature, samples):
    ig = calcEntropy(samples)
    if samples.count > 0:
        s1, s2 = ID3.parseAccordingFeature(samples, feature)
        e1 = calcEntropy(s1)
        e2 = calcEntropy(s2)
        ig += -(((s1.count / samples.count) * e1) + ((s2.count / samples.count) * e2))
    return ig


def calcEntropy(samples):
    if samples.count == 0:
        return 0
    if samples.majorityTag == 'M':
        pm = 10 * samples.m / (samples.m * 10 + samples.b)
        if 10 * samples.b <= samples.m * 10 + samples.b:
            pb = 10 * samples.b / (samples.m * 10 + samples.b)
        else:
            pb = samples.b / (samples.m * 10 + samples.b)
    else:
        pm = samples.m / (samples.m + samples.b * 10)
        pb = 10 * samples.b / (samples.m + samples.b * 10)

    entropy = 0
    if pm != 0:
        entropy += -(pm * log(pm, 2))
    if pb != 0:
        entropy += -(pb * log(pb, 2))
    return entropy


def lossID3(samples, res):
    fp = 0
    fn = 0
    for x in samples.list:
        classify = ID3.DTClassify(x, res)
        if x[0] == 'B' and classify == 'M':
            fp += 1
        if x[0] == 'M' and classify == 'B':
            fn += 1

    return (0.1 * fp + fn) / samples.count


if __name__ == '__main__':
    trainList = ID3.getSamplesFromCSV("train.csv")
    train_samples = Samples(trainList)
    testList = ID3.getSamplesFromCSV("test.csv")
    test_samples = Samples(testList)

    res = ID3.train(train_samples, maxIG, 1)
    print(lossID3(test_samples, res))
