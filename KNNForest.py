import ID3
from math import sqrt
import random
import CostSensitiveID3

from numpy import array
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def calcDistance(centroid, sample):
    lis = [pow(abs(centroid[i] - sample[i+1]), 2) for i in range(len(sample) - 1)]
    return sqrt(sum(lis))


def calcCentroid(samples):
    centroid = []
    for i in range(samples.num_features):
        column = [x[i+1] for x in samples.list]
        centroid.append(sum(column) / len(column))
    return centroid


def train(samples, N, p):
    if samples.count == 0:
        return

    trees_list = []
    for i in range(N):
        rand_samples = ID3.Samples(random.sample(samples.list, int(p * samples.count)))
        centroid = calcCentroid(rand_samples)
        result_tree = ID3.train(rand_samples, ID3.maxIG, 1)
        trees_list.append([result_tree, centroid])
    return trees_list


def classify(obj, K, trees_list):
    #[dis, [tree, centroid]]
    distances_list = [[calcDistance(x[1], obj), x] for x in trees_list]
    distances_list.sort(key=lambda k: k[0])
    k_distances_list = [distances_list[i] for i in range(K)]
    m = 0
    b = 0
    for x in k_distances_list:
        if ID3.DTClassify(obj, x[1][0]) == 'M':
            m += 1
        else:
            b += 1
    if b >= m:
        return 'B'
    else:
        return 'M'


def predict(samples, K, trees_list):
    countYes = 0
    for x in samples.list:
        if classify(x, K, trees_list) == x[0]:
            countYes += 1
    return countYes / samples.count


if __name__ == '__main__':
    trainList = ID3.getSamplesFromCSV("train.csv")
    train_samples = ID3.Samples(trainList)
    testList = ID3.getSamplesFromCSV("test.csv")
    test_samples = ID3.Samples(testList)

    trees_list = train(train_samples, 29, 0.4)
    print(predict(test_samples, 27, trees_list))






