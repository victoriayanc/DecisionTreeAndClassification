import KNNForest
import ID3
from math import sqrt
import random
from numpy import array
from sklearn.model_selection import KFold
import collections


def calcDistance(centroid, sample):
    lis = [pow((abs(x[1] - sample[x[0][0]]) * 1/x[0][1]), 2) for x in centroid]
    return sqrt(sum(lis))


def featuresFromTree(lis, Tree):
    if not Tree[1]:
        return
    lis.append(Tree[0][0])
    l1 = featuresFromTree(lis, Tree[1][True])
    l2 = featuresFromTree(lis, Tree[1][False])
    return lis


def calcCentroid(samples, Tree):
    """centroid = []
    for i in range(samples.num_features):
        column = [x[i+1] for x in samples.list]
        new_column = [1/x if x != 0 else 0 for x in column]
        centroid.append(len(new_column) / sum(new_column))
    return centroid"""

    lis = []
    lis = featuresFromTree(lis, Tree)
    res = dict(collections.Counter(lis))
    lis = list(res.items())
    lis.sort(key=lambda k: k[1], reverse=True)

    centroid = []

    for i in lis:
        column = [x[i[0]] for x in samples.list]
        centroid.append([i, sum(column) / len(column)])
    # [index, avg]
    return centroid


def train(samples, N, p):
    if samples.count == 0:
        return

    trees_list = []
    for i in range(N):
        lis = random.sample(samples.list, int(p * samples.count))

        samples_test_list = [x for x in samples.list if x not in lis]
        samples_test = ID3.Samples(samples_test_list)
        #minmax = findMinMaxInColumns(lis)
        #norm_samples_test = ID3.Samples(normalized_list(samples_test_list, minmax))

        rand_samples = ID3.Samples(lis)
        #norm_rand_samples = ID3.Samples(normalized_list(lis, minmax))

        result_tree = ID3.train(rand_samples, ID3.maxIG, 1)
        #norm_result_tree = ID3.train(norm_rand_samples, ID3.maxIG, 1)
        norm_result_tree = []
        acc = ID3.predict(samples_test, result_tree)

        centroid = KNNForest.calcCentroid(rand_samples)
        norm_centroid = calcCentroid(rand_samples, result_tree)

        trees_list.append([result_tree, centroid, norm_centroid, norm_result_tree, acc])
    return trees_list


def classify(obj, K, trees_list, norm):
    #[dis, [tree, centroid, norm_centroid, norm_tree, acc]]
    if norm:
        distances_list = [[calcDistance(x[2], obj), x] for x in trees_list]
    else:
        distances_list = [[KNNForest.calcDistance(x[1], obj), x] for x in trees_list]

    if norm:
        distances_list.sort(key=lambda k: (k[1][4], k[0]), reverse=True)
    else:
        distances_list.sort(key=lambda k: k[0])

    k_distances_list = [distances_list[i] for i in range(K)]
    m = 0
    b = 0
    """if norm:
        for x in k_distances_list:
            if ID3.DTClassify(obj, x[1][3]) == 'M':
                m += 1
            else:
                b += 1"""
    if norm:
        for x in k_distances_list:
            if ID3.DTClassify(obj, x[1][0]) == 'M':
                m += 1
            else:
                b += 1
    else:
        for x in k_distances_list:
            if ID3.DTClassify(obj, x[1][0]) == 'M':
                m += 1
            else:
                b += 1

    if b >= m:
        return 'B'
    else:
        return 'M'


def predict(samples, K, trees_list, norm):
    countYes = 0
    for x in samples.list:
        if classify(x, K, trees_list, norm) == x[0]:
            countYes += 1
    return countYes / samples.count


def findMinMaxInColumns(samples_list):
    minmax = []
    num_features = len(samples_list[0]) - 1
    for i in range(num_features):
        column = [x[i + 1] for x in samples_list]
        min_column = min(column)
        max_column = max(column)
        minmax.append([min_column, max_column])
    return minmax


def normalized_list(samples_list, minmax):
    num_features = len(samples_list[0]) - 1
    return [[x[i] if x[i] == 'M' or x[i] == 'B' else
                       (x[i] - minmax[i-1][0])/(minmax[i-1][1]-minmax[i-1][0])
                       for i in range(num_features+1)] for x in samples_list]


if __name__ == '__main__':
    trainList = ID3.getSamplesFromCSV("train.csv")
    train_samples = ID3.Samples(trainList)
    testList = ID3.getSamplesFromCSV("test.csv")
    test_samples = ID3.Samples(testList)

    results_trees = train(train_samples, 29, 0.4)
    print(predict(test_samples, 27, results_trees, True))
