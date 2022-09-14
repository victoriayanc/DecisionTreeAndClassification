from pandas import read_csv
from math import inf
from math import log
from sklearn.model_selection import KFold
from numpy import array
import matplotlib.pyplot as plt


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

        if self.m > self.b:
            self.majorityTag = 'M'
            self.majorityCount = self.m
        else:
            self.majorityTag = 'B'
            self.majorityCount = self.b


def getSamplesFromCSV(csv_name):
    kf = read_csv(csv_name)
    arr = kf.values
    new_list = [[y for y in x] for x in arr]
    return new_list


# selectFeature function
def maxIG(samples, num_features):
    listIG = []
    for i in range(num_features):
        fis = partFeature(i + 1, samples)
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


#suppose there is more than 2 samples
#index > 0
def partFeature(index, samples):
    column = list({x[index] for x in samples.list})
    if len(column) == 1:
        return [[index, column[0]]]
    column.sort()
    part = [[index, (column[i] + column[i+1]) / 2] for i in range(len(column) - 1)]
    return part


def parseAccordingFeature(samples, feature):
    samples_list_1 = []
    samples_list_2 = []
    for x in samples.list:
        (samples_list_1 if x[feature[0]] >= feature[1] else samples_list_2).append(x)
    samples1 = Samples(samples_list_1)
    samples2 = Samples(samples_list_2)
    return samples1, samples2


def calcIG(feature, samples):
    ig = calcEntropy(samples)
    if samples.count > 0:
        s1, s2 = parseAccordingFeature(samples, feature)
        e1 = calcEntropy(s1)
        e2 = calcEntropy(s2)
        ig += -(((s1.count / samples.count) * e1) + ((s2.count / samples.count) * e2))
    return ig


def calcEntropy(samples):
    if samples.count == 0:
        return 0
    pm = samples.m / samples.count
    pb = samples.b / samples.count
    entropy = 0
    if pm != 0:
        entropy += -(pm * log(pm, 2))
    if pb != 0:
        entropy += -(pb * log(pb, 2))
    return entropy


def TDIDT(samples, num_features, default, selectFeature, M):
    if samples.count == 0:
        return None, {}, default
    if samples.count < M:
        return None, {}, default

    majorityTag = samples.majorityTag
    # checks if all the samples are with the same tag:
    if samples.majorityCount == samples.count:
        return None, {}, majorityTag  # make it a leaf

    f = selectFeature(samples, num_features)
    samples1, samples2 = parseAccordingFeature(samples, f)

    subtree1 = TDIDT(samples1, num_features, majorityTag, selectFeature, M)
    subtree2 = TDIDT(samples2, num_features, majorityTag, selectFeature, M)
    subtrees = {True: subtree1, False: subtree2}
    return f, subtrees, majorityTag


def DTClassify(obj, Tree):
    if not Tree[1]:
        return Tree[2]
    if obj[Tree[0][0]] >= Tree[0][1]:
        return DTClassify(obj, Tree[1][True])
    else:
        return DTClassify(obj, Tree[1][False])


def train(samples, selectFunction, M):
    majorityTag = samples.majorityTag
    num_features = samples.num_features
    return TDIDT(samples, num_features, majorityTag, selectFunction, M)


def predict(samples, res):
    countYes = 0
    for x in samples.list:
        if DTClassify(x, res) == x[0]:
            countYes += 1
    return countYes / samples.count


def convertArrToFloatList(arr):
    return [[y if y == 'M' or y == 'B' else float(y) for y in x] for x in arr]


# for the experiment graph: run lines 176-179 (in note)
def experiment(samples):
    arr = array(samples.list)
    M = [1, 3, 5, 7, 10, 15, 20, 30, 40]
    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=316813971)
    for m in M:
        results_per_m = []
        for train_index, test_index in kf.split(arr):
            tmp_train, tmp_test = arr[train_index], arr[test_index]

            train_samples = Samples(convertArrToFloatList(tmp_train))
            result_tree = train(train_samples, maxIG, m)

            test_samples = Samples(convertArrToFloatList(tmp_test))
            results_per_m.append(predict(test_samples, result_tree))
        results.append(sum(results_per_m) / len(results_per_m))
    return array(M), array(results)


if __name__ == '__main__':
    trainList = getSamplesFromCSV("train.csv")
    train_samples = Samples(trainList)
    testList = getSamplesFromCSV("test.csv")
    test_samples = Samples(testList)

    # q. 1
    result_tree = train(train_samples, maxIG, -inf)
    print(predict(test_samples, result_tree))

    """# graph from experiment q. 3
    xpoints, ypoints = experiment(train_samples)
    plt.plot(xpoints, ypoints)
    plt.show()"""



