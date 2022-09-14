from math import log
from math import inf
from math import sqrt
from sklearn.model_selection import KFold
from numpy import array
from pandas import read_csv
import random
import ID3
import collections





if __name__ == '__main__':
    lis = [[100, [{}, [1, 1, 1], [1, 1, 1], {}, 0.96]], [80, [{}, [2, 2, 2], [2, 2, 2], {}, 0.95]], [90, [{}, [3, 3, 3], [3, 3, 3], {}, 0.92]]]


    # [dis, [tree, centroid, norm_centroid, norm_tree, acc]]

    lis.sort(key=lambda k: (k[1][4], k[0]), reverse=True)
    print(lis)





