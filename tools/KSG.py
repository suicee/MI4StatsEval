from scipy.stats import spearmanr, pearsonr
from scipy.special import digamma, gamma
import scipy.spatial as ss
from math import log, pi, exp
import numpy as np


def kraskov_mi(x, y, k=5):
    '''
        Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
        Using KSG mutual information estimator

        Input: x: 2D list of size N*d_x
        y: 2D list of size N*d_y
        k: k-nearest neighbor parameter

        Output: one number of I(X;Y)
    '''

    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    N = len(x)
    dx = len(x[0])
    dy = len(y[0])
    data = np.concatenate((x, y), axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [
        tree_xy.query(point, k + 1, p=float('inf'))[0][k] for point in data
    ]
    ans_xy = -digamma(k) + digamma(N) + (dx + dy) * log(
        2)  #2*log(N-1) - digamma(N) #+ vd(dx) + vd(dy) - vd(dx+dy)
    ans_x = digamma(N) + dx * log(2)
    ans_y = digamma(N) + dy * log(2)
    for i in range(N):
        ans_xy += (dx + dy) * log(knn_dis[i]) / N
        ans_x += -digamma(
            len(
                tree_x.query_ball_point(
                    x[i], knn_dis[i] - 1e-15, p=float('inf')))) / N + dx * log(
                        knn_dis[i]) / N
        ans_y += -digamma(
            len(
                tree_y.query_ball_point(
                    y[i], knn_dis[i] - 1e-15, p=float('inf')))) / N + dy * log(
                        knn_dis[i]) / N

    return ans_x + ans_y - ans_xy