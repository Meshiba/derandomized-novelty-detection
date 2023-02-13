######################################################################################################################
# This code was taken from https://github.com/arianemarandon/adadetect.git
#                   by "Machine learning meets false discovery rate" (Marandon et. al 2022)
#
# Additional implementations have been added to the file
######################################################################################################################

import math
import numpy as np
from scipy.stats import ttest_ind


def EmpBH(null_statistics, test_statistics, level):
    """
    Algorithm 1 of "Semi-supervised multiple testing", Roquain & Mary : faster than computing p-values and applying BH

    test_statistics: scoring function evaluated at the test sample i.e. g(X_1), ...., g(X_m)
    null_statistics: scoring function evaluated at the null sample that is used for calibration of the p-values i.e.
                    g(Z_k),...g(Z_n)
    level: nominal level 

    Return: rejection set 
    """
    n = len(null_statistics)
    rejections = []
    m = len(test_statistics)

    mixed_statistics = np.concatenate([null_statistics, test_statistics])
    sample_ind = np.concatenate([np.ones(len(null_statistics)), np.zeros(len(test_statistics))])

    sample_ind_sort = sample_ind[np.argsort(-mixed_statistics)]
    #np.argsort(-mixed_statistics) gives the order of the stats in descending order
    # sample_ind_sort sorts the 1-labels according to this order

    fdp = 1
    V = n
    K = m
    l=m+n

    while (fdp > level and K >= 1):
        l-=1
        if sample_ind_sort[l] == 1:
            V-=1
        else:
            K-=1
        fdp = (V+1)*m / ((n+1)*K) if K else 1

    test_statistics_sort_ind = np.argsort(-test_statistics)
    return test_statistics_sort_ind[:K]


def BH(pvalues, level):
    """
    Benjamini-Hochberg procedure.
    """
    rejections = []
    n = len(pvalues)
    pvalues_sort_ind = np.argsort(pvalues)
    pvalues_sort = np.sort(pvalues) #p(1) < p(2) < .... < p(n)

    comp = pvalues_sort <= (level* np.arange(1,n+1)/n)
    #get first location i0 at which p(k) <= level * k / n
    comp = comp[::-1]
    comp_true_ind = np.nonzero(comp)[0]
    i0 = comp_true_ind[0] if comp_true_ind.size > 0 else n
    nb_rej = n - i0
    threshold = pvalues[pvalues_sort_ind[nb_rej - 1]]
    return pvalues_sort_ind[:nb_rej], threshold


def adaptiveEmpBH(null_statistics, test_statistics, level, correction_type, storey_threshold=0.5):
    pvalues = np.array([compute_pvalue(x, null_statistics) for x in test_statistics])

    if correction_type == "storey": 
        null_prop= storey_estimator(pvalues=pvalues, threshold=storey_threshold)
    elif correction_type == "quantile":
        null_prop= quantile_estimator(pvalues=pvalues, k0=len(pvalues)//2)
    else:
        raise ValueError("correction_type is mis-specified")

    lvl_corr = level/null_prop
 
    return BH(pvalues=pvalues, level=lvl_corr), pvalues


def compute_pvalue(test_statistic, null_statistics):
    return (1 + np.sum(null_statistics >= test_statistic)) / (len(null_statistics)+1)


def storey_estimator(pvalues, threshold): 
    return (1 + np.sum(pvalues >= threshold))/ (len(pvalues)*(1-threshold)) 


def quantile_estimator(pvalues, k0):  # eg k0=m/2
    m = len(pvalues)
    pvalues_sorted = np.sort(pvalues)
    return (m-k0+1)/ (m*(1-pvalues_sorted[k0]))


def eBH(evalues, level):
    """
    E-values Benjamini-Hochberg procedure.
    """
    n = len(evalues)
    evalues_sort_ind = np.argsort(evalues)[::-1]
    evalues_sort = np.sort(evalues)[::-1] #e(1) > e(2) > .... > e(n)

    comp = evalues_sort >= (n / (level * np.arange(1,n+1)))
    # get first (largest) location i0 at which e(k) >= n / level * k
    comp = comp[::-1]
    comp_true_ind = np.nonzero(comp)[0]
    i0 = comp_true_ind[0] if comp_true_ind.size > 0 else n
    nb_rej = n - i0
    threshold = evalues[evalues_sort_ind[nb_rej - 1]]
    return evalues_sort_ind[:nb_rej], threshold


def compute_evalue(test_statistics, null_statistics, t):
    denominator = (1 + np.sum(null_statistics >= t)) / (1 + null_statistics.shape[0])
    evalues = ((test_statistics >= t).astype(int) / denominator)
    return evalues


def compute_threshold(test_statistics, null_statistics, level):
    n, m = len(null_statistics), len(test_statistics)
    mixed_statistics = np.concatenate([null_statistics, test_statistics])
    sample_ind = np.concatenate([np.ones(len(null_statistics)), np.zeros(len(test_statistics))])

    sample_ind_sort = sample_ind[np.argsort(-mixed_statistics)]

    fdp = 1
    V = n
    K = m
    l = m + n

    while (fdp > level and K >= 1):
        l -= 1
        if sample_ind_sort[l] == 1:
            V -= 1
        else:
            K -= 1
        fdp = (V + 1) * m / ((n + 1) * K) if K else 1

    test_statistics_sort_ind = np.argsort(-test_statistics)
    return test_statistics[test_statistics_sort_ind[K-1]]


def get_weight(test_scores, scores_null, weight_metric, trim_prec=0.1):
    if weight_metric == 'uniform':
        return 1
    elif weight_metric == 't-test' or weight_metric == 'avg_score':
        # sort and trim
        scores = np.concatenate((test_scores, scores_null), axis=0)
        sorted_scores = np.sort(scores)
        large_group = sorted_scores[-1*math.ceil(trim_prec*test_scores.shape[0]):]
        small_group = sorted_scores[:-1*math.ceil(trim_prec*test_scores.shape[0])]
        if weight_metric == 'avg_score':
            w = np.mean(small_group)  # can be negative
            weight = np.exp(-1 * w)
            return weight
        elif weight_metric == 't-test':
            statistic, _ = ttest_ind(small_group, large_group)
            return np.abs(statistic)
    else:
        raise ValueError(f'The following weight metric is not supported - {weight_metric}')

