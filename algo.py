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


def compute_pvalue(test_statistic, null_statistics):
    return (1 + np.sum(null_statistics >= test_statistic)) / (len(null_statistics)+1)


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
        fdp = (V * m) / (n * K) if K else 1

    mixed_statistics_sort_ind = np.argsort(-mixed_statistics)
    if fdp > level:
        threshold = mixed_statistics[mixed_statistics_sort_ind[0]] + 1
    else:
        threshold = mixed_statistics[mixed_statistics_sort_ind[l-1]]
    return threshold


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


def calibrator_p_to_e(pvalues, calibrator_type, null_s, test_s, r):
    if calibrator_type == 'Shafer':
        # 1/sqrt(p) - 1
        evalues = (1 / np.sqrt(pvalues)) - 1
        return evalues
    elif calibrator_type == 'VS':
        safe_pvalues = pvalues + (pvalues == 1).astype(float)
        evalues = (pvalues <= np.exp(-1)) * -1 * np.exp(-1) / (pvalues*np.log(safe_pvalues)) + (pvalues > np.exp(-1))
        return evalues
    elif calibrator_type == 'soft-rank':
        cal_min = np.min(null_s)
        cal_max = np.max(null_s)
        tmp_w_min = np.concatenate((test_s.reshape((-1, 1)), np.ones_like(test_s.reshape((-1, 1)))*cal_min), axis=-1)
        tmp_w_max = np.concatenate((test_s.reshape((-1, 1)), np.ones_like(test_s.reshape((-1, 1)))*cal_max), axis=-1)
        L_min = np.min(tmp_w_min, axis=-1).reshape((-1, 1))
        L_max = np.max(tmp_w_max, axis=-1).reshape((-1, 1))
        # normalize statistics values
        test_stats = np.divide((test_s.reshape((-1, 1)) - L_min), (L_max - L_min))  # (n_test,)
        cal_stats = np.divide((null_s.reshape((1, -1)) - L_min), (L_max - L_min))  # (n_test, n_cal)
        # Note: min value supposed to be 0
        cal_min = np.min(cal_stats, axis=-1).reshape((-1, 1))  # (n_test,)
        tmp = np.concatenate((test_stats.reshape((-1, 1)), cal_min), axis=-1)
        L_min = np.min(tmp, axis=-1).reshape((-1, 1))  # (n_test,)
        assert (L_min == np.zeros_like(L_min)).all(), "Error: after normalize, L_min is not zeros."
        if r == 0:
            R_values = test_stats.reshape((-1, 1)) - L_min
            R_cal_values = cal_stats - L_min
        else:
            R_values = (np.exp(r*test_stats.reshape((-1, 1))) - np.exp(r*L_min)) / r
            R_cal_values = (np.exp(r*cal_stats) - np.exp(r*L_min)) / r
        evalues = (1 + null_s.shape[0]) * R_values / (np.sum(R_cal_values, axis=-1).reshape((-1,1)) + R_values)
        return evalues.reshape((-1,))
    elif calibrator_type == 'integral':
        safe_pvalues = pvalues + (pvalues == 1).astype(float)
        numerator = 1 - pvalues + pvalues*np.log(pvalues)
        demon = pvalues * np.power(-1 * np.log(safe_pvalues), 2)
        evalues = (numerator / demon) * (pvalues != 1) + 0.5 * (pvalues == 1)
        return evalues
    else:
        raise ValueError(f'The following calibrator is not supported - {calibrator_type}')


def compute_all_pvalue(test_statistic, null_statistics):
    return (1 + np.sum(null_statistics.reshape((1, -1)) >= test_statistic.reshape((-1, 1)), axis=1)) / (len(null_statistics)+1)
