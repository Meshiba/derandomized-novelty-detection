######################################################################################################################
# This code was taken from https://github.com/arianemarandon/adadetect.git
#                   by "Machine learning meets false discovery rate" (Marandon et. al 2022)
#
# Additional implementations have been added to the file
######################################################################################################################

import numpy as np
from functools import reduce
from sklearn.ensemble import RandomForestClassifier

from algo import BH, EmpBH, adaptiveEmpBH, compute_evalue, compute_threshold, eBH, quantile_estimator, storey_estimator\
                 , compute_pvalue, get_weight


class AdaDetectBase(object):
    """
    Baseline: AdaDetect

    """

    def __init__(self, correction_type=None, storey_threshold=0.5):
        """
        correction_type: if 'storey'/'quantile', uses the adaptive AdaDetect procedure with storey/quantile correction
        """
        self.null_statistics = None
        self.test_statistics = None 
        self.correction_type = correction_type
        self.storey_threshold = storey_threshold

    def fit(self, x, level, xnull):
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: none. Sets the values for <null_statistics> / <test_statistics>. 
        """
        # This part depends specifically on the type of AdaDetect procedure:
        # whether the scoring function g is learned via density estimation, or an ERM approach (PU classification)
        # Thus, it is coded in separate AdaDetectBase objects, see below.

        pass

    def apply(self, x, level, xnull): 
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: rejection set of AdaDetect with scoring function g learned from <x> and <xnull> as per .fit(). 
        """ 
        self.fit(x, level, xnull)
        if self.correction_type is not None:
            return adaptiveEmpBH(self.null_statistics, self.test_statistics, level=level,
                                 correction_type=self.correction_type, storey_threshold=self.storey_threshold)
        else: 
            return (EmpBH(self.null_statistics, self.test_statistics, level=level), None,
                    None), None


class AdaDetectERM(AdaDetectBase):
    """
    AdaDetect procedure where the scoring function is learned by an ERM approach. 
    """
    def __init__(self, scoring_fn, split_size=0.5, correction_type=None, storey_threshold=0.5, **kwargs):
        AdaDetectBase.__init__(self, correction_type, storey_threshold)
        """
        scoring_fn: A class (estimator) that must have a .fit() and a .predict_proba() or .decision_function() method, 
        e.g. sklearn's LogisticRegression() 
                            The .fit() method takes as input a (training) data sample of observations AND labels 
                            <x_train, y_train> and may set/modify some parameters of scoring_fn
                            The .predict_proba() method takes as input a (test) data sample and should return the a 
                            posteriori class probabilities (estimates) for each element
        
        split_size: proportion of the part of the NTS used for fitting g i.e. k/n with the notations of the paper
        """

        self.scoring_fn = scoring_fn
        self.split_size = split_size

    def fit(self, x, level, xnull):
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: none. Sets the values for <null_statistics> / <test_statistics> properties (which are properties of any
        AdaDetectBase object)
        """
        m = len(x)
        n = len(xnull)

        n_null_train = int(self.split_size * n) 
        xnull_train = xnull[:n_null_train]
        xnull_calib = xnull[n_null_train:]

        x_mix_train = np.concatenate([x, xnull_calib])

        # fit a classifier using xnull_train and x_mix_train
        x_train = np.concatenate([xnull_train, x_mix_train])
        y_train = np.concatenate([np.zeros(len(xnull_train)), np.ones(len(x_mix_train))])
        
        self.scoring_fn.fit(x_train, y_train)

        # compute scores 
        methods_list = ["predict_proba", "decision_function"]
        prediction_method = [getattr(self.scoring_fn, method, None) for method in methods_list]
        prediction_method = reduce(lambda x, y: x or y, prediction_method)

        self.null_statistics = prediction_method(xnull_calib)
        self.test_statistics = prediction_method(x)

        if self.null_statistics.ndim != 1:
            self.null_statistics = self.null_statistics[:, 1]
            self.test_statistics = self.test_statistics[:, 1]


# ---------------------------------------------------------------------------------------
class E_value_AdaDetectERM(AdaDetectERM):
    def __init__(self, scoring_fn, split_size=0.5, correction_type=None, storey_threshold=0.5, n_repetitions=10,
                 alpha_t=[0.1], agg_alpha_t=np.squeeze, weight_metric='uniform',
                 random_params=False, models_params=[], **kwargs):
        AdaDetectERM.__init__(self, scoring_fn=scoring_fn, split_size=split_size, correction_type=correction_type,
                              storey_threshold=storey_threshold)
        self.n_repetitions = n_repetitions
        self.alpha_t = alpha_t
        self.agg_alpha_t = agg_alpha_t
        self.weight_metric = weight_metric
        self.random_params = random_params
        self.all_weights = []
        self.models_params = models_params
        if correction_type is not None:
            raise ValueError('Correction with E-AdaDetect is not supported.')

    def apply(self, x, level, xnull):
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: rejection set of AdaDetect with scoring function g learned from <x> and <xnull> as per .fit().
        """
        ths = []
        evalues_ = np.zeros(x.shape[0])
        sum_weights = 0
        for i in range(self.n_repetitions):
            xnull_ = np.random.permutation(xnull)
            if i > 0 and self.random_params:  # currently implemented only for RF model
                n_estimators = 100
                max_depth = self.models_params[i]
                curr_model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
                self.scoring_fn = curr_model
            self.fit(x, level, xnull_)
            evalues_t_all = None
            for i, alpha_t in enumerate(self.alpha_t):
                t = compute_threshold(self.test_statistics, self.null_statistics, alpha_t)
                ths.append(t)
                evalues_t = compute_evalue(self.test_statistics, self.null_statistics, t)
                evalues_t_all = np.concatenate([evalues_t_all, evalues_t.reshape((-1, 1))], axis=1) if i != 0 \
                    else evalues_t.reshape((-1, 1))
            # aggregate evalues for different alpha_t values
            evalues = self.agg_alpha_t(evalues_t_all)
            # compute weight
            curr_w = get_weight(self.test_statistics, self.null_statistics, self.weight_metric)
            evalues_ += curr_w * evalues
            sum_weights += curr_w
            self.all_weights.append(curr_w)
        evalues_ /= sum_weights
        # apply eBH
        return (*eBH(evalues_, level), ths), evalues_


class ConformalOCC(object):
    """
    Conformal OCC procedure
    """

    def __init__(self, scoring_fn, split_size=0.5, correction_type=None, storey_threshold=0.5,
                 **kwargs):
        """
        scoring_fn: A class (estimator) that must have a .fit() and a .predict_proba() or .decision_function() method,
        e.g. sklearn's LogisticRegression()
                The .fit() method takes as input a (training) data sample of observations
                <x_train> and may set/modify some parameters of scoring_fn
                The .predict_proba() method takes as input a (test) data sample and should return the a
                posteriori class probabilities (estimates) for each element

        split_size: proportion of the part of the NTS used for fitting g
        """

        self.null_statistics = None
        self.test_statistics = None
        self.correction_type = correction_type
        self.storey_threshold = storey_threshold
        self.scoring_fn = scoring_fn
        self.split_size = split_size

    def fit(self, x, level, xnull):
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: none. Sets the values for <null_statistics> / <test_statistics> properties (which are properties of any
        AdaDetectBase object)
        """
        m = len(x)
        n = len(xnull)

        n_null_train = int(self.split_size * n)
        xnull_train = xnull[:n_null_train]
        xnull_calib = xnull[n_null_train:]

        # fit a classifier using xnull_train
        self.scoring_fn.fit(xnull_train)

        # compute scores
        methods_list = ["predict_proba", "decision_function"]
        prediction_method = [getattr(self.scoring_fn, method, None) for method in methods_list]
        prediction_method = reduce(lambda x, y: x or y, prediction_method)

        self.null_statistics = -1 * prediction_method(xnull_calib)
        self.test_statistics = -1 * prediction_method(x)

        if self.null_statistics.ndim != 1:
            self.null_statistics = self.null_statistics[:, 1]
            self.test_statistics = self.test_statistics[:, 1]

    def apply(self, x, level, xnull):
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: rejection set of AdaDetect with scoring function g learned from <x> and <xnull> as per .fit().
        """
        self.fit(x, level, xnull)
        if self.correction_type is not None:
            return adaptiveEmpBH(self.null_statistics, self.test_statistics, level=level,
                                 correction_type=self.correction_type, storey_threshold=self.storey_threshold)
        else:
            pvalues = np.array([compute_pvalue(x, self.null_statistics) for x in self.test_statistics])
            return (*BH(pvalues, level=level), None), pvalues


class E_value_ConformalOCC(ConformalOCC):
    """
    Conformal OCC procedure
    """

    def __init__(self, scoring_fn, split_size=0.5, correction_type=None, storey_threshold=0.5, n_repetitions=10,
                 alpha_t=[0.1], agg_alpha_t=np.squeeze, weight_metric='uniform',
                 random_params=False, models_params=[], **kwargs):
        ConformalOCC.__init__(self, scoring_fn=scoring_fn, split_size=split_size, correction_type=correction_type,
                              storey_threshold=storey_threshold)
        self.n_repetitions = n_repetitions
        self.alpha_t = alpha_t
        self.agg_alpha_t = agg_alpha_t
        self.weight_metric = weight_metric
        self.random_params = random_params
        self.all_weights = []
        self.models_params = models_params
        if correction_type is not None:
            raise ValueError('Correction with E-OC-Conformal is not supported.')

    def apply(self, x, level, xnull):
        """
        x: test sample
        xnull: NTS
        level: nominal level

        Return: rejection set
        """
        ths = []
        sum_weights = 0
        evalues_ = np.zeros(x.shape[0])
        for i in range(self.n_repetitions):
            xnull_ = np.random.permutation(xnull)
            if i > 0 and self.random_params:  # currently implemented only for RF model
                n_estimators = 100
                max_depth = self.models_params[i]
                curr_model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
                self.scoring_fn = curr_model
            self.fit(x, level, xnull_)
            evalues_t_all = None
            for i, alpha_t in enumerate(self.alpha_t):
                t = compute_threshold(self.test_statistics, self.null_statistics, alpha_t)
                ths.append(t)
                evalues_t = compute_evalue(self.test_statistics, self.null_statistics, t)
                evalues_t_all = np.concatenate([evalues_t_all, evalues_t.reshape((-1, 1))], axis=1) if i != 0 \
                    else evalues_t.reshape((-1, 1))
            # aggregate evalues for different alpha_t values
            evalues = self.agg_alpha_t(evalues_t_all)
            # compute weight
            curr_w = get_weight(self.test_statistics, self.null_statistics, self.weight_metric)
            evalues_ += curr_w * evalues
            sum_weights += curr_w
            self.all_weights.append(curr_w)
        evalues_ /= sum_weights
        # apply eBH
        return (*eBH(evalues_, level), ths), evalues_
