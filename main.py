import torch
import numpy as np
import pandas as pd
import argparse
import random
import math
import os
import errno
from utils import set_seed, get_model, get_dataset, get_run_description, list_to_str, get_agg_method
from procedure import AdaDetectERM, E_value_AdaDetectERM, ConformalOCC, E_value_ConformalOCC

algo_to_object = {'AdaDetectERM': AdaDetectERM, 'E_value_AdaDetectERM': E_value_AdaDetectERM,
                  'ConformalOCC': ConformalOCC, 'E_value_ConformalOCC': E_value_ConformalOCC}
OCC_models = ['OC-SVM', 'IF']


def main(args, save=True, x_test=None, y_test=None, x_null=None):
    if not os.path.exists(args.save_path):
        try:
            os.makedirs(args.save_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    device = torch.device('cuda' if torch.cuda.is_available() and args.model == 'NN' else 'cpu')
    print("Is CUDA available? {}".format(torch.cuda.is_available()))
    args.device = device

    args_dict = dict(vars(args))
    del args_dict['device']
    if 'E_value' in args.algorithm:
        desc = args.algorithm + '_at_' + list_to_str(args.alpha_t) + '_' + args.agg_alpha_t +\
               '_n_' + str(args.n_e_value) + '_w_' + str(args.weight_metric) + ' ' + args.model
    else:
        desc = args.algorithm + ' ' + args.model
    if args.correction_type is not None:
        desc += '-' + args.correction_type
    args_dict['run-name'] = desc
    # add number of training samples - n_train - n_cal
    args_dict['actual_n_train'] = args_dict['n_train'] - args_dict['n_cal']
    # convert list to str
    args_dict['alpha_t'] = list_to_str(args.alpha_t)
    all_results = pd.DataFrame(args_dict, index=[0])

    set_seed(args.seed)
    # get dataset
    X, y, params = get_dataset(args)
    y = y.astype(float)
    # update n_features (relevant for real data)
    args.n_features = X.shape[1]
    # get model
    model = get_model(args, **params)

    # split dataset
    # test sample
    outlr, inlr = X[y == 1], X[y == 0]
    m1 = int(args.n_test * args.test_purity)
    m0 = args.n_test - m1
    if outlr.shape[0] < m1:
        raise ValueError(f'Not enough outliers samples in {args.dataset} dataset. #outliers={outlr.shape[0]} < '
                         f'{m1}')
    if inlr.shape[0] < m0 + args.n_train:
        raise ValueError(f'Not enough inliers samples in {args.dataset} dataset. #outliers={inlr.shape[0]} < '
                         f'{m0 + args.n_train}')
    test1, test0 = outlr[:m1], inlr[:m0]
    x = np.concatenate([test0, test1])
    y = np.concatenate([np.zeros(m0), np.ones(m1)])
    randomized = np.random.permutation(x.shape[0])
    x, y = x[randomized], y[randomized]
    if x_test is not None and y_test is not None:
        x, y = x_test, y_test
    # null samples
    xnull = inlr[m0:m0 + args.n_train]
    if x_null is not None:
        xnull = x_null
    randomized = np.random.permutation(xnull.shape[0])
    xnull = xnull[randomized]

    max_depth_l, max_depth_l_of_dict = [], []
    if args.random_params:
        if args.random_params_type == 'random':
            max_depth_l = random.choices(range(1, 25), k=(args.n_e_value - 1))
            max_depth_l.insert(0, args.max_depth)
        elif args.random_params_type == 'grid':
            max_depth_l = [2, 12, 20, 30, 7, 17, 5, 23, 11, 27]  # randomly sampled once from the range (1,30)
            if args.n_e_value != 10:
                f = math.ceil(args.n_e_value / 10)
                max_depth_l = max_depth_l * f
                max_depth_l = max_depth_l[:args.n_e_value]
        for d in max_depth_l:
            max_depth_l_of_dict.append({'max_depth': d})
    else:
        max_depth_l_of_dict = None

    proc = algo_to_object[args.algorithm](scoring_fn=model,
                                          split_size=1-(args.n_cal / args.n_train),
                                          correction_type=args.correction_type,
                                          storey_threshold=args.n_storey/args.d_storey,
                                          n_repetitions=args.n_e_value,
                                          alpha_t=args.alpha_t, agg_alpha_t=get_agg_method(args.agg_alpha_t),
                                          weight_metric=args.weight_metric,
                                          random_params=args.random_params,
                                          models_params=max_depth_l, #RF
                                          cv_params=max_depth_l_of_dict,
                                          )
    (rejections_indices, th, ths), values = proc.apply(x, args.alpha, xnull)  # fit + return rejection set
    if values is not None:
        values = np.concatenate([values.reshape((-1,1)), y.reshape((-1,1))], axis=1)
    print(f'Number of rejections: {len(rejections_indices)}')
    if 'E_value' in args.algorithm:
        all_results.insert(0, 'weights', [list_to_str(proc.all_weights)], allow_duplicates=False)
        if args.random_params:
            all_results.insert(0, 'model_params', [list_to_str(proc.models_params)], allow_duplicates=False)
    # analyze results
    n_false_rejections = np.sum(y[rejections_indices] == 0) if len(rejections_indices) else 0
    n_true_rejections = np.sum(y[rejections_indices] == 1) if len(rejections_indices) else 0
    fdr = n_false_rejections / len(rejections_indices) if len(rejections_indices) else 0
    tdr = n_true_rejections / m1 if m1 else 0
    all_results.insert(0, 'FDR', [fdr], allow_duplicates=False)
    all_results.insert(0, 'TDR', [tdr], allow_duplicates=False)
    all_results.insert(0, 'n_rejections', [len(rejections_indices)], allow_duplicates=False)

    if ths is not None:
        all_results.insert(0, 'thresholds', [list_to_str(ths)], allow_duplicates=False)
    if th is not None:
        all_results.insert(0, 'Rejection-threshold', [th], allow_duplicates=False)
    if save:
        all_results.to_pickle(args.save_path + get_run_description(args) + '.pkl')
        # save th and values
        if values is not None:
            from numpy import save
            # save to npy file
            save(args.save_path + get_run_description(args) + '.npy', values)
    print(f'FDR : {fdr}')
    print(f'TDR : {tdr}')
    return all_results, values, x, y, xnull, rejections_indices


def get_args(parser=None, get_parser=False):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='AdaDetectERM', choices=['AdaDetectERM',
                                                                                  'E_value_AdaDetectERM',
                                                                                  'ConformalOCC',
                                                                                  'E_value_ConformalOCC'])
    parser.add_argument('--model', type=str, default='RF', choices=['RF', 'OC-SVM', 'IF'])
    parser.add_argument('--correction_type', default=None, choices=[None, 'storey', 'quantile'])
    parser.add_argument('--n_storey', type=float, default=0.5, help='Numerator of Storey correction threshold')
    parser.add_argument('--d_storey', type=int, default=1, help='Denominator of Storey correction threshold')
    parser.add_argument('--n_e_value', type=int, default=0, help='Number of iterations (relevant for e-value methods)')
    parser.add_argument('--dataset', type=str, default='creditcard')
    parser.add_argument('--dataset_version', default=1)
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--alpha_t', type=float, nargs='+', default=[0.05])
    parser.add_argument('--agg_alpha_t', type=str, default='avg', choices=['avg'],
                        help='Method to aggregate e-values computed for different alpha_t values.')
    parser.add_argument('--weight_metric', type=str, default='t-test', choices=['uniform', 't-test', 'avg_score'],
                        help='Method to aggregate e-values computed for different repetition')
    parser.add_argument('--verbose', action='store_true')
    # artificial data parameters
    parser.add_argument('--n_features', type=int, default=10, help='Number of features')
    parser.add_argument('--rho', type=float, default=1, help='ρ - non-negative correlation coefficient for '
                                                             'artificial_example_3_1 dataset')
    parser.add_argument('--out_factor', type=float, default=1, help='Constant to multiply mu for outliers in '
                                                                    'artificial_example_3_1 dataset ')
    parser.add_argument('--mu_o', default=None, help='Signal amplitude - value of the first 5 features in the outlier gaussian'
                                                     'mean (all others are zero)')
    # n samples
    parser.add_argument('--n_test', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--test_purity', type=float, default=0.1, help='Proportion of class 1 (outliers)')
    parser.add_argument('--n_train', type=int, default=5000, help='Number of train samples (includes calibration set)')
    parser.add_argument('--n_cal', type=int, default=1000, help='Number of calibration samples')
    # model parameters
    parser.add_argument('--random_params', action='store_true', help='Draw random model parameters - only for RF.')
    parser.add_argument('--random_params_type', default='random', choices=['random', 'grid'])
    parser.add_argument('--max_depth', default=10, help='RF - The maximum depth of the tree. If None, then nodes are '
                                                        'expanded until all leaves are pure or until all leaves contain'
                                                        ' less than min_samples_split samples.')
    parser.add_argument('--max_samples', default="auto", help='IF - The number of samples to draw from X to train each '
                                                              'base estimator.')
    parser.add_argument('--n_estimators', type=int, default=100, help='IF/RF - The number of base estimators in the '
                                                                      'ensemble')
    parser.add_argument('--kernel_svm', type=str, default='rbf', help='OC-SVM - kernel type')

    if get_parser:
        return parser

    args = parser.parse_args()
    if 'E_value' in args.algorithm and args.n_e_value <= 0:
        raise ValueError('For E_value algorithm, n_e_value must be > 0.')
    if 'OCC' in args.algorithm and args.model not in OCC_models:
        raise ValueError(f'OCC algorithm must be with OCC model - {OCC_models}')
    if args.model != 'RF' and args.random_params:
        raise ValueError(f'Random parameters supported only for RF model - {args.model} model is not supported.')
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
