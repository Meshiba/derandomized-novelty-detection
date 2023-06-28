import argparse
import os
import numpy as np
import pandas as pd
import random
import copy
import errno
import matplotlib.pyplot as plt
from main import get_args, main, OCC_models
from utils import set_seed, get_run_description
from utils_plot import plot_visualization


def wrapper(args):
    exp_desc = get_run_description(args, no_seed=True)
    results = pd.DataFrame({})
    values = None
    scores_cal, scores_test = None, None
    main_args = copy.deepcopy(args)
    main_args_dict = vars(main_args)
    del main_args_dict['n_seeds']
    del main_args_dict['sv_exp']
    del main_args_dict['reuse_xnull']
    del main_args_dict['plot']
    set_seed(args.seed)
    seed_list = random.sample(range(1, 999999), args.n_seeds)
    for seed in seed_list:
        curr_args = copy.deepcopy(main_args)
        curr_args_dict = vars(curr_args)
        curr_args_dict['seed'] = seed
        result, values_, _, _, _, _, cal_scores, test_scores = main(curr_args, save=False)
        results = pd.concat([results, result])
        if values_ is not None:
            values = np.concatenate([values, values_], axis=0) if values is not None else values_
        if cal_scores is not None:
            scores_cal = np.concatenate([scores_cal, cal_scores], axis=0) if scores_cal is not None else cal_scores
        if test_scores is not None:
            scores_test = np.concatenate([scores_test, test_scores], axis=0) if scores_test is not None else test_scores
    # save th and values
    from numpy import save
    if values is not None:
        # save to npy file
        save(args.save_path + exp_desc + '.npy', values)
    if not os.path.exists(args.save_path + '/scores/') and (scores_cal is not None or scores_test is not None):
        try:
            os.makedirs(args.save_path + '/scores/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if scores_cal is not None:
        save(args.save_path + '/scores/calibration_scores_' + exp_desc + '.npy', scores_cal)
    if scores_test is not None:
        save(args.save_path + '/scores/test_scores_' + exp_desc + '.npy', scores_test)
    results.to_pickle(args.save_path + exp_desc + '.pkl')


def wrapper_sv_exp(args):
    x_test, y_test, x_null = None, None, None
    exp_desc = get_run_description(args, no_seed=True)
    results = pd.DataFrame({})
    values = None
    scores_cal, scores_test = None, None
    sv = None
    all_r = None
    main_args = copy.deepcopy(args)
    main_args_dict = vars(main_args)
    del main_args_dict['n_seeds']
    del main_args_dict['sv_exp']
    del main_args_dict['reuse_xnull']
    del main_args_dict['plot']
    set_seed(args.seed)
    seed_list = random.sample(range(1, 999999), args.n_seeds)
    for seed in seed_list:
        curr_args = copy.deepcopy(main_args)
        curr_args_dict = vars(curr_args)
        curr_args_dict['seed'] = seed
        result, values_, x_test, y_test, x_null, r_set, cal_scores, test_scores = main(curr_args, save=False, x_test=x_test, y_test=y_test, x_null=x_null)
        if not args.reuse_xnull:
            x_null = None
        results = pd.concat([results, result])
        if values_ is not None:
            values = np.concatenate([values, values_], axis=0) if values is not None else values_
        if cal_scores is not None:
            scores_cal = np.concatenate([scores_cal, cal_scores], axis=0) if scores_cal is not None else cal_scores
        if test_scores is not None:
            scores_test = np.concatenate([scores_test, test_scores], axis=0) if scores_test is not None else test_scores
        curr_sv = np.zeros(x_test.shape[0])
        if len(r_set):
            curr_sv[r_set] = 1
        sv = sv + curr_sv if sv is not None else curr_sv
        if all_r is None:
            all_r = curr_sv.reshape((1, -1))
        else:
            all_r = np.concatenate([all_r, curr_sv.reshape((1, -1))], axis=0)  # shape (n_seeds, n_test)
    # compute R variance
    r_hat = np.mean(all_r, axis=0)   # shape (n_test)
    r_hat_zero_m = all_r - r_hat.reshape((1, -1))  # shape (n_seeds, n_test)
    r_hat_zero_m_2 = np.power(r_hat_zero_m, 2)  # shape (n_seeds, n_test)
    var_hat = 1/(args.n_seeds - 1) * np.sum(r_hat_zero_m_2, axis=0)  # shape (n_test)
    avg_var = np.mean(var_hat)
    results['r-variance'] = avg_var

    # save th and values
    from numpy import save
    if values is not None:
        # save to npy file
        save(args.save_path + exp_desc + '.npy', values)
    if all_r is not None:
        save(args.save_path + 'rejections_' + exp_desc + '.npy', all_r)

    if not os.path.exists(args.save_path + '/scores/') and (scores_cal is not None or scores_test is not None):
        try:
            os.makedirs(args.save_path + '/scores/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if scores_cal is not None:
        save(args.save_path + '/scores/calibration_scores_' + exp_desc + '.npy', scores_cal)
    if scores_test is not None:
        save(args.save_path + '/scores/test_scores_' + exp_desc + '.npy', scores_test)
    results.to_pickle(args.save_path + exp_desc + '.pkl')
    if args.plot:
        if args.n_features != 2:
            print('Note: Plot is available only for n_features=2.')
            return
        plot_visualization(x_null, x_test, y_test, all_r, args.algorithm, save_path=args.save_path, seed_list=seed_list)


def get_args_wrapper():
    parser = get_args(get_parser=True)
    parser.add_argument('--n_seeds', type=int, default=100, help='Number of runs. Each run correspond to different seed.')
    parser.add_argument('--sv_exp', action='store_true', help='Selection variability experiment.')
    parser.add_argument('--reuse_xnull', action='store_true', help='For selection variability experiment - reuse xnull.')
    parser.add_argument('--plot', action='store_true', help='Plot 2d scatter plot of test set and mark the rejections. '
                                                            'only relevant for fixed settings with n_features=2.')
    args = parser.parse_args()
    if ('E_value' in args.algorithm or 'Calibrator' in args.algorithm) and args.n_e_value <= 0:
        raise ValueError('For E_value/Calibrator algorithm, n_e_value must be > 0.')
    if 'OCC' in args.algorithm and args.model not in OCC_models:
        raise ValueError(f'OCC algorithm must be with OCC model - {OCC_models}')
    if (args.model != 'RF' and args.model != 'LogisticRegression') and args.random_params:
        raise ValueError(f'Random parameters supported only for RF/LogisticRegression model - {args.model} model is not supported.')
    return args


if __name__ == "__main__":
    args = get_args_wrapper()
    if args.sv_exp:
        wrapper_sv_exp(args)
    else:
        wrapper(args)

