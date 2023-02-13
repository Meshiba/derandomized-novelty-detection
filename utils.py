import torch
import random
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.datasets import fetch_openml
from datasets import *


def get_run_description(args, no_seed=False):
    file_name = args.model
    # correction description
    if args.correction_type is not None:
        file_name = args.correction_type + '_' + file_name
    # e-value description
    if args.n_e_value > 0:
        file_name = args.algorithm + '_at_' + list_to_str(args.alpha_t) + '_' + args.agg_alpha_t + '_n_' +\
                    str(args.n_e_value) + '_w_' + str(args.weight_metric) + '_' + file_name
    # model description
    if args.model == 'IF':
        file_name += '_e_' + str(args.n_estimators) + '_s_' + str(args.max_samples)
    elif args.model == 'RF':
        file_name += '_e_' + str(args.n_estimators) + '_d_' + str(args.max_depth)
    elif args.model == 'OC-SVM':
        file_name += '_k_' + str(args.kernel_svm)
    # artificial data description
    if args.dataset.startswith('artificial_'):
        synthetic_dataset = args.dataset.replace('artificial_', '', 1)
        file_name += '_' + synthetic_dataset + '_d_' + str(args.n_features)
        if synthetic_dataset == 'gaussian':
            file_name += '_mu_o_' + str(args.mu_o)
    else:
        file_name += '_' + args.dataset.replace(' ', '-')

    file_name += '_' + str(args.dataset_version) + '_n_train_' + str(args.n_train) + '_n_cal_' + str(args.n_cal) \
                 + '_n_test_' + str(args.n_test) + '_' + str(args.test_purity) + '_a_' + str(args.alpha)
    if not no_seed:
        file_name += '_seed_' + str(args.seed)
    return file_name


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_model(args, **params):
    if args.model == 'RF':
        return RandomForestClassifier(max_depth=args.max_depth,
                                      n_estimators=args.n_estimators)
    elif args.model == 'IF':
        return IsolationForest(n_estimators=args.n_estimators,
                               max_samples=args.max_samples)
    elif args.model == 'OC-SVM':
        return OneClassSVM(kernel=args.kernel_svm)
    else:
        raise ValueError(f'the following model is not supported - {args.model}')


def get_dataset(args):
    if args.dataset.startswith('artificial_'):
        m1 = int(args.n_test * args.test_purity)
        m0 = args.n_test - m1
        nulls = m0 + args.n_train
        synthetic_dataset = args.dataset.replace('artificial_', '', 1)
        return generate_synthetic_dataset(name=synthetic_dataset, n_features=args.n_features, n_inliers=nulls,
                                          n_outliers=m1, args=args)
    else:
        X, y = fetch_openml(name=args.dataset, version=args.dataset_version, as_frame=False, return_X_y=True)
        # process labels
        normal_label = {'shuttle': '1', 'KDDCup99': 'normal', 'mammography': '-1'}
        if args.dataset in normal_label.keys():
            y_ = np.zeros(y.shape)
            y_[y == normal_label[args.dataset]] = 0
            y_[y != normal_label[args.dataset]] = 1
            y = y_
        randomized = np.random.permutation(X.shape[0])
        X, y = X[randomized], y[randomized]
        return X, y, {}


def get_agg_method(agg_method):
    if agg_method == 'max':
        def max_(values):
            return np.max(values, axis=-1)
        return max_
    elif agg_method == 'avg':
        def avg_(values):
            return np.mean(values, axis=-1)
        return avg_
    else:
        raise ValueError(f'The following aggregation method is not supported - {agg_method}')


def list_to_str(lst: list):
    s = ''
    for elem in lst:
        s += str(elem) + '_'
    s = s.rstrip('_')
    return s
