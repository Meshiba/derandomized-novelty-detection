from .gaussian import *


def generate_synthetic_dataset(name, n_features, n_inliers, n_outliers, args):
    if name == 'gaussian':
        return generate_gaussian_dataset(n_features, n_inliers, n_outliers, mu_o=args.mu_o)
    if name == 'gaussian_2':
        return generate_gaussian_2_dataset(n_features, n_inliers, n_outliers, mu_o=args.mu_o)
    else:
        raise ValueError(f'synthetic dataset {name} does not support')
