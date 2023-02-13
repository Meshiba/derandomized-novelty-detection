from .gaussian import *


def generate_synthetic_dataset(name, n_features, n_inliers, n_outliers, args):
    if name == 'gaussian':
        return generate_gaussian_dataset(n_features, n_inliers, n_outliers, mu_o=args.mu_o)
    else:
        raise ValueError(f'synthetic dataset {name} does not support')


def get_oracle_scores(name, **params):
    if name == 'gaussian':
        return Oracle(scoring_fn=oracle_scores_gaussian, params=params)
    else:
        raise ValueError(f'synthetic dataset {name} does not support')


class Oracle:
    def __init__(self, scoring_fn, params):
        self.scoring_fn = scoring_fn
        self.params = params

    def predict_proba(self, x):
        return self.scoring_fn(x, **self.params)

    def fit(self, x, y):
        pass
