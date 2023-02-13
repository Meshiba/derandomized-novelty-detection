import numpy as np


def generate_gaussian_dataset(n_features, n_inliers, n_outliers, mu_o):
    if mu_o is None:
        mu_o = np.sqrt(2*np.log(n_features))
    Z_inliers = np.random.normal(0, 1, (n_inliers, n_features))
    Z_outliers = np.random.normal(0, 1, (n_outliers, n_features))
    mu_outliers = np.zeros((1, n_features))
    mu_outliers[:, :5] = mu_o
    Z_outliers += mu_outliers
    Z = np.concatenate((Z_inliers, Z_outliers), axis=0)
    Y = np.concatenate((np.zeros((Z_inliers.shape[0],)), np.ones((Z_outliers.shape[0],))), axis=0)
    return Z, Y, {'mu_outliers': mu_outliers}


