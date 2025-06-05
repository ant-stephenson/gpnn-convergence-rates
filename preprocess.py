import numpy as np
from typing import Tuple

from gpprediction.utils import inv


def get_xy_preprocess_tranforms(x_train: np.ndarray, y_train: np.ndarray, xpreprocess: str, train_data_size: int, input_dim: int) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """derive quantities that can be used to preprocess (x,y) training values (in the case of x this might optionally either be none, prewhitening or axis-rescaling)
    derive mean and sd of y train values, mean of x train components, if 'prewhitenting then inverse covar of x train cpts, and if axis-rescaling then
    diag matrix with 1/(sd x-cpt) along the diag

    Args:
        x_train (np.ndarray): covariates
        y_train (np.ndarray): observations
        xpreprocess (str): method e.g. whitening)
        train_data_size (int): length of training set (n)
        input_dim (int): d

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]: y avg, y std, x avg,
        whitening/etc matrix (for covariates)
    """
    m_x = np.average(x_train, axis=0)
    m_y = np.average(y_train)
    sd_y = np.std(y_train, dtype=np.float64)
    if (xpreprocess == 'axis_rescale'):
        sd_x = np.std(x_train, axis=0, dtype=np.float64)
        prep_mat_x = np.diag(1.0/sd_x)
    if (xpreprocess == 'whiten'):
        cov_x = np.dot(np.transpose(x_train - m_x),
                       (x_train - m_x))/float(train_data_size)
        U, S, V = np.linalg.svd(cov_x)  # Singular Value Decomposition
        epsilon = 1e-5
        prep_mat_x = np.diag(1.0 / np.sqrt(S + epsilon)).dot(U.T)
    if (xpreprocess == 'none'):
        prep_mat_x = np.identity(input_dim, dtype=np.float64)
    prep_mat_x /= float(input_dim) ** 0.5
    ans = m_y, sd_y, m_x, prep_mat_x
    return (ans)


def preprocess_x_vals(x_vals: np.ndarray, m_x: np.ndarray, prep_mat_x: np.ndarray) -> np.ndarray:
    """ Applies preprocessing method to covariates. 

    Args:
        x_vals (np.ndarray): Input covariates
        m_x (np.ndarray): mean of covariates
        prep_mat_x (np.ndarray): whitening/etc matrix

    Returns:
        np.ndarray: transformed covariates
    """
    x_vals = x_vals - m_x
    x_vals = np.transpose(prep_mat_x.dot(np.transpose(x_vals)))
    return (x_vals)


def preprocess_y_vals(y_vals: np.ndarray, m_y: float, sd_y: float) -> np.ndarray:
    """ Apply preprocessing transform to observations (y)

    Args:
        y_vals (np.ndarray): observations
        m_y (float): mean of y
        sd_y (float): std of y

    Returns:
        np.ndarray: transformed y vector
    """
    y_vals = y_vals - m_y
    y_vals = y_vals/sd_y
    return (y_vals)


def JL_dim_reduction(x_vals, rng, p=10, eps=None):
    n, d = x_vals.shape
    if eps is not None:
        p = int(1/eps**2)
    S = rng.standard_normal((d, p)) / np.sqrt(p)
    return x_vals @ S


def rescale_dataset_noise(data: np.ndarray, new_nv: float, rng) -> np.ndarray:
    """ Rescales noise and outputscale of data. Assumes nv+ks = 1 and that
    original nv = 0.008 TODO: add input

    Args:
        data (np.ndarray): [X y]
        new_nv (float): noise var of output y
        rng (_type_): Random number generator (np)

    Returns:
        np.ndarray: [X ynew]
    """
    n = data.shape[0]
    orig_nv = 0.008
    orig_ks = 1-orig_nv
    eta = (new_nv-orig_nv)/(1-orig_nv)
    xi = rng.standard_normal((n,)) * np.sqrt(np.abs(eta))
    y1 = np.sqrt(1-eta) * data[:, -1] + np.sign(eta) * xi
    data[:, -1] = y1
    return data


def add_noisy_dims(xy_data: np.ndarray, d2: int, rng, method="proj") -> np.ndarray:
    """ Add new dimensions of noise, or transform data into larger space with noise

    Args:
        xy_data (np.ndarray): [X y]
        d2 (int): dimension of output data (X)
        rng (_type_): RNG
        method (str, optional): Method to use. Defaults to "proj".

    Returns:
        np.ndarray: [Xnew y]
    """
    if d2 == xy_data.shape[1]-1:
        return xy_data
    x_data = xy_data[:, :-1]
    new_x_data = add_noisy_dimsx(x_data, d2, rng, method=method)
    new_data = np.column_stack([new_x_data, xy_data[:, -1]])
    return new_data


def add_noisy_dimsx(data: np.ndarray, d2: int, rng, method="proj") -> np.ndarray:
    """ Apply noise transform (append, or projection) to covariates

    Args:
        data (np.ndarray): X data
        d2 (int): output dim.
        rng (_type_): _description_
        method (str, optional): Method to use. Defaults to "proj".

    Returns:
        np.ndarray: Xnew - noisy covariates with dim. d2>=d
    """
    n, d = data.shape
    if method == "proj":
        projection_mat = rng.standard_normal((d, d2)) / np.sqrt(d2)
        new_data = data @ projection_mat
    elif method == "proj2":
        R = rng.standard_normal((d, d2)) / np.sqrt(d2)
        projection_mat = R @ inv(R.T @ R)
        new_data = data @ projection_mat
    elif method == "noise":
        new_data = np.column_stack(
            [data, rng.standard_normal((n, d2-d)) / np.sqrt(d2-d)])
    elif method == "noise2":
        new_data = np.column_stack(
            [data, rng.standard_normal((n, d2-d)) / np.sqrt(d2-d)])
        mix_mat = rng.standard_normal((d2, d2)) / np.sqrt(d2)
        new_data = new_data @ mix_mat
    return new_data
