import time
import numpy as np
import torch
from typing import Tuple

from gpprediction.estimation import map_simple_params_to_cheat, SparseParams


def evaluate_predictions(pred_mean: np.ndarray, pred_sd: np.ndarray, y_true: np.ndarray) -> Tuple[float,float,float]:
    """ Take predicted mean and variance and test observations and evaluate
    performance metrics

    Args:
        nn_pred_mean (np.ndarray): mean predictions
        nn_pred_sd (np.ndarray): var predictions
        y_true (np.ndarray): test observations

    Returns:
        Tuple[float,float,float]: MSE, NLL, CAL
    """
    predict_data_size = pred_mean.shape[0]
    mse_nn = nll_nn = mscal_nn = 0.0
    const = (2.0*np.pi)**0.5
    mscal_nn = (((y_true-pred_mean)/pred_sd)**2).sum()
    mse_nn = ((y_true-pred_mean)**2).sum()
    nll_nn = np.log(1.0/(pred_sd*const)).sum() - 0.5 * mscal_nn

    nll_nn = -nll_nn
    nll_nn /= float(predict_data_size)
    mse_nn /= float(predict_data_size)
    mscal_nn /= float(predict_data_size)
    ans = mse_nn, nll_nn, mscal_nn
    return (ans)


def evaluate_sim_predictions(
    np_xsets: np.ndarray,
    np_ysets: np.ndarray,
    predict_data_size: int,
    num_nearest_neighbours: int,
    input_dim: int,
    assum_params: dict,
    assum_kern: str,
    likelihood,
    prediction_method,
) -> Tuple[float,float,float]:
    """ Get performance metrics for simulations over all test points. If nn
    prediction used need to do one test point at a time to find the relevant x
    and y values.

    Args:
        np_xsets (np.ndarray): covariates, organised for access
        np_ysets (np.ndarray): obs. organised for access
        predict_data_size (int): number of test points
        num_nearest_neighbours (int): m
        input_dim (int): d
        assum_params (dict): e.g. {"lengthscale": 0.5} etc
        assum_kern (str): e.g. 'RBF'
        likelihood (_type_): gpytorch likelihood
        prediction_method (_type_): see prediction.py

    Returns:
        Tuple[float,float,float]: MSE, NLL, CAL
    """
    mse_nn = 0.0
    me_nn = 0.0
    nll_nn = 0.0
    mscal_nn = 0.0
    mnn_predict_time = 0.0

    np_predict_x = np.zeros([1, input_dim], dtype=np.float64)
    np_nearest_x = np.zeros(
        [num_nearest_neighbours, input_dim], dtype=np.float64
    )
    np_nearest_y = np.zeros([num_nearest_neighbours], dtype=np.float64)

    assum_params = map_simple_params_to_cheat(assum_params)

    pred_mean = np.zeros([predict_data_size], dtype=np.float64)
    pred_sd = np.zeros([predict_data_size], dtype=np.float64)
    y_true = np.zeros([predict_data_size], dtype=np.float64)

    if prediction_method.__name__ in (
            "make_full_predictions", "make_straight_nn_predictions"):
        for i_trial in range(predict_data_size):
            nn_ind = np.s_[
                (i_trial * (num_nearest_neighbours + 1)): (
                    (i_trial + 1) * (num_nearest_neighbours + 1) - 1
                )
            ]
            last_ind = (
                i_trial * (num_nearest_neighbours + 1)
            ) + num_nearest_neighbours

            # collect together both nn xvals and target xval for this trial
            np_predict_x = np_xsets[np.newaxis, last_ind, :]
            np_nearest_x = np_xsets[nn_ind, :]
            # collect together both nn yvals and true target yval for this trial
            y_true[i_trial] = np_ysets[last_ind]
            np_nearest_y = np_ysets[nn_ind]
            # convert to the tensor format required for the gpytorch follow-on processing
            predict_x = torch.from_numpy(np_predict_x)
            # set up nn model with the assumed (possibly mispecified) kernel family and parameters and put in eval mode ready to make predictions
            tic = time.perf_counter()
            pred_mean[i_trial], pred_sd[i_trial] = prediction_method(
                np_predict_x, np_nearest_x, np_nearest_y, likelihood,
                assum_params, None, assum_kern)
            toc = time.perf_counter()
            mnn_predict_time += toc - tic
            # add this trial's results to the stats collection
    else:
        tot_len = np_xsets.shape[0]
        neigh = SparseParams(n_inducing_pts=num_nearest_neighbours)
        pred_mean, pred_sd = prediction_method(
            np_xsets[tot_len - predict_data_size:, :],
            np_xsets[: tot_len - predict_data_size, :],
            np_ysets[: tot_len - predict_data_size],
            likelihood, assum_params, neigh, assum_kern)
        y_true = np_ysets[tot_len - predict_data_size:]

    # compute average stats over all predictions
    mse_nn, nll_nn, mscal_nn = evaluate_predictions(
        pred_mean, pred_sd, y_true)
    ans = mse_nn, nll_nn, mscal_nn
    return ans
