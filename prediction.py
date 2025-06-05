import torch
import numpy as np
from typing import Tuple
import warnings
import gpytorch

from gpprediction.models import GP, kNN, get_model, sample_uniform_grid, CurseOfDimensionalityException


def get_pred_mean_sd(model: GP, predict_x: torch.Tensor, assum_params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Take model and make predictions at predict_x and then extract mean and
    std of predictive dist.

    Args:
        model (GP): _description_
        predict_x (torch.Tensor): _description_
        assum_params (dict): _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """
    output_model = model(predict_x)
    pred_mean = output_model.mean
    predf_var = output_model.variance  # doesn't account for noise
    pred_var = predf_var + assum_params['likelihood.noise_covar.raw_noise']
    pred_sd = pred_var**0.5
    ans = pred_mean.detach().numpy(), pred_sd.detach().numpy()
    return (ans)


def _make_predictions(x_predict: np.ndarray, assum_params: dict, model: GP) -> Tuple[np.ndarray, np.ndarray]:
    """ Using given params make predictions using GP at x_predict

    Args:
        x_predict (np.ndarray): test loc.
        assum_params (_type_): params to use to predict
        model (_type_): GP

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """
    predict_x = torch.from_numpy(x_predict)
    model.set_parameters(assum_params)
    model.eval()

    ans = get_pred_mean_sd(model, predict_x, assum_params)
    return (ans)


def get_subset(
        predict_x: np.ndarray, x_train: np.ndarray, n: int, m: int, method='nn',
        neigh=None) -> np.ndarray:
    """Using method specified find a subset of the training data

    Args:
        predict_x (np.ndarray): test loc
        x_train (np.ndarray): training locs
        n (int): size of training set
        m (int): number of nearest neighbours
        method (str, optional): method to select. Defaults to 'nn'.
        neigh (_type_, optional): e.g. Sklearn (might not be though, bad naming). Defaults to None.

    Raises:
        NotImplementedError: _description_

    Returns:
        np.ndarray: indices to select subset from training
    """
    if method == 'nn':
        # neigh_list = neigh.query(predict_x, k=m, epsilon=0.4)[0]
        neigh_list = neigh.kneighbors(predict_x, return_distance=False)
        inds = neigh_list.flatten()
    elif method == 'sparse_nn':
        k = np.min([m//2, 400])
        # neigh_list = neigh.query(predict_x, k=k, epsilon=0.4)[0]
        neigh_list = neigh.kneighbors(
            predict_x, n_neighbors=k, return_distance=False)
        inds = neigh_list.flatten()
        try:
            extra_inds, _ = sample_uniform_grid(m-k, x_train)
        except CurseOfDimensionalityException as cde:
            warnings.warn(
                "Curse of dimensionality prevents grid structure, reverting to uniform.")
            extra_inds = np.random.choice(
                list(set(range(0, n)).difference(inds)), m-k, replace=False)
        inds = np.concatenate([inds, extra_inds])
        return inds
    elif method == 'random':
        inds = np.random.choice(range(0, n), m)
    elif method == 'ray_nn':
        # WARNING: VERY SLOW
        k = np.min([m//2, 400])
        neigh_list = neigh.kneighbors(
            predict_x, n_neighbors=k, return_distance=False)
        inds = neigh_list.flatten()
        inds = get_rays(predict_x, x_train, m, inds)
        return inds
    else:
        raise NotImplementedError
    return inds


def get_rays(predict_x, X, m, inds):
    n = X.shape[0]
    S = 10000
    M = len(inds)
    all_inds = inds
    MAX_ITER = 100
    iter = 0
    while M < m and S > 1000 and iter < MAX_ITER:
        # get uniform subsample
        remaining_inds = list(set(range(0, n)).difference(all_inds))
        S = np.min([3*len(remaining_inds)//4, 10000])
        extra_inds = np.random.choice(remaining_inds, S, replace=False)
        all_inds = np.concatenate([inds, extra_inds])
        new_pts = get_points_in_ray(predict_x, X, extra_inds)
        M += len(new_pts)
        inds = np.concatenate([inds, new_pts])
        iter += 1
    if iter == MAX_ITER:
        warnings.warn(f"Max. iterations reached. S={S}, M={M}")
    return inds[:m].astype(int)


def get_points_in_ray(predict_x, X, extra_inds):
    n, d = X.shape
    # pick a direction
    u = np.random.randn(d)
    u /= np.sqrt(np.dot(u, u))
    # check if any points are in the cone:
    theta = np.pi/2*(1-1/d)
    def R(h): return np.abs(h * np.sin(theta))

    Xi = X[extra_inds, :]
    pXi = Xi @ u
    ri = [pxi * u for pxi in pXi] - (Xi - predict_x)
    rinorms = np.sqrt(np.inner(ri, ri).diagonal())
    cone_inds = extra_inds[rinorms < R(pXi)]
    return cone_inds


def _make_subset_prediction(
        x_predict: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
        likelihood, assum_params: dict, neigh, assum_kern: str, method: str, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Given test points, make predictions and output their mean and variance

    Args:
        x_predict (np.ndarray): test loc
        x_train (np.ndarray): training locs
        y_train (np.ndarray): training obs.
        likelihood (_type_): GpyTorch likelihood
        assum_params (dict): params to use for GP prediction
        neigh (_type_): e.g. SKlearn object
        assum_kern (str): e.g. 'RBF'
        method (str): e.g. 'nn'
        m (int): number of nearest neighbours / inducing pts/ etc

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """
    predict_data_size = x_predict.shape[0]
    pred_mean = np.zeros([predict_data_size], dtype=np.float64)
    pred_sd = np.zeros([predict_data_size], dtype=np.float64)
    for i_trial in range(predict_data_size):
        np_predict_x = x_predict[np.newaxis, i_trial, :]
        predict_x = torch.from_numpy(np_predict_x)

        inds = get_subset(np_predict_x, x_train,
                          y_train.shape[0], m, method, neigh)
        subset_x = x_train[inds, :]
        subset_y = y_train[inds]
        # set up nn model with the assumed (possibly mispecified) kernel family and parameters and put in eval mode ready to make predictions
        model = get_model(subset_x, subset_y, likelihood, assum_kern, 'exact')
        model.set_parameters(assum_params)
        model.eval()
        # NOTE: to speed up
        pred_mean[i_trial], pred_sd[i_trial] = get_pred_mean_sd(
            model, predict_x, assum_params)

    ans = pred_mean, pred_sd
    return (ans)


def make_sparse_predictions(
    x_predict: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
        likelihood, assum_params: dict, neigh, assum_kern: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """SGPR prediction with random inducing locs.

    Args:
        x_predict (np.ndarray): test loc
        x_train (np.ndarray): training locs
        y_train (np.ndarray): training obs.
        likelihood (_type_): GpyTorch likelihood
        assum_params (dict): params to use for GP prediction
        neigh (_type_): e.g. SKlearn object
        assum_kern (str): e.g. 'RBF'

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """
    MAX_INDUCING_POINTS = 2048
    with gpytorch.beta_features.checkpoint_kernel(1500), gpytorch.settings.max_root_decomposition_size(neigh.n_inducing_pts):
        if neigh.n_inducing_pts > MAX_INDUCING_POINTS:
            M = neigh.n_inducing_pts // MAX_INDUCING_POINTS
            model = get_model(
                x_train, y_train, likelihood, assum_kern, 'sparse',
                n_inducing_pts=MAX_INDUCING_POINTS, **kwargs)
            pred_mean, pred_sd = _make_predictions(
                x_predict,  assum_params, model)
            pred_var = pred_sd**2 - \
                assum_params['likelihood.noise_covar.raw_noise'].item()
            for m in range(1, M):
                model.init_inducing_pts(MAX_INDUCING_POINTS, m)
                _pred_mean, _pred_sd = _make_predictions(
                    x_predict, assum_params, model)
                pred_var += _pred_sd**2 - \
                    assum_params['likelihood.noise_covar.raw_noise'].item()
                pred_mean += _pred_mean
            pred_mean /= M
            pred_sd = np.sqrt(pred_var/M)
            ans = pred_mean, pred_sd
        else:
            model = get_model(
                x_train, y_train, likelihood, assum_kern, 'sparse',
                n_inducing_pts=neigh.n_inducing_pts, **kwargs)
            ans = _make_predictions(x_predict, assum_params, model)
    return (ans)


def make_full_predictions(
    x_predict: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
    likelihood, assum_params: dict, neigh, assum_kern: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Use exact GP predictions. When x_train, y_train are the NN sets for
    x_predict equivalent to GPnn

    Args:
        x_predict (np.ndarray): test loc
        x_train (np.ndarray): training locs
        y_train (np.ndarray): training obs.
        likelihood (_type_): GpyTorch likelihood
        assum_params (dict): params to use for GP prediction
        neigh (_type_): e.g. SKlearn object
        assum_kern (str): e.g. 'RBF'

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """
    model = get_model(x_train, y_train, likelihood, assum_kern, 'exact')
    ans = _make_predictions(x_predict, assum_params, model)
    return (ans)


def make_sod_predictions(
    x_predict: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
    likelihood, assum_params: dict, neigh, assum_kern: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Run (random) subset-of-data prediction

    Args:
        x_predict (np.ndarray): test loc
        x_train (np.ndarray): training locs
        y_train (np.ndarray): training obs.
        likelihood (_type_): GpyTorch likelihood
        assum_params (dict): params to use for GP prediction
        neigh (_type_): e.g. SKlearn object
        assum_kern (str): e.g. 'RBF'

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """
    inds = get_subset(x_predict, x_train,
                      y_train.shape[0], neigh.n_subset_pts, 'random')
    model = get_model(x_train[inds, :], y_train[inds],
                      likelihood, assum_kern, 'exact')
    ans = _make_predictions(x_predict, assum_params, model)
    return (ans)


def make_var_predictions(
    x_predict: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
        likelihood, assum_params: dict, neigh, assum_kern: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Use SVGP as predictive mechanism w optim. inducing locations

    Args:
        x_predict (np.ndarray): test loc
        x_train (np.ndarray): training locs
        y_train (np.ndarray): training obs.
        likelihood (_type_): GpyTorch likelihood
        assum_params (dict): params to use for GP prediction
        neigh (_type_): e.g. SKlearn object
        assum_kern (str): e.g. 'RBF'

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """
    model = get_model(
        x_train, y_train, likelihood, assum_kern, 'sparse',
        n_inducing_pts=neigh.n_inducing_pts, sampling_strategy="uniform")
    ans = _make_predictions(x_predict, assum_params, model)
    return (ans)


def make_ensemble_predictions(
    x_predict: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
        likelihood, assum_params: dict, neigh, assum_kern: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    predict_data_size = x_predict.shape[0]
    pred_mean = np.zeros([predict_data_size], dtype=np.float64)
    pred_sd = np.zeros([predict_data_size], dtype=np.float64)
    # which methods to ensemble, and how? BCM/Wass/?
    # GpoE currently
    n_neighbors = np.min([neigh.n_neighbors//2, 400])
    neigh.n_inducing_pts = neigh.n_neighbors - n_neighbors
    neigh.n_neighbors = n_neighbors
    sparse_pred = make_sparse_predictions(
        x_predict, x_train, y_train, likelihood, assum_params, neigh, assum_kern)
    nn_pred = make_nn_predictions(
        x_predict, x_train, y_train, likelihood, assum_params, neigh,
        assum_kern)

    sparse_mean, sparse_sd = sparse_pred
    nn_mean, nn_sd = nn_pred

    sparse_var = sparse_sd**2 - \
        assum_params['likelihood.noise_covar.raw_noise'].item()
    nn_var = nn_sd**2 - assum_params['likelihood.noise_covar.raw_noise'].item()

    pred_var = 1/((1/sparse_var**2 + 1/nn_var**2)/2)
    pred_mean = (nn_mean/nn_var**2 + sparse_mean/sparse_var**2)/(2/pred_var)

    pred_sd = (
        pred_var + assum_params['likelihood.noise_covar.raw_noise'].item())**0.5

    ans = pred_mean, pred_sd
    return (ans)


def make_nn_predictions(
    x_predict: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
        likelihood, assum_params: dict, neigh, assum_kern: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Predict with GPnn

    Args:
        x_predict (np.ndarray): test loc
        x_train (np.ndarray): training locs
        y_train (np.ndarray): training obs.
        likelihood (_type_): GpyTorch likelihood
        assum_params (dict): params to use for GP prediction
        neigh (_type_): e.g. SKlearn object
        assum_kern (str): e.g. 'RBF'

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """
    ans = _make_subset_prediction(
        x_predict, x_train, y_train, likelihood, assum_params, neigh,
        assum_kern, 'nn', neigh.n_neighbors)
    return (ans)


def make_sparse_nn_predictions(
    x_predict: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
        likelihood, assum_params: dict, neigh, assum_kern: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Use a combination of sparse and GPnn to predict

    Args:
        x_predict (np.ndarray): test loc
        x_train (np.ndarray): training locs
        y_train (np.ndarray): training obs.
        likelihood (_type_): GpyTorch likelihood
        assum_params (dict): params to use for GP prediction
        neigh (_type_): e.g. SKlearn object
        assum_kern (str): e.g. 'RBF'

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """
    ans = _make_subset_prediction(
        x_predict, x_train, y_train, likelihood, assum_params, neigh,
        assum_kern, 'sparse_nn', neigh.n_neighbors)
    return (ans)


def make_ray_nn_predictions(
    x_predict: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
        likelihood, assum_params: dict, neigh, assum_kern: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Augment NN sets with 'rays' before predicting

    Args:
    Args:
        x_predict (np.ndarray): test loc
        x_train (np.ndarray): training locs
        y_train (np.ndarray): training obs.
        likelihood (_type_): GpyTorch likelihood
        assum_params (dict): params to use for GP prediction
        neigh (_type_): e.g. SKlearn object
        assum_kern (str): e.g. 'RBF'

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """
    ans = _make_subset_prediction(
        x_predict, x_train, y_train, likelihood, assum_params, neigh,
        assum_kern, 'ray_nn', neigh.n_neighbors)
    return (ans)


def make_npae_predictions(
    x_predict, x_train, y_train, likelihood, assum_params, neigh,
        assum_kern):
    raise NotImplementedError


def make_ski_predictions(
    x_predict: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
        likelihood, assum_params: dict, neigh, assum_kern: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """ Use SKI-GP (Gpytorch) to make predictions
    
    Args:
        x_predict (np.ndarray): test loc
        x_train (np.ndarray): training locs
        y_train (np.ndarray): training obs.
        likelihood (_type_): GpyTorch likelihood
        assum_params (dict): params to use for GP prediction
        neigh (_type_): e.g. SKlearn object
        assum_kern (str): e.g. 'RBF'

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """
    model = get_model(x_train, y_train, likelihood, assum_kern, 'ski')
    ans = _make_predictions(x_predict, assum_params, model)
    return (ans)


def make_dkl_predictions(
    x_predict, x_train, y_train, likelihood, assum_params, neigh,
        assum_kern):
    raise NotImplementedError


def make_straight_nn_predictions(
    x_predict: np.ndarray, x_train: np.ndarray, y_train: np.ndarray,
        likelihood, assum_params: dict, neigh, assum_kern: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """ Make straight k-NN predictions by averaging NN y values. Also takes
    their variance and uses that for predictions.

    Args:
        x_predict (np.ndarray): test loc
        x_train (np.ndarray): training locs
        y_train (np.ndarray): training obs.
        likelihood (_type_): GpyTorch likelihood
        assum_params (dict): params to use for GP prediction
        neigh (_type_): e.g. SKlearn object
        assum_kern (str): e.g. 'RBF'

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean, std
    """
    predict_data_size = x_predict.shape[0]
    pred_mean = np.zeros([predict_data_size], dtype=np.float64)
    pred_sd = np.zeros([predict_data_size], dtype=np.float64)
    for i_trial in range(predict_data_size):
        np_predict_x = x_predict[np.newaxis, i_trial, :]
        predict_x = torch.from_numpy(np_predict_x)

        # temp comment while simulating for speed
        inds = get_subset(np_predict_x, x_train,
                          y_train.shape[0], neigh.n_neighbors, 'nn', neigh)
        subset_x = x_train[inds, :]
        subset_y = y_train[inds]
        model = kNN(subset_x, subset_y)
        # NOTE: to speed up
        pred_mean[i_trial], pred_sd[i_trial] = get_pred_mean_sd(
            model, predict_x, assum_params)

    ans = pred_mean, pred_sd
    return (ans)
