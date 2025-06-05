import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import gpytorch
import warnings

from gpprediction.utils import inv

from gpprediction.models import get_model, simplify_parameters, map_param_keys, map_simple_params_to_full


def compute_param_estimates(
        all_subset_xvals: np.ndarray, all_subset_yvals: np.ndarray,
        likelihood: gpytorch.likelihoods.Likelihood, subset_size: int,
        num_subsets: int, training_iter: int, assum_kern: str,
        params_to_exc: dict) -> dict:
    """ Runs Adam to compute parameter estimates. Runs latched param. est. (i.e.
    with block diag kernel defined by subsets). If num_subsets=1 usual SoD estimation.

    Args:
        all_subset_xvals (np.ndarray): covariates for training
        all_subset_yvals (np.ndarray): obs. for training
        likelihood (gpytorch.likelihoods.Likelihood): to optimise
        subset_size (int): subset of X vals to optim. over
        num_subsets (int): number of times to repeat (latched)
        training_iter (int): Number of Adam iterations
        assum_kern (str): e.g. 'RBF'
        params_to_exc (dict): Parameters to NOT optim. over, i.e. keep fixed;
        e.g. {"nu": 0.2}

    Returns:
        dict: Parameters with optim. vals, e.g. {"lengthscale": 0.123, ...}
    """
    model, optimizer, mll = setup_model_opt(
        all_subset_xvals, all_subset_yvals, likelihood, assum_kern,
        params_to_exc)
    loss = 0.0
    for i in range(training_iter):
        if i == training_iter - 1 or num_subsets == 1:
            graph_params = {}
        else:
            graph_params = {"retain_graph": True}
        optimizer.zero_grad()
        train_xx = torch.from_numpy(
            all_subset_xvals[(0*subset_size):(1*subset_size), :])
        train_yy = torch.from_numpy(
            all_subset_yvals[(0*subset_size):(1*subset_size)])
        with gpytorch.settings.debug(False):
            output = model(train_xx)
            loss = -mll(output, train_yy)
            for i_subset in range(1, num_subsets):
                train_xx = torch.from_numpy(
                    all_subset_xvals[(i_subset*subset_size):((i_subset+1)*subset_size), :])
                train_yy = torch.from_numpy(
                    all_subset_yvals
                    [(i_subset * subset_size): ((i_subset + 1) * subset_size)])
                output = model(train_xx)
                loss -= mll(output, train_yy)
        loss.backward(**graph_params)
        optimizer.step()
    # Optimiser iterations complete so pull out hyper-param estimates
    ans = model.get_parameters()
    return (ans)


def compute_param_estimates_coord(
        all_subset_xvals: np.ndarray, all_subset_yvals: np.ndarray,
        likelihood: gpytorch.likelihoods.Likelihood, subset_size: int,
        num_subsets: int, training_iter: int, assum_kern: str,
        params: dict) -> dict:
    """ As above, but run coordinate descent instead. Not used.

    Args:
        all_subset_xvals (np.ndarray): covariates for training
        all_subset_yvals (np.ndarray): obs. for training
        likelihood (gpytorch.likelihoods.Likelihood): to optimise
        subset_size (int): subset of X vals to optim. over
        num_subsets (int): number of times to repeat (latched)
        training_iter (int): Number of Adam iterations
        assum_kern (str): e.g. 'RBF'
        params (dict): Parameters to NOT optim. over, i.e. keep fixed;
        e.g. {"nu": 0.2}

    Returns:
        dict: Parameters with optim. vals, e.g. {"lengthscale": 0.123, ...}
    """
    model, optimizers, mll = setup_model_opt_coord(
        all_subset_xvals, all_subset_yvals, likelihood, assum_kern, params)
    loss = 0.0
    for i in range(training_iter // len(optimizers)):
        if i == training_iter - 1 or num_subsets == 1:
            graph_params = {}
        else:
            graph_params = {"retain_graph": True}
        for opt in optimizers:
            opt.zero_grad()
            train_xx = torch.from_numpy(
                all_subset_xvals[(0*subset_size):(1*subset_size), :])
            train_yy = torch.from_numpy(
                all_subset_yvals[(0*subset_size):(1*subset_size)])
            with gpytorch.settings.debug(False):
                output = model(train_xx)
                loss = -mll(output, train_yy)
                for i_subset in range(1, num_subsets):
                    train_xx = torch.from_numpy(
                        all_subset_xvals[(i_subset*subset_size):((i_subset+1)*subset_size), :])
                    train_yy = torch.from_numpy(
                        all_subset_yvals[(i_subset*subset_size):((i_subset+1)*subset_size)])
                    output = model(train_xx)
                    loss -= mll(output, train_yy)
            loss.backward(**graph_params)
            opt.step()
    # Optimiser iterations complete so pull out hyper-param estimates
    ans = model.get_parameters()
    return (ans)


def setup_model_opt(
        x: np.ndarray, y: np.ndarray, likelihood, assum_kern,
        params_to_exc: Optional[dict] = None):
    """ create model, opt and mll objects. if params_to_exc is passed, it sets up the
    optimizer to exclude the parameters named in the "params_to_exc" variable.

    Args:
        x (np.ndarray): _description_
        y (np.ndarray): _description_
        likelihood (_type_): _description_
        assum_kern (_type_): _description_
        params_to_exc (_type_, optional): Parameters to exclude from optimisation. Defaults to None.

    Returns:
        _type_: _description_
    """
    model = get_model(x, y, likelihood, assum_kern, 'exact')
    likelihood.double()
    model.train()
    likelihood.train()
    if params_to_exc is not None:
        optimizer = torch.optim.Adam(
            [p for n, p in model.named_parameters()
             if n not in params_to_exc.keys()],
            lr=0.1)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    return model, optimizer, mll


def setup_model_opt_coord(
        x: np.ndarray, y: np.ndarray, likelihood, assum_kern, params=None):
    """As above, but create different optim. for each param. Not used.

    Args:
        x (np.ndarray): _description_
        y (np.ndarray): _description_
        likelihood (_type_): _description_
        assum_kern (_type_): _description_
        params (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    model = get_model(x, y, likelihood, assum_kern, 'exact')
    likelihood.double()
    model.train()
    likelihood.train()
    optimizers = [torch.optim.Adam([p], lr=0.1)
                  for n, p in model.named_parameters()]
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    return model, optimizers, mll


def get_param_avg(
    all_subset_xvals: np.ndarray, all_subset_yvals: np.ndarray,
        likelihood: gpytorch.likelihoods.Likelihood, subset_size: int,
        num_subsets: int, training_iter: int, assum_kern: str,
        params: dict) -> dict:
    """Instead of latched, estimates parameters independently for each subset
    and then averages, possibly taking into account correlations between subsets.


    Args:
        all_subset_xvals (np.ndarray): covariates for training
        all_subset_yvals (np.ndarray): obs. for training
        likelihood (gpytorch.likelihoods.Likelihood): to optimise
        subset_size (int): subset of X vals to optim. over
        num_subsets (int): number of times to repeat (latched)
        training_iter (int): Number of Adam iterations
        assum_kern (str): e.g. 'RBF'
        params (dict): Parameters to NOT optim. over, i.e. keep fixed;
        e.g. {"nu": 0.2}

    Returns:
        dict: Parameters with optim. vals, e.g. {"lengthscale": 0.123, ...}
    """

    model, optimizer, mll = setup_model_opt(
        all_subset_xvals, all_subset_yvals, likelihood, assum_kern, params)
    all_params = [[]] * (num_subsets-1)

    for i_subset in range(1, num_subsets):
        train_xx = all_subset_xvals[(
            i_subset*subset_size):((i_subset+1)*subset_size), :]
        train_yy = all_subset_yvals[(
            i_subset*subset_size):((i_subset+1)*subset_size)]

        model = get_model(train_xx, train_yy, likelihood, assum_kern, 'exact')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        train_xx = torch.from_numpy(train_xx)
        train_yy = torch.from_numpy(train_yy)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_xx)
            loss = -mll(output, train_yy)
            loss.backward()
            optimizer.step()
        subset_params = model.get_parameters()
        all_params[i_subset-1] = simplify_parameters(subset_params)
    # Optimiser iterations complete so pull out hyper-param estimates
    avg_params = _average_params(all_params, "corr")
    ans = map_simple_params_to_full(model, avg_params)
    return (ans)


def _average_params(params: List[Dict], method: str):
    """If different params computed for each subset, then average them. Either
    naiively or using a method that accounts for correlations between subset
    data (and hence estimates).

    Args:
        params (List[Dict]): _description_
        method (str): _description_

    Returns:
        _type_: _description_
    """
    avg_params = {k: p*0.0 for k, p in params[0].items()}
    for name, param in avg_params.items():
        pvec = np.asarray([p[name] for p in params])
        if method == "straight":
            avg_params[name] = pvec.mean()
        if method == "corr":
            avg_params[name] = correlated_avg(pvec)
    return avg_params


def correlated_avg(param_vec: np.ndarray):
    """ Proc. for averaging correlated data, taken from
    Michael Schmelling 1995 Phys. Scr. 51 676
    https://iopscience.iop.org/article/10.1088/0031-8949/51/6/002/pdf

    Args:
        param_vec (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    Cinv = inv(np.outer(param_vec, param_vec))
    a = (Cinv @ param_vec).sum()/Cinv.sum()
    return a


def map_simple_params_to_cheat(simple_params):
    """ Takes a dict with 'simple' param names, e.g. 'nu' and floats, and maps
    to a dict with complicated names and possibly nested arrays. Dumb version. 
    e.g. {'noise': 0.1} -> {'likelihood.noise_covar.raw_noise': torch.Tensor([0.1]

    Args:
        simple_params (Dict[str,float]): Simplifed form of hyperparameters for
        ease of use.

    Returns:
        _type_: _description_
    """
    # TODO: replace with general version from .models
    if 'nu' not in simple_params.keys():
        simple_params['nu'] = np.nan
    return {'likelihood.noise_covar.raw_noise': torch.Tensor([simple_params['noise']]), 'covar_module.raw_outputscale': torch.tensor(simple_params['outputscale']), 'covar_module.base_kernel.raw_lengthscale': torch.Tensor([[simple_params['lengthscale']]]), 'covar_module.base_kernel.raw_nu': torch.tensor(simple_params['nu'])}


def get_param_estimates(
        all_subset_xvals, all_subset_yvals, likelihood, subset_size,
    num_subsets, num_adam_iters, assum_kern, res_file, force_est,
        dataset_name, params=None):
    """ Obtains hyperparameters. First checks for results csvs to see if
    precomputed. If not found, or the "force_est" bool flag is used, then runs
    an estimation proc. 

    Args:
        all_subset_xvals (_type_): input data to use for param. est., size
        num_subsets x subset_size
        all_subset_yvals (_type_): observations "
        likelihood (_type_): gpytorch likelihood object
        subset_size (_type_): size of each subset
        num_subsets (_type_): number to use
        num_adam_iters (_type_): ...
        assum_kern (_type_): kernel for model
        res_file (_type_): csv file string/path to look for existing params.
        force_est (_type_): Bool flag to insist on estimating fresh params.
        dataset_name (_type_): ???
        params (dict, optional): Parameters to keep fixed, in model-dict (not
        simple) format. Defaults to None.

    Returns:
        dict: Estimated parameters
    """
    try:
        res = pd.read_csv(res_file)
        res = res.query('dataset == @dataset_name and k_model == @assum_kern')
        if res.shape[0] == 0:
            force_est = True
    except (FileNotFoundError, pd.errors.ParserError) as err:
        warnings.warn(err.args[1])
        force_est = True
    if force_est:
        warnings.warn("No previous params found, estimating new ones...")
        if "dim" in assum_kern and params is None:
            params_est = compute_param_estimates_coord(
                all_subset_xvals, all_subset_yvals, likelihood, subset_size,
                num_subsets, num_adam_iters, assum_kern, params)
        else:
            params_est = compute_param_estimates(
                all_subset_xvals, all_subset_yvals, likelihood, subset_size,
                num_subsets, num_adam_iters, assum_kern, params)
        if params is not None:
            params_est.update(params)
        return params_est
    # if multiple, get smallest mse
    res = res.loc[res.mse.idxmin(), :]
    params = {
        'likelihood.noise_covar.raw_noise': torch.Tensor([res.est_nv]),
        'covar_module.raw_outputscale': torch.tensor(res.est_ks),
        'covar_module.base_kernel.raw_lengthscale': torch.Tensor(
            [[res.est_ls]])}
    # is this still needed? (below)
    if assum_kern.lower() == "matern":
        params['covar_module.base_kernel.raw_nu'] = torch.tensor(res.est_nu)
    return params


class SparseParams(object):
    def __init__(self, n_inducing_pts=None, n_subset_pts=None):
        self.n_inducing_pts = n_inducing_pts
        self.n_subset_pts = n_subset_pts
