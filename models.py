from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
import torch
import numpy as np
import pandas as pd
import warnings
import gpytorch
from typing import Callable, Tuple
from itertools import product as cartesian_prod

from gpytorch.constraints import Interval
from gpytorch.module import Module
from gpybench.kernels import MaternKernel, ScaledKernel, InducingPointKernel, SparseKernel, DimReductionKernel, SparseARDKernel

# from dppy.finite_dpps import FiniteDPP
from sklearn.neighbors import NearestNeighbors

from gpprediction.utils import invert_dict, noti


def sample_ints_from_range(n, n_samples, seed=1):
    return torch.multinomial(torch.ones(n)/n,
                             n_samples, generator=torch.Generator().manual_seed(seed)
                             ).type(torch.long)


def simplify_name(name: str) -> str:
    """ remove path/_ etc to create simple names for params,
    e.g. 'covar_module.base_kernel.raw_lengthscale' -> 'lengthscale'

    Args:
        name (str): input complex name str

    Returns:
        str: e.g. 'lengthscale'
    """
    simple_name = name[name.rfind(".")+1:]
    return simple_name.replace("raw_", "")

# note all of the GP defs below use zero mean option


def simplify_parameters(params: dict) -> dict:
    """Take dict of complex param names and return dict of simple ones (with
    same values)

    Args:
        params (dict): _description_

    Returns:
        dict: _description_
    """
    parameters = {}
    for name, param in params.items():
        if np.prod(param.detach().numpy().shape) == 1:
            parameters[simplify_name(name)] = param.item()
        else:
            parameters[simplify_name(name)] = param
    return parameters


def map_param_keys(params: dict) -> dict:
    """ helper to map between potentially complex path to parameter in object
    and the relevant name, e.g. 'covar_module.base_kernel.raw_lengthscale' and 'lengthscale'

    Args:
        params (dict): _description_

    Returns:
        dict: e.g. {'lengthscale': 'covar_module.base_kernel.raw_lengthscale'}
    """
    map = {}
    for name, param in params.items():
        map[simplify_name(name)] = name
    return map


def get_base_kernel(assum_kern: str, **kwargs) -> gpytorch.kernels.Kernel:
    """Constructs GpyTorch kernel object from str name

    Args:
        assum_kern (str): e.g. 'RBF'

    Raises:
        ValueError: If not given a known name

    Returns:
        gpytorch.kernels.Kernel: _description_
    """
    if (assum_kern.lower() == 'rbf'):
        base_kernel = gpytorch.kernels.RBFKernel(**kwargs)
    elif (assum_kern.lower() == 'matern'):
        base_kernel = MaternKernel(**kwargs)
    elif (assum_kern.lower() == 'matern32'):
        base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, **kwargs)
    elif (assum_kern.lower() == 'exp'):
        base_kernel = gpytorch.kernels.MaternKernel(nu=0.5, **kwargs)
    else:
        raise ValueError(
            "Valid kernel not supplied. Options: RBF, Matern, Matern32, Exp.")
    return base_kernel

class kNN(Module):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, *args):
        self.train_inputs = train_x
        self.train_targets = train_y

    def forward(self, x):
        mean_x = torch.as_tensor(self.train_targets.mean(keepdims=True))
        covar_x = torch.as_tensor(
            self.train_targets.var(keepdims=True)).unsqueeze(1)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_parameters(self) -> dict:
        """ Retrieves dict of parameters and their values, in form used by
        GPyTorch, e.g. {'covar_module.base_kernel.lengthscale':
        np.tensor([0.5])} (or similar)

        Returns:
            dict: _description_
        """
        parameters = {}
        for name, param, con in self.named_parameters_and_constraints():
            if con is not None:
                parameters[name] = con.transform(param)
            else:
                parameters[name] = param
        return parameters

    def get_simple_parameters(self) -> dict:
        """As above, but simplifies so output looks like {'lengthscale': 0.5}
        rather than inc. the full depth of GpyTorch

        Returns:
            dict: _description_
        """
        parameters = {}
        for name, param, con in self.named_parameters_and_constraints():
            if con is not None:
                parameters[simplify_name(name)] = con.transform(param).item()
            else:
                parameters[simplify_name(name)] = param.item()
        return parameters

    def set_parameters(self, params: dict) -> None:
        """Takes a dict of 'simple' parameters and updates this objects model
        parameters to the values given

        Args:
            params (dict): _description_
        """
        state_dict = self.state_dict()
        # use the key-mapping function to match parameters defined with
        # different kernel structures, e.g. covar_module.lengthscale <->
        # base_kernel.lengthscale, via the "simple" key (lengthscale here). Need
        # to invert the key-value pairs, hence calls invert_dict.
        local_params = self.get_parameters()
        local_map = map_param_keys(local_params)
        input_map = map_param_keys(params)
        # for name,param in params.items():
        # state_dict[local_map[invert_dict(input_map)[name]]] = param
        for name, _, con in self.named_parameters_and_constraints():
            try:
                input_key = input_map[invert_dict(local_map)[name]]
            except KeyError as ke:
                message = f"Missing input param: {ke.args[0]}"
                warnings.warn(message)
                continue
            param = params[input_key]
            if con is not None:
                state_dict[name] = con.inverse_transform(param)
            else:
                state_dict[name] = param
        self.load_state_dict(state_dict)

def map_simple_params_to_full(model: GP, params: dict) -> dict:
    """    use the key-mapping function to match parameters defined with
    different kernel structures, e.g. covar_module.lengthscale <->
    base_kernel.lengthscale, via the "simple" key (lengthscale here). Need
    to invert the key-value pairs, hence calls invert_dict.

    Args:
        model (GP): GP object
        params (dict): e.g. {'lengthscale': 0.5}

    Returns:
        dict: Full complex GPyTorch compatible dict
    """
 
    model_params = model.get_parameters()
    model_map = map_param_keys(model_params)
    input_map = map_param_keys(params)
    new_params = {}
    for name, _, con in model.named_parameters_and_constraints():
        try:
            input_key = input_map[invert_dict(model_map)[name]]
        except KeyError as ke:
            message = f"Missing input param: {ke.args[0]}"
            warnings.warn(message)
            continue
        param = params[input_key]
        new_params[name] = torch.ones_like(model_params[name]) * param
    return new_params

def get_model(
        x_data: np.ndarray, y_data: np.ndarray, likelihood, assum_kern: str,
        model_type: str, **kwargs) -> GP:
    """ Creates GP object using x and y for training data.

    Args:
        x_data (np.ndarray): training locs
        y_data (np.ndarray): training obs
        likelihood (_type_): GpyTorch likelihood
        assum_kern (_type_): e.g. 'RBF'
        model_type (_type_): e.g. 'exact'

    Returns:
        GP: GP object
    """
    x_torch = torch.from_numpy(x_data)
    y_torch = torch.from_numpy(y_data)
    if 'dim' in assum_kern.lower():
        _base_kernel_str = assum_kern.lower()[assum_kern.lower().rfind("_")+1:]
        d = x_torch.shape[1]
        projection_mat = torch.randn(d, d//2)
        _base_kernel = get_base_kernel(_base_kernel_str)
        base_kernel = DimReductionKernel(_base_kernel,
                                         projection_mat)
    elif 'ard' in assum_kern.lower():
        _base_kernel_str = assum_kern.lower()[assum_kern.lower().rfind("_")+1:]
        d = x_torch.shape[1]
        _base_kernel = get_base_kernel(_base_kernel_str, ard_num_dims=d)
        base_kernel = SparseARDKernel(
            _base_kernel, reg_scale=10e-2, threshold=0.1)
    else:
        base_kernel = get_base_kernel(assum_kern, **kwargs)

    if (model_type.lower() == 'sparse'):
        model = SGPR(x_torch, y_torch, likelihood, base_kernel, **kwargs)
    if (model_type.lower() == 'ski'):
        model = SKIGP(x_torch, y_torch, likelihood, base_kernel)
    if (model_type.lower() == 'exact'):
        model = ExactGP(x_torch, y_torch, likelihood, base_kernel)
    model.double()
    return model
class ExactGP(GP):
    def __init__(self, train_x, train_y, likelihood, base_kernel):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)


class SGPR(GP):
    def __init__(self, train_x, train_y, likelihood, base_kernel,
                 n_inducing_pts, inducing_idx=None,
                 sampling_strategy="uniform", seed=42):
        super(SGPR, self).__init__(train_x, train_y, likelihood)

        if inducing_idx is not None:
            inducing_points = train_x[inducing_idx, :]
        else:
            n = train_y.shape[0]
            if sampling_strategy == "uniform":
                inducing_idx = sample_ints_from_range(n, n_inducing_pts, seed)
            elif sampling_strategy == "kdpp":
                L = base_kernel.forward(train_x, train_x).detach().numpy()
                dpp = FiniteDPP('likelihood', **{'L': L})
                inducing_idx = dpp.sample_mcmc_k_dpp(n_inducing_pts)
            elif sampling_strategy == "herd":
                inducing_idx = self.herd_inducing_idx(
                    base_kernel, n_inducing_pts)
            elif sampling_strategy == "grid":
                try:
                    inducing_idx = self.sample_uniform_grid(n_inducing_pts)
                except CurseOfDimensionalityException as cde:
                    warnings.warn(
                        "Curse of dimensionality prevents grid structure, reverting to uniform.")
                    inducing_idx = sample_ints_from_range(
                        n, n_inducing_pts, seed)

            inducing_points = train_x[inducing_idx, :]
        base_kernel = gpytorch.kernels.ScaleKernel(base_kernel)
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            base_kernel,
            inducing_points=inducing_points,
            likelihood=likelihood,
        )

    def init_inducing_pts(self, n_inducing_pts, seed):
        n = self.train_targets.shape[0]
        inducing_idx = sample_ints_from_range(n, n_inducing_pts, seed)
        inducing_points = self.train_inputs[0][inducing_idx, :]
        self.covar_module.inducing_points = torch.nn.Parameter(inducing_points)

    def herd_inducing_idx(self, kernel, n_inducing_pts):
        train_x = self.train_inputs[0]
        n = train_x.shape[0]
        N = np.min([10000, n])
        K = kernel.forward(train_x[:N, :], train_x[:N, :]).detach().numpy()
        inducing_idx = herd_kernel_locs(N, K, n_inducing_pts)
        return inducing_idx

    def sample_uniform_grid(self, n_inducing_pts):
        # Wrong/Terrible. Pretty sure it's not working
        train_x = self.train_inputs[0].detach().numpy()
        inducing_idx, _ = sample_uniform_grid(n_inducing_pts, train_x)
        return inducing_idx

    def optimise_inducing_pts(
            self, x_train, y_train, training_iter=100,
            loss_fun: gpytorch.mlls.
            MarginalLogLikelihood = gpytorch.mlls.ExactMarginalLogLikelihood):
        # this version applies latched param estimation (i.e. with block diag kernel defined by subsets).
        # Find optimal model inducing pts
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        self.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            [p for n, p in self.named_parameters()
             if n == 'covar_module.inducing_points'],
            lr=0.1)  # min([0.1, 1/x_train.shape[1]])

        # "Loss" for GPs - the marginal log likelihood
        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        mll = loss_fun(self.likelihood, self)
        # mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(self.likelihood, self)
        # mll = MSPE(self.likelihood, self, y_train.shape[0])

        for i in range(training_iter):
            if i == training_iter - 1:
                graph_params = {}
            else:
                graph_params = {"retain_graph": True}
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            output = self(x_train)
            loss = -mll(output, y_train)
            loss.backward(**graph_params)
            optimizer.step()
        # Optimiser iterations complete so pull out hyper-param estimates
        ans = self.get_parameters()
        return (ans)


def herd_kernel_locs(n, K, m) -> np.ndarray:
    avg_ks = np.concatenate(
        [np.tril(K, -1).sum(axis=0)[:-1]/(n-np.arange(1, n)), [0.0]])

    I = np.empty((m,), dtype=int)
    imax = np.argmax(avg_ks)
    I[0] = imax
    mask = np.ones(n, dtype=bool)
    mask[imax] = False
    for i in range(1, m):
        imax = np.argmax(avg_ks[mask] - 1/(i+1) * K[~mask, :].sum(axis=0)[mask])
        I[i] = imax
        mask[imax] = False
    return I


def sample_uniform_grid(n_inducing_pts, train_x):
    n, d = train_x.shape
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(train_x)
    m_d = int(n_inducing_pts ** (1/d))
    if m_d < 2:
        raise CurseOfDimensionalityException(
            "Dimension too high for this number of inducing points to form a grid.")
    xd = []
    inducing_idx = set()
    for di in range(d):
        min_di, max_di = train_x[:, di].min(), train_x[:, di].max()
        xd += [np.linspace(min_di, max_di, m_d+2)[1:-1]]
    grid_pts = list(cartesian_prod(*xd))
    for ni in range(m_d ** d):
        grid_pt = np.asarray(grid_pts[ni])
        neigh_i = neigh.kneighbors(grid_pt.reshape(
            1, -1), return_distance=False).flatten()[0]
        inducing_idx.add(neigh_i)
    return np.asarray(list(inducing_idx)), grid_pts


class CurseOfDimensionalityException(Exception):
    pass


class SKIGP(GP):
    def __init__(self, train_x, train_y, likelihood, base_kernel):
        super(SKIGP, self).__init__(train_x, train_y, likelihood)
        self.covar_module = gpytorch.kernels.ProductStructureKernel(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.GridInterpolationKernel(
                    base_kernel, grid_size=100, num_dims=1)),
            num_dims=train_x.shape[1])


# class ExactGP_ScaledRBF(ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(ExactGP_ScaledRBF, self).__init__(train_x, train_y, likelihood)
#         self.base_kernel = gpytorch.kernels.RBFKernel()
#         self.scaled_kernel = ScaledKernel(self.base_kernel, rkhsscale=0.9, rkhsscale_constraint=Interval(0,1))
#         self.covar_module = gpytorch.kernels.ScaleKernel(self.scaled_kernel)


class MSPE(_ApproximateMarginalLogLikelihood):
    r"""
    Mean square prediction error. Likelihood = exact GP? model = approx.? Both
    MultivariateNormal objects? Essentially want kernel function for both. maybe
    just want likelihood to be replaced by exact_model and model by approx_model?

    .. math::


    """

    def __init__(self, likelihood, model, num_data):
        super().__init__(likelihood, model, num_data)
        self.num_data = num_data

    def loss(self, target, approximate_dist_f, **kwargs):
        # get subset of covar etc
        # start with just one, then do some small number, dumb and
        # inefficiently, then make more efficient with clever LA
        n = self.num_data
        n_ss = 5
        subset_size = n//n_ss
        I = np.arange(0, n, subset_size)

        # K = self.model.covar_module.base_kernel(self.model.train_inputs[0]).evaluate()
        # Q = self.model.covar_module(self.model.train_inputs[0])

        X = self.model.train_inputs[0]

        _loss = 0.0
        for i in range(n_ss):
            Xi = X[(i*subset_size):((i+1)*subset_size), :]
            K = self.model.covar_module.base_kernel(Xi).evaluate()
            Ki = K[noti(0, subset_size), :]  # apropriate subset...
            Ki = Ki[:, noti(0, subset_size)]
            ki = K[noti(0, subset_size), 0].unsqueeze(-1)

            Q = self.model.covar_module(Xi)
            Qi = Q[noti(0, subset_size), :]
            Qi = Qi[:, noti(0, subset_size)]
            vi = Qi.inv_matmul(ki)
            _loss += vi.T @ (Ki @ vi - 2*ki)
        return _loss

    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        return self.loss(target, approximate_dist_f, **kwargs).sum(-1)

    def forward(self, approximate_dist_f, target, **kwargs):
        r"""
        ???
        """
        return self._log_likelihood_term(approximate_dist_f, target, **kwargs)
        # return super().forward(approximate_dist_f, target, **kwargs)
