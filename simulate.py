# To understand what this is doing see algorithm 1 in paper

import argparse
from distutils.errors import LibError
import os
import torch
import math
import numpy as np
import gpytorch
import time
from itertools import product as cartesian_prod
from sklearn.neighbors import NearestNeighbors
from typing import Callable, Optional, List, Dict

from gpprediction.datasets import rescale_dataset_noise, get_xy_nn_prediction_subsets, generate_xy_nn_prediction_subsets, generate_xy_nn_prediction_subsets2, generate_spherical_gaus_xvals
from gpprediction.models import get_model
from gpprediction.prediction import make_full_predictions, make_sparse_predictions
from gpprediction.eval import evaluate_sim_predictions
from gpprediction.preprocess import add_noisy_dimsx, get_xy_preprocess_tranforms, preprocess_x_vals, preprocess_y_vals
from gpprediction.utils import inc_varypar

data_path = "synthetic-datasets/ciq_synthetic_datasets/noise_var=0.008/"

lengthscale = 0.3
INPUT_DIM = 10
# f'DIM{{{INPUT_DIM}}}_LENSCALE{{{lengthscale}}}_{{1}}'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-xyinfl', '--xy_input_file', help='xy vals input file',
        default=None)

    # Problem Shape Args
    parser.add_argument(
        "-d",
        "--dimension",
        help="Dimensionality of input data",
        default=INPUT_DIM,
        type=int,
    )
    parser.add_argument(
        "-n_train",
        "--train-data-size",
        help="Size of train data set",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "-n_test",
        "--test-data-size",
        help="Size of test data set",
        default=1000,
        type=int,
    )

    # Truth GP Args
    parser.add_argument(
        "-tker",
        "--true-kernel-type",
        help="Kernel type to use in data-generation",
        default="RBF",
        choices=["RBF", "Exp"],
    )
    parser.add_argument(
        "-tks",
        "--true-kernel-scale",
        help="Truth value for output scale hyperparameter ",
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "-tl",
        "--true-lengthscale",
        help="Truth value for length scale hyperparameter",
        default=lengthscale,
        type=float,
    )
    parser.add_argument(
        "-tnv",
        "--true-noise",
        help="Truth value for noise_var",
        default=0.1,
        type=float,
    )

    # Assumed GP params - in sensitivity analysis two of the hyparams (currently
    # anv and aks) will be held constant while the other is varied
    parser.add_argument(
        "-aker",
        "--assumed-kernel-type",
        help="Kernel type to use in data-generation",
        nargs="*",
        default=["RBF", "Exp"],
        choices=["RBF", "Matern", "Exp"],
    )
    parser.add_argument(
        "-aks",
        "--assum-kernel-scale",
        help="Assumed value for output scale hyperparameter ",
        default=0.8,
        type=float,
    )
    parser.add_argument(
        "-al",
        "--assum-lengthscale",
        help="Assumed value for length scale hyperparameter",
        default=lengthscale,
        type=float,
    )
    parser.add_argument(
        "-anv",
        "--assum-noise",
        help="Assumed value for noise_var",
        default=0.2,
        type=float,
    )

    # Args for param to vary

    parser.add_argument(
        "-varpar",
        "--param_to_vary",
        help="parameter to be varied in sensitivity analysis",
        nargs="*",
        default="lengthscale",
        choices=["lengthscale", "noise", "outputscale"],
    )
    parser.add_argument(
        "-maxval",
        "--max_parval",
        help="Largest param value",
        default=5.0,
        type=float,
    )
    parser.add_argument(
        "-minval",
        "--min_parval",
        help="Smallest param value",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "-numvals",
        "--num_parvals",
        help="Number of values between limits",
        default=40,
        type=int,
    )

    # prediction params
    parser.add_argument(
        "-numnn",
        "--number-nn",
        help="Number of nearest neighbours used for prediction",
        default=400,
        type=int,
    )

    # Miscellaneous Args
    parser.add_argument(
        "-seed",
        "--random-seed",
        help="Random seed to initialise",
        default=42,
        type=int,
    )

    parser.add_argument(
        "-out",
        "--output_file",
        help="Name of output file",
        default="sim_gpnn_limits_results_test",
        type=str,
    )

    parser.add_argument(
        '-pmeth', '--pred-method', help='Prediction mechanism',
        default=make_full_predictions, type=Callable)  # type: ignore

    parser.add_argument(
        "-array_idx", "--array_index", help="", default=1, type=int
    )

    return parser.parse_args()


array_sets = cartesian_prod(
    [int(10 ** i) for i in range(4, 8)],
    ["RBF", "Exp"],
    [5, 15, 20, 50],
    [0.5, 1.0, 3.0])
# array_sets = cartesian_prod([10000], [
#                             "RBF"], [int(10**x) for x in np.linspace(1,5,50)], [0.5,1.0,3.0,5.0,10.0,30.0])
array_sets_dict = {
    k + 1:
    {'train_data_size': v[0],
     'true_kernel_type': v[1],
     'dimension': v[2],
     'lengthscale': v[3]} for k, v in enumerate(array_sets)}
array_sets_dict.update({0: {}})

args = parse_args()

array_set = array_sets_dict[args.array_index]

if args.array_index != 0:
    args.true_kernel_type = array_set["true_kernel_type"]
    args.train_data_size = array_set["train_data_size"]
    args.dimension = array_set["dimension"]
    args.true_lengthscale = array_set["lengthscale"]
    args.assum_lengthscale = array_set["lengthscale"]

act_kern = args.true_kernel_type  # default rbf
act_ks = args.true_kernel_scale  # default 0.8
act_ls = args.true_lengthscale  # default 1.0
act_nv = args.true_noise  # default 0.2
assum_ks = args.assum_kernel_scale  # default 0.8
assum_ls = args.assum_lengthscale  # default 1.0
assum_nv = args.assum_noise  # default 0.2
assum_kerns = args.assumed_kernel_type
if isinstance(assum_kerns, str):
    assum_kerns = [assum_kerns]
elif not isinstance(assum_kerns, list):
    raise TypeError("Require a list, or at least an Iterable.")

input_dim = args.dimension  # default 10
train_data_size = args.train_data_size  # default 100000
test_data_size = args.test_data_size  # default 1000
seed = args.random_seed  # default 42
num_nearest_neighbours = args.number_nn  # default 400
varypars = args.param_to_vary  # default lengthscale
if isinstance(varypars, str):
    varypars = [varypars]
elif not isinstance(varypars, list):
    raise TypeError("Require a list, or at least an Iterable.")
max_varypar = args.max_parval  # default 5.0
min_varypar = args.min_parval  # default 0.1
num_vals = args.num_parvals
prediction_method = args.pred_method
out_file_name = args.output_file

if args.xy_input_file is not None:
    xy_data_file = f"{data_path}{args.xy_input_file}/data.npy"

print("input dim = %d" % (input_dim))
print("kernel-type and parms used to generate data:")
print(act_kern)
print("act_ls = %f, act_ks = %f, act_nv =%f " % (act_ls, act_ks, act_nv))
print(
    "assum_ls = %f, assum_ks = %f, assum_nv =%f "
    % (assum_ls, assum_ks, assum_nv)
)
print("WILL BE VARYING FOLLOWING ASSUMED PARAMETER VALUE")
print(varypars)
print("over %d values between %f and %f" % (num_vals, min_varypar, max_varypar))
print("seed = %d" % (seed))
print("num nearest neighbours for prediction = %d" % (num_nearest_neighbours))
print("training data size = %d" % (train_data_size))
print("test data size = %d" % (test_data_size))

dim_scale_factor = 1

rng = np.random.default_rng(seed)  # being treated like global param

if args.xy_input_file is not None:
    data = np.load(xy_data_file, mmap_mode='c')
    input_data_size = data.shape[0]
    if train_data_size+test_data_size < input_data_size:
        perm0 = rng.permutation(input_data_size)[
            : (train_data_size + test_data_size)]
        data = data[perm0, :]
    else:
        train_data_size = input_data_size - test_data_size
    # data = rescale_dataset_noise(data, 0.1, rng)
    x_train = data[:train_data_size, :-1]
    y_train = data[:train_data_size, -1]
    x_test = data[train_data_size:train_data_size+test_data_size, :-1]
    y_test = data[train_data_size:train_data_size+test_data_size, -1]

    m_y, sd_y, m_x, prep_mat_x = get_xy_preprocess_tranforms(
        x_train, y_train, "whiten", train_data_size, input_dim)
    # Apply transform y <- (y-m_y)/sd_y and x <- inv_M.(x-m_x) to the x and y vals in training data
    x_train = preprocess_x_vals(x_train, m_x, prep_mat_x)
    y_train = preprocess_y_vals(y_train, m_y, sd_y)

    x_test = preprocess_x_vals(x_test, m_x, prep_mat_x)
    y_test = preprocess_y_vals(y_test, m_y, sd_y)

    input_dim = x_train.shape[1]
else:
    print("generate x vals", flush=True)
    z_train, z_test = generate_spherical_gaus_xvals(
        train_data_size, test_data_size, input_dim, rng
    )

    intrinsic_dim = z_train.shape[1]
    input_dim = dim_scale_factor * intrinsic_dim

    x_train = add_noisy_dimsx(z_train, input_dim, rng, "noise")
    x_test = add_noisy_dimsx(z_test, input_dim, rng,  "noise")


# being treated like global param
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# best_pos_mse = act_nv
# perfect_limit_nll = 0.5 * (1.0 + math.log(act_nv) + math.log(2.0 * math.pi))
# perfect_limit_mscal = 1.0

# set up neigh for prediction purposes (to be applied for all predictions henceforth)
print("generate nn table for nn prediction capability", flush=True)

tic = time.perf_counter()
neigh = NearestNeighbors(n_neighbors=num_nearest_neighbours)
neigh.fit(x_train)
toc = time.perf_counter()
duration = toc - tic
print(
    " nn table construction complete and took %.8f seconds" % (duration),
    flush=True,
)


# generate x,y predict points and nearest neighbour sets
tic = time.perf_counter()
if args.xy_input_file is not None:
    if prediction_method.__name__ == 'make_full_predictions':
        np_xsets, np_ysets, mnn_retrieve_time = get_xy_nn_prediction_subsets(
            x_test,
            y_test,
            test_data_size,
            x_train,
            y_train,
            train_data_size,
            num_nearest_neighbours,
            input_dim,
            neigh,
        )
    else:
        np_xsets, np_ysets, mnn_retrieve_time = np.vstack(
            [x_train, x_test]), np.hstack(
            [y_train, y_test]), 0.0
else:
    np_xsets, np_ysets, mnn_retrieve_time = generate_xy_nn_prediction_subsets2(
        x_test,
        x_train,
        z_test,
        z_train,
        num_nearest_neighbours,
        neigh,
        act_ls,
        act_ks,
        act_nv,
        act_kern,
        rng
    )

toc = time.perf_counter()
duration = toc - tic
print(
    " xy nn based sets contructed for all %d target s vals complete and took %f seconds"
    % (test_data_size, duration)
)
print(
    "average time per target x value to retrieve its nearest neighbours = %.8f seconds"
    % (mnn_retrieve_time),
    flush=True,
)

# now make predictions based on varying asumptions concerning the kernel family and hyperparameter values

# varypars = ["lengthscale"]
max_varypars = {"lengthscale": 20.0, "outputscale": 3.0, "noise": 1.0}
min_varypars = {"lengthscale": 0.1, "outputscale": 0.1, "noise": 0.01}

file_exists = os.path.isfile(out_file_name)

with open(out_file_name, "a+") as out_file:
    if not file_exists and args.array_index == 0:
        header = "n,n_test,d,m,seed,k_true,k_model,ks,ls,nv,assum_ks,assum_ls,assum_nv,varypar,mse,nll,mscal"
        print(header, file=out_file, flush=True)

    for assum_kern in assum_kerns:

        print(
            f"XXXXXXXXXXXX prediction performance using assumed ker = {assum_kern}",
            flush=True)
        for varypar in varypars:
            print(
                "with assum_ls = %f, assum_ks = %f, assum_nv = %f"
                % (assum_ls, assum_ks, assum_nv)
            )
            assum_vars = {
                "lengthscale": assum_ls,
                "outputscale": assum_ks,
                "noise": assum_nv,
            }
            act_vars = {
                "lengthscale": act_ls,
                "outputscale": act_ks,
                "noise": act_nv,
            }
            incrementer = inc_varypar(
                varypar, act_vars[varypar],
                num_vals, min_varypars[varypar],
                max_varypars[varypar])
            for i in range(num_vals):
                assum_vars[varypar] = next(incrementer)

                (
                    mse,
                    nll,
                    mscal
                ) = evaluate_sim_predictions(
                    np_xsets,
                    np_ysets,
                    test_data_size,
                    num_nearest_neighbours,
                    input_dim,
                    assum_vars,
                    assum_kern,
                    likelihood,
                    prediction_method,
                )

                line = f"{train_data_size},{test_data_size},{input_dim},{num_nearest_neighbours},{seed},{act_kern},{assum_kern},{act_ks},{act_ls},{act_nv},{assum_vars['outputscale']},{assum_vars['lengthscale']},{assum_vars['noise']},{varypar},{mse.item()},{nll.item()},{mscal.item()}"
                print(line, file=out_file, flush=True)
