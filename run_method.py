import math
import numpy as np
import os
import time
from sklearn.neighbors import NearestNeighbors
# import pynndescent

from gpybench.utils import wrapped_partial
from typing import Callable, Optional, List, Dict
import argparse

from gpprediction.models import map_param_keys, map_simple_params_to_full
from gpprediction.prediction import *
from gpprediction.preprocess import *
from gpprediction.estimation import *
from gpprediction.eval import evaluate_predictions

data_path1 = ".tmp_data/UCI datasets/array_datasets/"
data_path2 = "../synthetic-datasets/ciq_synthetic_datasets/noise_var=0.008/"

data_path = data_path2

make_sparse_grid_predictions = wrapped_partial(
    make_sparse_predictions, sampling_strategy="grid")

# default_m = {"sparse": 1024, "nn": 400}
INPUT_DIM = 100
lengthscale = 0.5

def parse_args(args: Optional[List] = None):
    parser = argparse.ArgumentParser()

    # Input and output file args
    parser.add_argument('-xyinfl', '--xy_input_file',
                        help='xy vals input file',
                        default=f'DIM{{{INPUT_DIM}}}_LENSCALE{{{lengthscale}}}_{{1}}')
    parser.add_argument('-resfl', '--results_file',
                        help='Results output file',
                        default='results_output_file_ciq.csv')

    # type of x preprocsseing to be applied
    parser.add_argument(
        '-xprep', '--x_preprocess',
        help='type of x preprocessing to be applied', default='whiten',
        choices=['axis_rescale', 'whiten', 'none'])

    # Num test and recalibration points args
    parser.add_argument('-n_recal', '--recal-data-size',
                        help='Size of recalibration data set',
                        default=1000, type=int)
    parser.add_argument('-n_test', '--test-data-size',
                        help='Size of test data set',
                        default=1000, type=int)
    parser.add_argument('-n_test_cap', '--test-data-size_cap',
                        help='Cap on size of test data set',
                        default=100000000, type=int)

    # Assumed kernel family
    parser.add_argument('-aker', '--assum-kernel-type',
                        help='Assumed kernel type',
                        default='RBF', choices=['RBF', 'Matern', 'Exp'])

    # some estimation and prediction parameters
    parser.add_argument('-ssize', '--subset-size',
                        help='Size of subsets used in param-estimation',
                        default=300, type=int)
    parser.add_argument('-maxns', '--max-nsubsets',
                        help='Max num subsets used in param-estimation',
                        default=10, type=int)
    parser.add_argument('-numnn', '--number-nn',
                        help='Number of nearest neighbours used for prediction',
                        default=400, type=int)

    parser.add_argument(
        '-pmeth', '--pred-method', help='Prediction mechanism',
        default=make_nn_predictions, type=Callable)  # type: ignore

    # Miscellaneous Args
    parser.add_argument(
        '-seed', '--random-seed',
        help='Random seed to initialise data split, random subset selection  etc',
        default=1, type=int)

    parser.add_argument('-cheat', '--cheat-flag',
                        help='Flag to decide whether to use cheating params.',
                        default=False, type=bool)

    parser.add_argument(
        '-force', '--force-est',
        help='Flag to decide whether to force param estimation rather than lookup.',
        default=True, type=bool)

    parser.add_argument(
        '-cheat_params', '--cheat-params',
        help='If cheating, use these parameters.',
        default={'noise': 0.03, 'outputscale': 0.98, 'lengthscale': 7},
        type=dict)  # type: ignore

    return parser.parse_args(args)


args = parse_args()
assum_kern = args.assum_kernel_type  # default rbf
recal_data_size = args.recal_data_size  # default 1000
test_data_size = args.test_data_size  # default 1000
# default 100000000 - can be used to keep test time manageable on my laptop
test_data_size_lim = args.test_data_size_cap
xpreprocess = args.x_preprocess  # type of pre-proceesing to apply to xvals
seed = args.random_seed  # default 42
subset_size = args.subset_size  # default 300
max_num_subsets = args.max_nsubsets  # default 10
NUM_NEAREST_NEIGHBOURS = args.number_nn  # default 400
xy_data_file = f"{data_path}{args.xy_input_file}/data.npy"
res_file = args.results_file  # results file
prediction_method = args.pred_method
cheat_params = args.cheat_params
force_est = args.force_est
cheat = args.cheat_flag

# dataset_name = re.findall(r"\/(\w+)\/data\.npy", xy_data_file)[0]
dataset_name = args.xy_input_file

likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1.000E-06))

print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
print('kernel-type used for param estimation and prediction :')
print(assum_kern)
print('seed = %d' % (seed))
print('for param estimation num-subsets = %d, subset size = %d (fewer subsets used if training data too small)' %
      (max_num_subsets, subset_size))
print('num nearest neighbours for prediction = %d' % (NUM_NEAREST_NEIGHBOURS))
print('requested test data size = %d' % (test_data_size))
print('recalibration data size = %d' % (recal_data_size))
print('type of preprocessing applied to x values:')
print(xpreprocess)
print('xy data input file')
print(xy_data_file)
print('results file')
print(res_file)

rng = np.random.default_rng(seed)

# << (needed to deal with small nv's that arise)
# may be overkill, but this is what I used for my synthetic dataset evals so far
num_adam_iters = 200

xy_data_array = np.load(xy_data_file, mmap_mode='c')

if "noise_var=0.008" in xy_data_file:
    xy_data_array = rescale_dataset_noise(xy_data_array, 0.1, rng)
    xy_data_array = add_noisy_dims(
        xy_data_array, (xy_data_array.shape[1]-1)*2, rng, "noise2")

# shrink dataset (by random selection) to see if GPnn does worse
# SIZE=5000
# perm0 = rng.permutation(xy_data_array.shape[0])[:SIZE]
# xy_data_array = xy_data_array[perm0,:]

nrows, ncols = xy_data_array.shape
print('in original npy  data file nrows = %d, ncols =%d' % (nrows, ncols))
xy_data_array = xy_data_array[~np.isnan(xy_data_array).any(axis=1)]
nrows, ncols = xy_data_array.shape
print('after removal of any bad rows,  npy  data file nrows = %d, ncols =%d' %
      (nrows, ncols))

x_data_set = xy_data_array[:, :-1]
y_data_set = xy_data_array[:, -1]

total_data_size, input_dim = x_data_set.shape
print('total_data_size = %d, input_dim  = %d' % (total_data_size, input_dim))

perm = rng.permutation(total_data_size)

train_data_size = total_data_size - (recal_data_size + test_data_size)

if (test_data_size > test_data_size_lim):
    # temporary measure to be removed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    test_data_size = test_data_size_lim
    print('!!!!!!!!!!!!!!reduced test_data_size  to %d, but kept same train_data_size = %d, and same recal_data_size = %d' % (
        test_data_size, train_data_size, recal_data_size))


y_train = y_data_set[perm[:train_data_size]]
x_train = x_data_set[perm[:train_data_size], :]

y_recal = y_data_set[perm[train_data_size:train_data_size+recal_data_size]]
x_recal = x_data_set[perm[train_data_size:train_data_size+recal_data_size], :]

y_test = y_data_set[perm[train_data_size +
                         recal_data_size:train_data_size+recal_data_size+test_data_size]]
x_test = x_data_set[perm[train_data_size +
                         recal_data_size:train_data_size+recal_data_size+test_data_size], :]


num_subsets = math.floor(float(train_data_size)/float(subset_size))
if (num_subsets > max_num_subsets):
    num_subsets = max_num_subsets
print('num_subsets = %d' % (num_subsets))


train_time = 0.0

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# preprocess the (x,y) training values  (in the case of x this might optionally either be 'prewhitening' or axis rescaling)
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
print('preprocess training data', flush=True)
tic = time.perf_counter()
# derive mean and sd of y train values, mean of x train components, inverse
# covar of x train cpts, and diag matrix with 1/(sd x-cpt) along the diag
if data_path == data_path2 and xpreprocess != 'none':
    m_y, sd_y, m_x, prep_mat_x = get_xy_preprocess_tranforms(
    x_train, y_train, xpreprocess, train_data_size, 1)
else:
    m_y, sd_y, m_x, prep_mat_x = get_xy_preprocess_tranforms(
        x_train, y_train, xpreprocess, train_data_size, input_dim)
# Apply transform y <- (y-m_y)/sd_y and x <- inv_M.(x-m_x) to the x and y vals in training data
x_train = preprocess_x_vals(x_train, m_x, prep_mat_x)
y_train = preprocess_y_vals(y_train, m_y, sd_y)
toc = time.perf_counter()
set_up_time = toc - tic
print('set_up time to preprocess training data = %f' % (set_up_time))
train_time += set_up_time
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


# class SparseParams(object):
#     def __init__(self, n_inducing_pts):
#         self.n_inducing_pts = n_inducing_pts


# if prediction_method.__name__ in ('make_nn_predictions',
# 'make_sparse_nn_predictions'):
if 'nn' in prediction_method.__name__ or prediction_method.__name__ == 'make_ensemble_predictions':
    print('generate nn table for nn prediction capability')
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    tic = time.perf_counter()
    p_norm = 2  # 2 - (input_dim > 20)  # guesswork atm
    neigh = NearestNeighbors(n_neighbors=NUM_NEAREST_NEIGHBOURS, p=p_norm)
    neigh.fit(x_train)
    # neigh = pynndescent.NNDescent(x_train, n_neighbors=NUM_NEAREST_NEIGHBOURS)
    # neigh.prepare()
    toc = time.perf_counter()
    nn_table_time = toc - tic
    print('nn_table_time =  %.8f seconds' % (nn_table_time))
    train_time += nn_table_time
else:
    neigh = SparseParams(n_inducing_pts=NUM_NEAREST_NEIGHBOURS,
                         n_subset_pts=NUM_NEAREST_NEIGHBOURS)
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

print('Phase 1 parameter estimation')
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
tic = time.perf_counter()
x_phase1 = x_train[0:(num_subsets*subset_size), ::]
y_phase1 = y_train[0:(num_subsets*subset_size)]

params_to_exc = {"nu": 0.2}

if cheat:
    model = get_model(x_phase1, y_phase1, likelihood, assum_kern, 'exact')
    params = map_simple_params_to_full(model, cheat_params)
    params = get_param_estimates(
        x_phase1, y_phase1, likelihood, subset_size, num_subsets,
        num_adam_iters, assum_kern, res_file, force_est, dataset_name, params)
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
else:
    model = get_model(x_phase1, y_phase1, likelihood, assum_kern, 'exact')
    params_to_exc = map_simple_params_to_full(model, params_to_exc)
    params = get_param_estimates(
        x_phase1, y_phase1, likelihood, subset_size, num_subsets,
        num_adam_iters, assum_kern, res_file, force_est, dataset_name, params_to_exc)
    if prediction_method.__name__ == 'make_var_predictions':
        model = get_model(x_phase1, y_phase1, likelihood, assum_kern,
                          'sparse', n_inducing_pts=NUM_NEAREST_NEIGHBOURS)
        model.set_parameters(params)
        with gpytorch.settings.debug(False):
            params = model.optimise_inducing_pts(x_phase1, y_phase1)
    # params = get_param_avg(x_phase1,y_phase1,input_dim,subset_size,num_subsets,num_adam_iters,assum_kern)

    toc = time.perf_counter()
    time_phase1 = toc - tic
    print(simplify_parameters(params))
    train_time += time_phase1
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    # Phase 2 parameter calibration (only needed for improved uncertainty measures)
    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    print('Phase2 estimation: compute mscal from calibation data in order to enable subsequent calibration')
    # compute mscal on calibration data in order to be able to perform recalibration
    tic = time.perf_counter()
    # Apply preprocessing transform to the x,y vals in the calibration data
    x_recal = preprocess_x_vals(x_recal, m_x, prep_mat_x)
    y_recal = preprocess_y_vals(y_recal, m_y, sd_y)
    pred_mean_recal, pred_sd_recal = prediction_method(
        x_recal, x_train, y_train, likelihood, params, neigh, assum_kern)
    mse, nll, mscal = evaluate_predictions(
        pred_mean_recal, pred_sd_recal, y_recal)
    toc = time.perf_counter()
    time_phase2 = toc - tic
    print('obtained mscal = %f (to be used for recalibration purposes) ' % (mscal))
    print('[on calibration data: mse = %.5f,  nll = %.5f]' % (mse, nll))
    print('phase2 time (for calibration) = %f' % (time_phase2))
    train_time += time_phase2

    # note the recalibration below where est_ks and est_nv are scaled by the phase2
    # estimated mscal value
    simple_params = simplify_parameters(params)
    param_map = map_param_keys(params)
    params.update(
        {param_map['outputscale']: mscal * params[param_map['outputscale']],
         param_map['noise']: mscal * params[param_map['noise']]})
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

print('overall train_time = %f' % (train_time))

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# obtain predictions from the test data
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
print('use  parameters to obtain predictions from test data')
tic = time.perf_counter()
# Apply preprocessing transform to the x vals in the test data
x_test = preprocess_x_vals(x_test, m_x, prep_mat_x)

final_params = simplify_parameters(params)

pred_mean_test, pred_sd_test = prediction_method(
    x_test, x_train, y_train, likelihood, params, neigh, assum_kern)
toc = time.perf_counter()
tot_predict_time = toc-tic
per_predict_time = tot_predict_time / float(test_data_size)
print('tot_predict_time = %f, per_predict_time = %f ' %
      (tot_predict_time, per_predict_time))
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
print('find performance of those predictions on the test data')
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# first on preprocessed y vals:
prep_y_test = preprocess_y_vals(y_test, m_y, sd_y)
mse, nll, mscal = evaluate_predictions(
    pred_mean_test, pred_sd_test, prep_y_test)
print('for preprocessed test y vals : mse = %.5f,  nll = %.5f, test data mscal = %.5f ' %
      (mse, nll, mscal))
# rmse_million, nll_million, time_million, compute_million = get_million_paper_results(
#     dataset_name)
# print('xy datafile =', xy_data_file , file = out_file, end ='')
# print( '     kernel =', assum_kern, '   seed = %d'  %(seed), file = out_file)
# print('mse = %.5f,  nll = %.5f, mscal = %.5f, rmse = %.5f,  rmse_million = %f, nll_million = %f, per_predict_time = %f, train_time = %f '  %(mse, nll, mscal, mse**.5, rmse_million, nll_million, per_predict_time, train_time), file = out_file, flush = True)
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

file_exists = os.path.isfile(res_file)

with open(res_file, 'a+') as out_file:
    if not file_exists:
        header = "n,n_test,d,m,seed,dataset,k_model,est_ks,est_ls,est_nv,est_nu,mse,nll,mscal, per_predict_time, train_time, pred_method"
        print(header, file=out_file, flush=True)

    if 'nu' not in final_params.keys():
        final_params['nu'] = np.nan

    line = f"{train_data_size},{test_data_size},{input_dim},{NUM_NEAREST_NEIGHBOURS},{seed},{dataset_name},{assum_kern},{final_params['outputscale']},{final_params['lengthscale']},{final_params['noise']},{final_params['nu']},{mse.item()},{nll.item()},{mscal.item()}, {per_predict_time},{train_time}, {prediction_method.__name__}"
    print(line, file=out_file, flush=True)

# now transform the predictive means and sds to be compatible with the original unpreprocessed y vals
# nn_pred_mean_test = (sd_y*nn_pred_mean_test) + m_y
# nn_pred_sd_test = sd_y * nn_pred_sd_test
# mse, nll, mscal = evaluate_predictions(nn_pred_mean_test, nn_pred_sd_test, test_data_size, y_test)
# print('for unpreprocessed test y vals: mse = %.5f,  nll = %.5f, test data mscal = %.5f '  %(mse, nll, mscal))
