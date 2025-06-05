import time
import numpy as np
from typing import Tuple

from gpprediction.utils import k_se, k_mat_half as k_exp, k_mat_3half, k_mat, k_per


def generate_spherical_gaus_xvals(train_data_size: int, test_data_size: int, input_dim: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Sample x~ N(0,(1/d)I_d) with E||X||=1

    Args:
        train_data_size (int): n
        test_data_size (int): n*
        input_dim (int): d
        rng (_type_): RNG

    Returns:
        Tuple[np.ndarray, np.ndarray]: Xtrain, Xtest
    """
    tic = time.perf_counter()
    mean_x = np.zeros([input_dim], dtype=np.float64)
    cov_x = np.identity(input_dim)/float(input_dim)
    data_size = train_data_size + test_data_size
    # x = rng.multivariate_normal(mean_x, cov_x, data_size).astype(np.float64)
    x = rng.standard_normal((data_size, input_dim))/float(input_dim) + mean_x
    x_train = x[:train_data_size, :]
    x_test = x[train_data_size:, :]
    toc = time.perf_counter()
    print('time to generate all required xvals = %f' % (toc-tic))
    return x_train, x_test


def get_subset_yvals(
        all_subset_xvals: np.ndarray, combined_subset_size: int, act_kern: str, act_ls: float,
        act_nv: float, act_ks: float, rng) -> np.ndarray:
    """Generate y values for x data using kernel + params supplied (from MVN)

    Args:
        all_subset_xvals (np.ndarray): covariates
        combined_subset_size (int): size of output vector
        act_kern (str): e.g. 'RBF'
        act_ls (float): lengthscale
        act_nv (float): noise var
        act_ks (float): kernel/outputscale
        rng (_type_): RNG

    Returns:
        np.ndarray: generated y values
    """
    ymean = np.zeros([combined_subset_size], dtype=np.float64)
    ycovar = gen_ycovar(act_ks, act_nv, act_ls, all_subset_xvals,
                        combined_subset_size, act_kern)
    y_vals = rng.multivariate_normal(ymean, ycovar)
    return y_vals


def oakoh04(xx):
    a1 = np.asarray(
        [
            [
                0.0118,
                0.0456,
                0.2297,
                0.0393,
                0.1177,
                0.3865,
                0.3897,
                0.6061,
                0.6159,
                0.4005,
                1.0741,
                1.1474,
                0.7880,
                1.1242,
                1.1982,
            ]
        ]
    )
    a2 = np.asarray(
        [
            [
                0.4341,
                0.0887,
                0.0512,
                0.3233,
                0.1489,
                1.0360,
                0.9892,
                0.9672,
                0.8977,
                0.8083,
                1.8426,
                2.4712,
                2.3946,
                2.0045,
                2.2621,
            ]
        ]
    )
    a3 = np.asarray(
        [
            [
                0.1044,
                0.2057,
                0.0774,
                0.2730,
                0.1253,
                0.7526,
                0.8570,
                1.0331,
                0.8388,
                0.7970,
                2.2145,
                2.0382,
                2.4004,
                2.0541,
                1.9845,
            ]
        ]
    )
    M = np.asarray(
        [
            [
                -0.022482886,
                -0.18501666,
                0.13418263,
                0.36867264,
                0.17172785,
                0.13651143,
                -0.44034404,
                -0.081422854,
                0.71321025,
                -0.44361072,
                0.50383394,
                -0.024101458,
                -0.045939684,
                0.21666181,
                0.055887417,
            ],
            [
                0.25659630,
                0.053792287,
                0.25800381,
                0.23795905,
                -0.59125756,
                -0.081627077,
                -0.28749073,
                0.41581639,
                0.49752241,
                0.083893165,
                -0.11056683,
                0.033222351,
                -0.13979497,
                -0.031020556,
                -0.22318721,
            ],
            [
                -0.055999811,
                0.19542252,
                0.095529005,
                -0.28626530,
                -0.14441303,
                0.22369356,
                0.14527412,
                0.28998481,
                0.23105010,
                -0.31929879,
                -0.29039128,
                -0.20956898,
                0.43139047,
                0.024429152,
                0.044904409,
            ],
            [
                0.66448103,
                0.43069872,
                0.29924645,
                -0.16202441,
                -0.31479544,
                -0.39026802,
                0.17679822,
                0.057952663,
                0.17230342,
                0.13466011,
                -0.35275240,
                0.25146896,
                -0.018810529,
                0.36482392,
                -0.32504618,
            ],
            [
                -0.12127800,
                0.12463327,
                0.10656519,
                0.046562296,
                -0.21678617,
                0.19492172,
                -0.065521126,
                0.024404669,
                -0.096828860,
                0.19366196,
                0.33354757,
                0.31295994,
                -0.083615456,
                -0.25342082,
                0.37325717,
            ],
            [
                -0.28376230,
                -0.32820154,
                -0.10496068,
                -0.22073452,
                -0.13708154,
                -0.14426375,
                -0.11503319,
                0.22424151,
                -0.030395022,
                -0.51505615,
                0.017254978,
                0.038957118,
                0.36069184,
                0.30902452,
                0.050030193,
            ],
            [
                -0.077875893,
                0.0037456560,
                0.88685604,
                -0.26590028,
                -0.079325357,
                -0.042734919,
                -0.18653782,
                -0.35604718,
                -0.17497421,
                0.088699956,
                0.40025886,
                -0.055979693,
                0.13724479,
                0.21485613,
                -0.011265799,
            ],
            [
                -0.092294730,
                0.59209563,
                0.031338285,
                -0.033080861,
                -0.24308858,
                -0.099798547,
                0.034460195,
                0.095119813,
                -0.33801620,
                0.0063860024,
                -0.61207299,
                0.081325416,
                0.88683114,
                0.14254905,
                0.14776204,
            ],
            [
                -0.13189434,
                0.52878496,
                0.12652391,
                0.045113625,
                0.58373514,
                0.37291503,
                0.11395325,
                -0.29479222,
                -0.57014085,
                0.46291592,
                -0.094050179,
                0.13959097,
                -0.38607402,
                -0.44897060,
                -0.14602419,
            ],
            [
                0.058107658,
                -0.32289338,
                0.093139162,
                0.072427234,
                -0.56919401,
                0.52554237,
                0.23656926,
                -0.011782016,
                0.071820601,
                0.078277291,
                -0.13355752,
                0.22722721,
                0.14369455,
                -0.45198935,
                -0.55574794,
            ],
            [
                0.66145875,
                0.34633299,
                0.14098019,
                0.51882591,
                -0.28019898,
                -0.16032260,
                -0.068413337,
                -0.20428242,
                0.069672173,
                0.23112577,
                -0.044368579,
                -0.16455425,
                0.21620977,
                0.0042702105,
                -0.087399014,
            ],
            [
                0.31599556,
                -0.027551859,
                0.13434254,
                0.13497371,
                0.054005680,
                -0.17374789,
                0.17525393,
                0.060258929,
                -0.17914162,
                -0.31056619,
                -0.25358691,
                0.025847535,
                -0.43006001,
                -0.62266361,
                -0.033996882,
            ],
            [
                -0.29038151,
                0.034101270,
                0.034903413,
                -0.12121764,
                0.026030714,
                -0.33546274,
                -0.41424111,
                0.053248380,
                -0.27099455,
                -0.026251302,
                0.41024137,
                0.26636349,
                0.15582891,
                -0.18666254,
                0.019895831,
            ],
            [
                -0.24388652,
                -0.44098852,
                0.012618825,
                0.24945112,
                0.071101888,
                0.24623792,
                0.17484502,
                0.0085286769,
                0.25147070,
                -0.14659862,
                -0.084625150,
                0.36931333,
                -0.29955293,
                0.11044360,
                -0.75690139,
            ],
            [
                0.041494323,
                -0.25980564,
                0.46402128,
                -0.36112127,
                -0.94980789,
                -0.16504063,
                0.0030943325,
                0.052792942,
                0.22523648,
                0.38390366,
                0.45562427,
                -0.18631744,
                0.0082333995,
                0.16670803,
                0.16045688,
            ],
        ]
    )

    term1 = a1 @ xx.T
    term2 = a2 @ np.sin(xx.T)
    term3 = a3 @ np.cos(xx.T)
    term4 = np.sum(xx.T * (M @ xx.T), axis=0)
    y = term1 + term2 + term3 + term4
    return y.T


def get_oak_yvals(
    all_subset_xvals,
    combined_subset_size,
    input_dim,
    act_kern,
    act_ls,
    act_nv,
    act_ks,
    rng
):
    if input_dim != 15:
        raise ValueError("If using `Oak` need X to have 15 dimensions.")
    f = oakoh04(all_subset_xvals).flatten()
    noise = ((0.5 * act_nv) ** 0.5) * np.random.laplace(
        size=combined_subset_size, loc=0.0, scale=1.0
    )
    y = f + noise
    return y


def gen_ycovar(act_ks: float, act_nv: float, act_ls: float, x_double: np.ndarray, size: int, act_kern: str) -> np.ndarray:
    """Compute Gram matrix using kernel + params supplied

    Args:
        act_ks (float): kernrelscale
        act_nv (float): noise var
        act_ls (float): lengthscale
        x_double (np.ndarray): covariates (dtype double)
        size (int): size of output matrix (size x size)
        act_kern (str): e.g. 'RBF'

    Returns:
        np.ndarray: kernel/Gram matrix for y
    """
    y_covar = np.zeros([size, size], dtype=np.float64)
    if act_kern.lower() == "rbf":
        y_covar = k_se(x_double, x_double, act_ks, act_ls)
    if act_kern.lower() == "matern":
        y_covar = k_mat_3half(x_double, x_double, act_ks, act_ls)
    if act_kern.lower() == "exp":
        y_covar = k_exp(x_double, x_double, act_ks, act_ls)
    y_covar += act_nv * np.identity(size)
    return y_covar


def get_xy_nn_prediction_subsets(x_predict: np.ndarray, y_predict: np.ndarray,
                                 predict_data_size: int,
                                 x_train: np.ndarray,
                                 y_train: np.ndarray,
                                 train_data_size: int,
                                 num_nearest_neighbours: int,
                                 input_dim: int,
                                 neigh,
                                 **kwargs
                                 ) -> Tuple[np.ndarray,np.ndarray,float]:
    """ For each test point find m nearest neighbours, and then select the y
    values corresponding to those from y_train

    Args:
        x_predict (np.ndarray): test locations
        y_predict (np.ndarray): test obs.
        predict_data_size (int): n*
        x_train (np.ndarray): train locs.
        y_train (np.ndarray): train obs.
        train_data_size (int): n
        num_nearest_neighbours (int): m
        input_dim (int): d
        neigh (_type_): NN object with precomputed dist. matrix

    Returns:
        Tuple[np.ndarray,np.ndarray,float]: NN covariates, NN obs, time taken
    """
    np_this_xset = np.zeros(
        [(num_nearest_neighbours + 1), input_dim], dtype=np.float64
    )
    np_predict_x = np.zeros([1, input_dim], dtype=np.float64)
    np_xsets = np.zeros(
        [(predict_data_size * (num_nearest_neighbours + 1)), input_dim],
        dtype=np.float64,
    )
    np_ysets = np.zeros(
        [(predict_data_size * (num_nearest_neighbours + 1))], dtype=np.float64
    )
    mnn_retrieve_time = 0.0
    # first construct x sets
    for i_nnsubset in range(predict_data_size):
        if (i_nnsubset % 10) == 0:
            print("generating nn_xysubsets for predict point %d" % (i_nnsubset))
        np_predict_x = x_predict[np.newaxis, i_nnsubset, :]
        tic = time.perf_counter()
        neigh_list = neigh.kneighbors(np_predict_x, return_distance=False)
        np_this_xset[:-1, :] = np_xsets[
            (i_nnsubset * (num_nearest_neighbours + 1))
            + np.asarray(range(num_nearest_neighbours)),
            :,
        ] = x_train[neigh_list[0], :]
        np_this_xset[-1, :] = np_xsets[
            (i_nnsubset * (num_nearest_neighbours + 1))
            + num_nearest_neighbours,
            :,
        ] = np_predict_x
        toc = time.perf_counter()
        mnn_retrieve_time += toc - tic
        # now construct y sets based on actual ('correct') kernel family and actual ('correct') param vals
        y_vals = np.concatenate(
            [y_train[neigh_list[0]], y_predict[np.newaxis, i_nnsubset]])
        np_ysets[
            (i_nnsubset * (num_nearest_neighbours + 1))
            + np.asarray(range(num_nearest_neighbours+1))
        ] = y_vals
    mnn_retrieve_time /= float(predict_data_size)
    ans = np_xsets, np_ysets, mnn_retrieve_time
    return ans


def generate_xy_nn_prediction_subsets(
    x_predict: np.ndarray,
    x_train: np.ndarray,
    num_nearest_neighbours: int,
    neigh,
    act_ls: float,
    act_ks: float,
    act_nv: float,
    act_kern: str,
    rng,
    method="nn",
) -> Tuple[np.ndarray,np.ndarray,float]:
    """ For each test point find the m nearest neighbour covariates and then
    generate y values for them (using kernel and params given)Generate a concatination of predict_data_size sets, each of size (1+num_nearest_neighbour), both for x vals and for y vals
    # These are placed in np_xsets and np_ysets respectively

    Args:
        x_predict (np.ndarray): test locs.
        x_train (np.ndarray): train locs
        num_nearest_neighbours (int): m
        neigh (_type_): NN object (sklearn usually)
        act_ls (float): lengthscale
        act_ks (float): kernelscale
        act_nv (float): noise var
        act_kern (str): e.g. 'RBF'
        rng (_type_): RNG
        method (str, optional): how to generate y. Defaults to "nn".

    Returns:
        Tuple[np.ndarray,np.ndarray,float]: locs, obs, time
    """
    predict_data_size, input_dim = x_predict.shape
    np_this_xset = np.zeros(
        [(num_nearest_neighbours + 1), input_dim], dtype=np.float64
    )
    np_predict_x = np.zeros([1, input_dim], dtype=np.float64)
    np_xsets = np.zeros(
        [(predict_data_size * (num_nearest_neighbours + 1)), input_dim],
        dtype=np.float64,
    )
    np_ysets = np.zeros(
        [(predict_data_size * (num_nearest_neighbours + 1))], dtype=np.float64
    )
    mnn_retrieve_time = 0.0
    # first construct x sets
    for i_nnsubset in range(predict_data_size):
        if (i_nnsubset % 10) == 0:
            print("generating nn_xysubsets for predict point %d" % (i_nnsubset))
        np_predict_x = x_predict[np.newaxis, i_nnsubset, :]
        tic = time.perf_counter()
        neigh_list = neigh.kneighbors(np_predict_x, return_distance=False)
        np_this_xset[:-1, :] = np_xsets[
            (i_nnsubset * (num_nearest_neighbours + 1))
            + np.asarray(range(num_nearest_neighbours)),
            :,
        ] = x_train[neigh_list[0], :]
        np_this_xset[-1, :] = np_xsets[
            (i_nnsubset * (num_nearest_neighbours + 1))
            + num_nearest_neighbours,
            :,
        ] = np_predict_x
        toc = time.perf_counter()
        mnn_retrieve_time += toc - tic
        # now construct y sets based on actual ('correct') kernel family and
        # actual ('correct') param vals
        if method == "oak":
            y_vals = get_oak_yvals(
                np_this_xset,
                (num_nearest_neighbours + 1),
                input_dim,
                act_kern,
                act_ls,
                act_nv,
                act_ks,
                rng
            )
        elif method == "nn":
            y_vals = get_subset_yvals(
                np_this_xset,
                (num_nearest_neighbours + 1),
                act_kern,
                act_ls,
                act_nv,
                act_ks,
                rng
            )
        np_ysets[
            (i_nnsubset * (num_nearest_neighbours + 1))
            + np.asarray(range(num_nearest_neighbours + 1))
        ] = y_vals
    mnn_retrieve_time /= float(predict_data_size)
    ans = np_xsets, np_ysets, mnn_retrieve_time
    return ans

def generate_xy_nn_prediction_subsets2(
    x_predict,
    x_train,
    z_predict,
    z_train,
    num_nearest_neighbours,
    neigh_x,
    act_ls,
    act_ks,
    act_nv,
    act_kern,
    rng
):
    """
        Generate a concatination of predict_data_size sets, each of size (1+num_nearest_neighbour), both for x vals and for y vals
        These are placed in np_xsets and np_ysets respectively
        let Z be the latent intrinsic dim. variable and X the observed locations

        Find m NN indices for each test x* and z* point (since they may differ),
        N_x and N_z.
        Generate ym vector using the actual z points, but on the set of indices
        Nx, so that we have the y values drawn from the correct distribution,
        but only the marginal at the associated observed locations

        Might need to modify the eval step too
    """
    
    n_predict, dx = x_predict.shape
    n_predict, dz = z_predict.shape
    np_this_xset = np.zeros(
        [(num_nearest_neighbours + 1), dx], dtype=np.float64
    )
    np_this_zset = np.zeros(
        [(num_nearest_neighbours + 1), dz], dtype=np.float64
    )
    np_xsets = np.zeros(
        [(n_predict * (num_nearest_neighbours + 1)), dx],
        dtype=np.float64,
    )
    np_ysets = np.zeros(
        [(n_predict * (num_nearest_neighbours + 1))], dtype=np.float64
    )
    mnn_retrieve_time = 0.0
    # first construct x sets
    for i_nnsubset in range(n_predict):
        if (i_nnsubset % 10) == 0:
            print("generating nn_xysubsets for predict point %d" % (i_nnsubset))
        np_predict_x = x_predict[np.newaxis, i_nnsubset, :]
        np_predict_z = z_predict[np.newaxis, i_nnsubset, :]
        tic = time.perf_counter()
        neigh_list_x = neigh_x.kneighbors(np_predict_x, return_distance=False)[0]
        # neigh_z = neigh_z.kneighbors(np_predict_z, return_distance=False)[0]
        # neigh_xz = [np.unique(x) for x in np.column_stack((neigh_x,neigh_z))]
        np_this_zset[:-1, :] = z_train[neigh_list_x, :]
        np_this_zset[-1, :] = np_predict_z
        np_this_xset[:-1, :] = np_xsets[
            (i_nnsubset * (num_nearest_neighbours + 1))
            + np.asarray(range(num_nearest_neighbours)),
            :,
        ] = x_train[neigh_list_x, :]
        np_this_xset[-1, :] = np_xsets[
            (i_nnsubset * (num_nearest_neighbours + 1))
            + num_nearest_neighbours,
            :,
        ] = np_predict_x
        toc = time.perf_counter()
        mnn_retrieve_time += toc - tic
        # now construct y sets based on actual ('correct') kernel family and
        # actual ('correct') param vals
        y_vals = get_subset_yvals(
            np_this_zset,
            (num_nearest_neighbours + 1),
            act_kern,
            act_ls,
            act_nv,
            act_ks,
            rng
        )
        np_ysets[
            (i_nnsubset * (num_nearest_neighbours + 1))
            + np.asarray(range(num_nearest_neighbours + 1))
        ] = y_vals
    mnn_retrieve_time /= float(n_predict)
    ans = np_xsets, np_ysets, mnn_retrieve_time
    return ans
