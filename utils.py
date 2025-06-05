import numpy as np
import warnings
from scipy.spatial import distance_matrix
from scipy.special import gamma, kv
from functools import partial, singledispatch
from scipy import linalg, optimize


def k_true(sigma: float, l: float, xp: float, xq: float) -> float:
    return sigma*np.exp(-0.5*np.dot(xp-xq, xp-xq)/l**2)


def k_se(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    return sigma * np.exp(-distance_matrix(x1, x2)**2/(2*ls**2))


def k_per(x1: np.ndarray, x2: np.ndarray, sigma, ls, p) -> np.ndarray:
    return sigma * np.exp(-2*np.sin(np.pi/p * distance_matrix(x1, x2)/(ls**2))**2)


def k_mat_half(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    return sigma * np.exp(-distance_matrix(x1, x2)/ls)


def k_mat_3half(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    D = distance_matrix(x1, x2)
    return sigma * (1 + np.sqrt(3) * D/ls) * np.exp(-np.sqrt(3)*D/ls)


def k_mat_5half(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    D = distance_matrix(x1, x2)
    return sigma * (1 + np.sqrt(5) * D/ls + 5/3 * D**2/ls**2) * np.exp(-np.sqrt(5)*D/ls)


def k_mat(x1: np.ndarray, x2: np.ndarray, sigma, ls, nu) -> np.ndarray:
    if nu == 0.5:
        return k_mat_half(x1, x2, sigma, ls)
    if nu == 1.5:
        return k_mat_3half(x1, x2, sigma, ls)
    if nu == 2.5:
        return k_mat_5half(x1, x2, sigma, ls)
    if nu >= 1000:
        warnings.warn("Large nu; treating as squared exp.")
        return k_se(x1, x2, sigma, ls)
    else:
        D = distance_matrix(x1, x2)
        return sigma * 2**(1-nu) / gamma(nu) * (np.sqrt(2*nu) * D/ls)**nu * kv(nu, np.sqrt(2*nu)*D/ls)


@singledispatch
def inv(M: np.ndarray, thresh=1e-9) -> np.ndarray:
    """Matrix inverse using SVD.

    Args:
        M (np.ndarray): Input matrix
        thresh (_type_, optional): To avoid numerical issues. Defaults to 1e-9.

    Returns:
        np.ndarray: Matrix inverse.
    """
    U, s, V = linalg.svd(M)
    sinv = 1/s
    sinv[s < thresh] = 0.0
    Minv = U @ np.diag(sinv) @ V
    return Minv


def invert_dict(a: dict) -> dict:
    """Creates a new dict from an input dict by swapping the keys and values around

    Args:
        a (dict): _description_

    Returns:
        dict: _description_
    """
    return dict((v, k) for k, v in a.items())


@singledispatch
def noti(i: int, n: int) -> list:
    """ returns list of integers in 1..n excl. i

    Args:
        i (int): integer to exclude
        n (int): length of integers

    Returns:
        list: 1,...,i-1,i+1,...,n
    """
    return list(set(range(n)).difference(set([i])))


def centred_geomspace(minval: float, maxval: float, num_vals: int,
                      centre: float) -> np.ndarray:
    """ Like np.geomspace, creates geometrically spaced points, but centred on a
    point given.

    Args:
        minval (float): minimum of interval
        maxval (float): max. of interval
        num_vals (int): number of points
        centre (float): central point around which points are spaced.

    Returns:
        np.ndarray: points
    """
    if maxval == centre:
        vals = minval+maxval - np.geomspace(maxval, minval, num_vals)
    elif minval == centre:
        vals = np.geomspace(maxval, minval, num_vals)
    else:
        # num_below = int(num_vals / ((maxval-minval) / centre))
        num_below = num_vals // 2
        min_to_mid = centre - np.geomspace(minval,
                                           centre, num=num_below+1) + minval
        mid_to_max = centre + np.geomspace(
            minval, maxval - centre + minval, num=num_vals -
            num_below) - minval
        vals = np.concatenate([min_to_mid, mid_to_max[1:]])
    vals.sort()
    return vals


def round_ordermag(x: float, method=np.round, astype=int):
    """ Rounds a float x to the nearest order of magnitude. (Or floors, if
    np.floor used, etc)

    Args:
        x (float): Float to round
        method (_type_, optional): rounding function. Defaults to np.round.
        astype (_type_, optional): Type for output. Defaults to int.

    Returns:
        _type_: Rounded x of type determined by astype.
    """
    order = np.floor(np.log10(x))
    dec = x/10**order
    return (method(dec)*10**order).astype(astype)


def inc_varypar(
        varypar: str, act_par: float, num_vals: int,
        min_varypar: float,
        max_varypar: float,
        method="geom"):
    """ Generator for next parameter value when varying in simulations.

    Args:
        varypar (str): Parameter name, e.g. 'lengthscale'
        act_par (float): Value of true (generative) parameter
        num_vals (int): Number of incremements to generate
        min_varypar (float): Min. value to generate
        max_varypar (float): Max. value to generate.
        method (str, optional): Linear or geometric spacing. Defaults to "geom".

    Raises:
        ValueError: If other method specified.

    Yields:
        _type_: sequence of floats between min_varypar and max_varypar centred
        on act_par.
    """
    if num_vals == 1:
        yield act_par
    if method == "geom":
        # geometrically space, but with most points centred on true param
        varypars = centred_geomspace(
            min_varypar, max_varypar, num_vals, act_par)
    elif method in ("lin", "linear"):
        varypars = np.linspace(min_varypar, max_varypar, num_vals)
    else:
        raise ValueError(
            "Method not recognised. Use either 'geom' or 'lin' (uses numpy)")
    for varypar in sorted(varypars):
        yield varypar

def get_off_diagonal(a: np.ndarray) -> np.ndarray:
    """Return array of off-diagonal elements of a

    Args:
        a (np.ndarray): input array

    Raises:
        TypeError: [description]

    Returns:
        np.ndarray: [description]
    """
    n, m = a.shape
    if n != m:
        raise TypeError("Array must be square.")
    return a[np.where(~np.eye(n, dtype=bool))]