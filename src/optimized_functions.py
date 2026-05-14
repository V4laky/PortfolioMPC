import numpy as np
from numba import njit


@njit
def ewma_1d(x, span):
    alpha = 2.0 / (span + 1.0)

    out = np.empty_like(x)
    out[:] = np.nan

    acc = x[0]

    for i in range(len(x)):
        acc = alpha * x[i] + (1.0 - alpha) * acc

        if i >= span - 1:
            out[i] = acc

    return out

@njit
def ewma_2d(x, span):
    alpha = 2.0 / (span + 1.0)

    T, N = x.shape
    out = np.empty((T, N))

    for j in range(N):
        acc = x[0, j]

        for i in range(T):
            acc = alpha * x[i, j] + (1-alpha) * acc
            out[i, j] = acc

    return out


@njit
def simulate_gjr_garch(
    nobs,
    burn,
    mu,
    phi,
    omega,
    alpha,
    gamma,
    beta,
    z,
    initial_return=0.0,
    initial_vol=None,
):
    """
    GJR-GARCH(1,1,1) with AR(1) mean.

    Parameters
    ----------
    z : ndarray
        Pre-generated standardized innovations.
        Shape: (nobs + burn,)
    """

    T = nobs + burn

    returns = np.empty(T)
    eps = np.empty(T)
    sigma2 = np.empty(T)

    # unconditional variance
    if initial_vol is None:
        unc_var = omega / (1.0 - alpha - beta - 0.5 * gamma)
        sigma2[0] = unc_var
    else:
        sigma2[0] = initial_vol ** 2

    returns[0] = initial_return
    eps[0] = np.sqrt(sigma2[0]) * z[0]

    for t in range(1, T):

        leverage = gamma * eps[t - 1] ** 2 if eps[t - 1] < 0 else 0.0

        sigma2[t] = (
            omega
            + alpha * eps[t - 1] ** 2
            + leverage
            + beta * sigma2[t - 1]
        )

        sigma = np.sqrt(sigma2[t])

        eps[t] = sigma * z[t]

        returns[t] = mu + phi * returns[t - 1] + eps[t]

    return (
        returns[burn:],
        np.sqrt(sigma2[burn:])
    )
