from __future__ import annotations

import numpy as np
from himena_stats._lazy_import import stats


def fit_norm(obs: np.ndarray) -> stats.rv_frozen:
    loc, scale = stats.norm.fit(obs)
    return stats.norm(loc=loc, scale=scale)


def fit_gamma(obs: np.ndarray) -> stats.rv_frozen:
    a, loc, scale = stats.gamma.fit(obs)
    return stats.gamma(a=a, scale=scale)


def fit_expon(obs: np.ndarray) -> stats.rv_frozen:
    loc, scale = stats.expon.fit(obs)
    return stats.expon(scale=scale)


def fit_uniform(obs: np.ndarray) -> stats.rv_frozen:
    # NOTE: scipy.stats uses [loc, loc + scale] to specify a uniform distribution.
    loc, scale = stats.uniform.fit(obs)
    return stats.uniform(loc=loc, scale=scale)


def fit_beta(obs: np.ndarray) -> stats.rv_frozen:
    a, b, loc, scale = stats.beta.fit(obs)
    return stats.beta(a=a, b=b)


def fit_cauchy(obs: np.ndarray) -> stats.rv_frozen:
    loc, scale = stats.cauchy.fit(obs)
    return stats.cauchy(loc=loc, scale=scale)


def fit_t(obs: np.ndarray) -> stats.rv_frozen:
    df, loc, scale = stats.t.fit(obs)
    return stats.t(df=df, loc=loc, scale=scale)


def fit_chi2(obs: np.ndarray) -> stats.rv_frozen:
    df, loc, scale = stats.chi2.fit(obs)
    return stats.chi2(df=df, loc=loc, scale=scale)


def fit_binom(obs: np.ndarray) -> stats.rv_frozen:
    mean = obs.mean()
    std = obs.std()
    p = 1 - std**2 / mean
    n = mean / p
    return stats.binom(n=n, p=p)


def fit_poisson(obs: np.ndarray) -> stats.rv_frozen:
    mu = obs.mean()
    return stats.poisson(mu=mu)


def fit_dist(obs, dist: stats.rv_frozen) -> stats.rv_frozen:
    dist_name: str = dist.dist.name
    if dist_name == "norm":
        return fit_norm(obs)
    elif dist_name == "gamma":
        return fit_gamma(obs)
    elif dist_name == "expon":
        return fit_expon(obs)
    elif dist_name == "uniform":
        return fit_uniform(obs)
    elif dist_name == "beta":
        return fit_beta(obs)
    elif dist_name == "cauchy":
        return fit_cauchy(obs)
    elif dist_name == "t":
        return fit_t(obs)
    elif dist_name == "chi2":
        return fit_chi2(obs)
    elif dist_name == "binom":
        return fit_binom(obs)
    elif dist_name == "poisson":
        return fit_poisson(obs)
    raise NotImplementedError
