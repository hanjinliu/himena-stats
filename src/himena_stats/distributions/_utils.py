from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from scipy import stats
    from numpy.typing import NDArray


def draw_pdf_or_pmf(
    dist: stats.rv_frozen,
    xlow: float,
    xhigh: float,
) -> tuple[NDArray[np.number], NDArray[np.number]]:
    if hasattr(dist, "pdf"):  # contiuous
        x0 = np.linspace(xlow, xhigh, 100)
        y0: np.ndarray = dist.pdf(x0)
        x = np.concatenate([[x0[0]], x0, [x0[-1]]])
        y = np.concatenate([[0.0], y0, [0.0]])
    elif hasattr(dist, "pmf"):  # discrete
        x0 = np.arange(int(xlow), int(xhigh) + 1)
        y0: np.ndarray = dist.pmf(x0)
        x = np.empty(4 * len(x0))
        x[0::4] = x[1::4] = x0 - 0.1
        x[2::4] = x[3::4] = x0 + 0.1
        y = np.zeros(4 * len(x0))
        y[1::4] = y0
        y[2::4] = y0
    else:
        raise TypeError(f"Type {type(dist)} not allowed.")
    return x, y


def infer_edges(dist: stats.rv_frozen) -> tuple[float, float]:
    if np.isfinite(dist.a):
        xlow = dist.a
    else:
        xlow = dist.ppf(0.001)
    if np.isfinite(dist.b):
        xhigh = dist.b
    else:
        xhigh = dist.ppf(0.999)
    return xlow, xhigh
