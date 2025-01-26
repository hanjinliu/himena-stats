from typing import TypeVar
import numpy as np
from numpy.typing import NDArray
from magicgui.widgets.bases import ValueWidget
from himena import Parametric, StandardType, WidgetDataModel
from himena.widgets import SubWindow
from himena.plugins import register_function, configure_gui
from himena.data_wrappers import wrap_dataframe
from himena.standards import plotting as hplt
from himena.standards.model_meta import TableMeta
from himena_stats._lazy_import import stats
from himena_stats.consts import MENUS_DIST
from himena_stats.distributions._utils import draw_pdf_or_pmf, infer_edges
from himena_stats.distributions._fit import fit_dist

OBS_TYPES = [StandardType.TABLE, StandardType.ARRAY, StandardType.DATAFRAME]


@register_function(
    menus=MENUS_DIST,
    title="Fit Distribution ...",
    types=StandardType.DISTRIBUTION,
    command_id="himna-stats:dist-convert:fit-mle",
)
def fit_mle(model: WidgetDataModel) -> Parametric:
    """Fit distribution model to observations by maximum likelihood estimation."""

    @configure_gui(
        obs={"types": OBS_TYPES, "label": "observations"},
        obs_range={"bind": _get_range},
    )
    def run_fit(
        obs: WidgetDataModel,
        obs_range: tuple[tuple[int, int], tuple[int, int]],
    ) -> WidgetDataModel:
        """
        Parameters
        ----------
        obs : WidgetDataModel, optional
            Observations to which model will be fit. This value can be a table or an
            array, optionally with a selection area that specifies the range.
        param_as_guess : bool, default False
            If checked, the current parameter of the distribution will be used as the
            initial guess. Otherwise, only the distribution model will be considered.
        """
        dist: "stats.rv_frozen" = model.value
        dtype = np.float64 if hasattr(dist, "pdf") else np.int64
        arr = _norm_obs(obs, obs_range, np.dtype(dtype))
        dist_fitted = fit_dist(arr, dist)
        return WidgetDataModel(
            value=dist_fitted,
            type=StandardType.DISTRIBUTION,
            title=f"{model.title} fitted",
        )

    return run_fit


@register_function(
    menus=MENUS_DIST,
    title="Plot Distribution ...",
    types=StandardType.DISTRIBUTION,
    command_id="himena-stats:dist-plot:plot",
)
def plot_dist(win: SubWindow) -> Parametric:
    """Plot distribution with observations."""

    @configure_gui(
        obs={"types": OBS_TYPES, "label": "observations"},
        obs_range={"bind": _get_range},
    )
    def run_plot(obs: WidgetDataModel | None, obs_range) -> WidgetDataModel:
        """Plot distribution, optional with the observations.

        Parameters
        ----------
        obs : WidgetDataModel, optional
            If given, this observation data will also be plotted as a histogram. This
            value can be a table or an array, optionally with a selection area that
            specifies the range to plot.
        """
        model = win.to_model()
        dist: "stats.rv_frozen" = model.value
        xlow, xhigh = infer_edges(dist)
        is_continuous = hasattr(dist, "pdf")
        fig = hplt.figure()
        if obs is not None:
            dtype = np.float64 if is_continuous else np.int64
            arr = _norm_obs(obs, obs_range, np.dtype(dtype))
            if is_continuous:
                fig.hist(
                    arr,
                    bins=min(int(np.sqrt(arr.size)), 64),
                    stat="density",
                    color="skyblue",
                    name="observations",
                )
            else:
                values, counts = np.unique_counts(arr)
                density = counts / arr.size
                fig.bar(values, density, color="skyblue", name="observations")
            xlow = min(xlow, arr.min())
            xhigh = max(xhigh, arr.max())

        x, y = draw_pdf_or_pmf(dist, xlow, xhigh)
        fig.plot(x, y, width=2.5, color="red", name="model")
        return WidgetDataModel(
            value=fig,
            type=StandardType.PLOT,
            title=f"Plot of {model.title}",
        )

    return run_plot


@register_function(
    menus=MENUS_DIST,
    title="Random Sampling ...",
    types=StandardType.DISTRIBUTION,
    command_id="himena-stats:dist:sample",
)
def sample_dist(win: SubWindow) -> Parametric:
    """Random sampling from a distribution."""
    random_state_default = np.random.randint(0, 10000)

    @configure_gui(random_state={"value": random_state_default})
    def run_sample(sample_size: list[int] = (100,), random_state: int | None = None):
        """Sample from the distribution.

        Parameters
        ----------
        sample_size : list of int
            The size of the sample. Can be a verctor for n-dimentional sampling.
        """
        model = win.to_model()
        dist: "stats.rv_frozen" = model.value
        samples = dist.rvs(size=sample_size, random_state=random_state)
        return WidgetDataModel(
            value=samples,
            type=StandardType.ARRAY,
            title=f"Samples from {model.title}",
        )

    return run_sample


@register_function(
    menus=MENUS_DIST,
    title="Show statistics",
    types=StandardType.DISTRIBUTION,
    command_id="himena-stats:dist:stats",
)
def show_stats(model: WidgetDataModel) -> WidgetDataModel:
    """Show statistics of the distribution."""
    dist: "stats.rv_frozen" = model.value
    value = [
        ["min", dist.a],
        ["mean", dist.mean()],
        ["median", dist.median()],
        ["max", dist.b],
        ["std", dist.std()],
        ["entropy", dist.entropy()],
    ]
    return WidgetDataModel(
        value=value,
        type=StandardType.TABLE,
        title=f"Statistics of {model.title}",
    )


def _get_range(widget: ValueWidget):
    model: WidgetDataModel = widget.parent["obs"].value
    if not isinstance(meta := model.metadata, TableMeta):
        return None
    if len(meta.selections) != 1:
        raise ValueError(f"Data {model.title} must contain single selection.")
    return meta.selections[0]


@register_function(
    menus=MENUS_DIST,
    title="Plot CDF",
    types=StandardType.DISTRIBUTION,
    command_id="himena-stats:dist-plot:cdf",
)
def plot_cdf(model: WidgetDataModel) -> WidgetDataModel:
    """Plot the cumulative distribution function of the model."""
    dist: "stats.rv_frozen" = model.value
    xlow, xhigh = infer_edges(dist)
    x = np.linspace(xlow, xhigh, 256)
    y = dist.cdf(x)
    fig = hplt.figure()
    fig.plot(x, y, color="red", name=model.title)
    return WidgetDataModel(
        value=fig, type=StandardType.PLOT, title=f"CDF of {model.title}"
    )


@register_function(
    menus=MENUS_DIST,
    title="Plot Survival Function",
    types=StandardType.DISTRIBUTION,
    command_id="himena-stats:dist-plot:sf",
)
def plot_sf(model: WidgetDataModel) -> WidgetDataModel:
    """Plot the survival function of the model."""
    dist: "stats.rv_frozen" = model.value
    xlow, xhigh = infer_edges(dist)
    x = np.linspace(xlow, xhigh, 256)
    y = dist.sf(x)
    fig = hplt.figure()
    fig.plot(x, y, color="red", name=model.title)
    return WidgetDataModel(
        value=fig, type=StandardType.PLOT, title=f"Survival function of {model.title}"
    )


_D = TypeVar("_D", bound=np.generic)


def _norm_obs(obs: WidgetDataModel, obs_range, dtype: np.dtype[_D]) -> NDArray[_D]:
    if obs_range is None:
        obs_slice = slice(None)
    else:
        rinds, cinds = obs_range
        obs_slice = slice(*rinds), slice(*cinds)
    if obs.is_subtype_of(StandardType.TABLE):
        arr = obs.value[obs_slice].astype(dtype)
    elif obs.is_subtype_of(StandardType.DATAFRAME):
        rsl, csl = obs_slice
        if csl.start - csl.stop != 1:
            raise ValueError("Only single-column selection is allowed")
        df = wrap_dataframe(obs.value)
        arr = df[rsl, csl.start].astype(dtype)
    elif obs.is_subtype_of(StandardType.ARRAY):
        arr = obs.value[obs_slice].astype(dtype)
    else:
        raise NotImplementedError
    return arr
