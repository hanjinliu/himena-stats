from typing import TypeVar
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from magicgui.widgets.bases import ValueWidget
from himena import Parametric, StandardType, WidgetDataModel
from himena.widgets import SubWindow
from himena.plugins import register_function, configure_gui
from himena.data_wrappers import wrap_dataframe
from himena.standards import plotting as hplt
from himena.standards.model_meta import TableMeta
from himena_stats.consts import MENUS_DIST
from himena_stats.distributions._utils import draw_pdf_or_pmf, infer_edges

OBS_TYPES = [StandardType.TABLE, StandardType.ARRAY, StandardType.DATAFRAME]


@register_function(
    menus=MENUS_DIST,
    title="Normal Distribution ...",
    command_id="himena-stats:dist-construct:continuous:norm",
)
def dist_norm() -> Parametric:
    """Construct normal distribution."""

    @configure_gui
    def construct_norm(mu: float = 0.0, sigma: float = 1.0):
        return WidgetDataModel(
            value=stats.norm(loc=mu, scale=sigma),
            type=StandardType.DISTRIBUTION,
            title="Normal",
        )

    return construct_norm


@register_function(
    menus=MENUS_DIST,
    title="Uniform Distribution ...",
    command_id="himena-stats:dist-construct:continuous:uniform",
)
def dist_uniform() -> Parametric:
    """Construct uniform distribution."""

    @configure_gui
    def construct_uniform(a: float = 0.0, b: float = 1.0):
        return WidgetDataModel(
            value=stats.uniform(loc=a, scale=b - a),
            type=StandardType.DISTRIBUTION,
            title="Uniform",
        )

    return construct_uniform


@register_function(
    menus=MENUS_DIST,
    title="Exponential Distribution ...",
    command_id="himena-stats:dist-construct:continuous:expon",
)
def dist_expon() -> Parametric:
    """Construct exponential distribution."""

    @configure_gui
    def construct_expon(scale: float = 1.0):
        return WidgetDataModel(
            value=stats.expon(scale=scale),
            type=StandardType.DISTRIBUTION,
            title="Exponential",
        )

    return construct_expon


@register_function(
    menus=MENUS_DIST,
    title="Gamma Distribution ...",
    command_id="himena-stats:dist-construct:continuous:gamma",
)
def dist_gamma() -> Parametric:
    """Construct Gamma distribution."""

    @configure_gui
    def construct_gamma(a: float = 1.0, scale: float = 1.0):
        return WidgetDataModel(
            value=stats.gamma(a=a, scale=scale),
            type=StandardType.DISTRIBUTION,
            title="Gamma",
        )

    return construct_gamma


@register_function(
    menus=MENUS_DIST,
    title="Beta Distribution ...",
    command_id="himena-stats:dist-construct:continuous:beta",
)
def dist_beta() -> Parametric:
    """Construct Beta distribution."""

    @configure_gui
    def construct_beta(a: float = 2.0, b: float = 2.0):
        return WidgetDataModel(
            value=stats.beta(a=a, b=b),
            type=StandardType.DISTRIBUTION,
            title="Beta",
        )

    return construct_beta


@register_function(
    menus=MENUS_DIST,
    title="Cauchy Distribution ...",
    command_id="himena-stats:dist-construct:continuous:cauchy",
)
def dist_cauchy() -> Parametric:
    """Construct Cauchy distribution."""

    @configure_gui
    def construct_cauchy(loc: float = 0.0, scale: float = 1.0):
        return WidgetDataModel(
            value=stats.cauchy(loc=loc, scale=scale),
            type=StandardType.DISTRIBUTION,
            title="Cauchy",
        )

    return construct_cauchy


@register_function(
    menus=MENUS_DIST,
    title="T Distribution ...",
    command_id="himena-stats:dist-construct:continuous:t",
)
def dist_t() -> Parametric:
    """Construct t distribution."""

    @configure_gui
    def construct_t(df: float = 1.0):
        return WidgetDataModel(
            value=stats.t(df=df),
            type=StandardType.DISTRIBUTION,
            title="T",
        )

    return construct_t


@register_function(
    menus=MENUS_DIST,
    title="Binomial Distribution ...",
    command_id="himena-stats:dist-construct:discrete:binom",
)
def dist_binom() -> Parametric:
    """Construct binomial distribution."""

    @configure_gui
    def construct_binom(n: int = 10, p: float = 0.5):
        return WidgetDataModel(
            value=stats.binom(n=n, p=p),
            type=StandardType.DISTRIBUTION,
            title="Binomial",
        )

    return construct_binom


@register_function(
    menus=MENUS_DIST,
    title="Poisson Distribution ...",
    command_id="himena-stats:dist-construct:discrete:poisson",
)
def dist_poisson() -> Parametric:
    """Construct Poisson distribution."""

    @configure_gui
    def construct_poisson(lambda_: float = 5.0):
        return WidgetDataModel(
            value=stats.poisson(mu=lambda_),
            type=StandardType.DISTRIBUTION,
            title="Poisson",
        )

    return construct_poisson


@register_function(
    menus=MENUS_DIST,
    title="Geometric Distribution ...",
    command_id="himena-stats:dist-construct:discrete:geom",
)
def dist_geom() -> Parametric:
    """Construct geometric distribution."""

    @configure_gui
    def construct_geom(p: float = 0.5):
        return WidgetDataModel(
            value=stats.geom(p=p),
            type=StandardType.DISTRIBUTION,
            title="Geometric",
        )

    return construct_geom


@register_function(
    menus=MENUS_DIST,
    title="Fit Distribution ...",
    types=StandardType.DISTRIBUTION,
    command_id="himna-stats:dist-convert:fit",
)
def fit_dist(model: WidgetDataModel) -> Parametric:
    """Fit distribution model to observations."""

    @configure_gui(
        obs={"types": OBS_TYPES, "label": "observations"},
        obs_range={"bind": _get_range},
        param_as_guess={"label": "Use current parameters as the initial guess"},
    )
    def run_fit(
        obs: WidgetDataModel,
        obs_range: tuple[tuple[int, int], tuple[int, int]],
        param_as_guess: bool = False,
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
        if param_as_guess:
            guess = dist.kwds
        else:
            guess = None
        # Bounds of "loc" and "scale" must be set in stats.fit Here we set them wide
        # enough.
        arr_min, arr_max = arr.min(), arr.max()
        arr_dif = arr_max - arr_min
        bounds = {"loc": (arr_min, arr_max), "scale": (arr_dif / 10000, arr_dif * 10)}
        fit_result = stats.fit(dist.dist, arr.ravel(), guess=guess, bounds=bounds)
        dist_fitted = dist.dist(**fit_result.params._asdict())
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
    command_id="himena-stats:dist-convert:plot",
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
                )
            else:
                values, counts = np.unique_counts(arr)
                density = counts / arr.size
                fig.bar(values, density, color="skyblue")
            xlow = min(xlow, arr.min())
            xhigh = max(xhigh, arr.max())

        x, y = draw_pdf_or_pmf(dist, xlow, xhigh)
        fig.plot(x, y, width=2.5, color="red")
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
    command_id="himena-stats:dist-convert:sample",
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


def _get_range(widget: ValueWidget):
    model: WidgetDataModel = widget.parent["obs"].value
    if not isinstance(meta := model.metadata, TableMeta):
        return None
    if len(meta.selections) != 1:
        raise ValueError(f"Data {model.title} must contain single selection.")
    return meta.selections[0]


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
