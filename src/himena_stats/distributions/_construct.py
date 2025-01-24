from scipy import stats

from himena import Parametric, StandardType, WidgetDataModel
from himena.widgets import SubWindow
from himena.plugins import register_function, configure_gui
from himena.utils.table_selection import range_getter, model_to_vals_arrays
from himena.qt.magicgui import SelectionEdit
from himena_stats.consts import MENUS_DIST


@register_function(
    menus=MENUS_DIST,
    title="Normal Distribution ...",
    command_id="himena-stats:dist-construct:continuous:norm",
)
def dist_norm() -> Parametric:
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
    @configure_gui
    def construct_gamma(a: float = 1.0, scale: float = 1.0):
        return WidgetDataModel(
            value=stats.gamma(a, scale=scale),
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
    @configure_gui
    def construct_beta(a: float = 2.0, b: float = 2.0):
        return WidgetDataModel(
            value=stats.beta(a, b),
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
    @configure_gui
    def construct_cauchy(loc: float = 0.0, scale: float = 1.0):
        return WidgetDataModel(
            value=stats.cauchy(loc, scale),
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
    @configure_gui
    def construct_t(df: float = 1.0):
        return WidgetDataModel(
            value=stats.t(df),
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
    @configure_gui
    def construct_binom(n: int = 10, p: float = 0.5):
        return WidgetDataModel(
            value=stats.binom(n, p),
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
    @configure_gui
    def construct_geom(p: float = 0.5):
        return WidgetDataModel(
            value=stats.geom(p),
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
def fit_dist(win: SubWindow) -> Parametric:
    @configure_gui(
        observations={"widget_type": SelectionEdit, "getter": range_getter(win)},
        param_as_guess={"label": "Use current parameters as the initial guess"},
    )
    def run_fit(observations, param_as_guess: bool = False):
        model = win.to_model()
        obs = model_to_vals_arrays(model, [observations])[0]
        dist = model.value
        if param_as_guess:
            guess = dist.kwds
        else:
            guess = None
        dist_fitted = stats.fit(dist.dist, obs.array, guess=guess)
        return WidgetDataModel(
            value=dist_fitted,
            type=StandardType.DISTRIBUTION,
            title=f"{model.title} fitted",
        )

    return run_fit


@register_function(
    menus=MENUS_DIST,
    title="Random Sampling ...",
    types=StandardType.DISTRIBUTION,
    command_id="himena-stats:dist-convert:sample",
)
def sample_dist(win: SubWindow) -> Parametric:
    def run_sample(sample_size: list[int] = (100,)):
        """Sample from the distribution.

        Parameters
        ----------
        sample_size : list of int
            The size of the sample. Can be a verctor for n-dimentional sampling.
        """
        model = win.to_model()
        dist = model.value
        samples = dist.rvs(size=sample_size)
        return WidgetDataModel(
            value=samples,
            type=StandardType.ARRAY,
            title=f"Samples from {model.title}",
        )

    return run_sample
