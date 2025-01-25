from scipy import stats
from himena import Parametric, StandardType, WidgetDataModel
from himena.plugins import register_function, configure_gui
from himena_stats.consts import MENUS_DIST


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
