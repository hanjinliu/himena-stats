from himena.plugins import register_widget_class
from himena import StandardType
from himena_stats.distributions import _construct, _methods
from himena_stats.distributions._widget import QDistributionView


def _register_widgets():
    register_widget_class(StandardType.DISTRIBUTION, QDistributionView)


_register_widgets()

del _construct, _methods, _register_widgets
