from __future__ import annotations
from typing import Literal

from himena import Parametric, StandardType, WidgetDataModel
from himena.widgets import SubWindow
from himena.plugins import configure_gui
from himena.utils.table_selection import range_getter, model_to_vals_arrays
from himena.qt.magicgui import SelectionEdit

from himena_stats._lazy_import import stats
from himena_stats.consts import ALTERNATIVE
from himena_stats.test_tools._utils import pvalue_to_asterisks


def shapiro_wilk_test(win: SubWindow) -> Parametric:
    """Run a Shapiro-Wilk test on a table-like data."""

    @configure_gui(
        sample_data={"widget_type": SelectionEdit, "getter": range_getter(win)},
    )
    def run_shapiro_wilk_test(sample_data: tuple[tuple[int, int], tuple[int, int]]):
        model = win.to_model()
        x0 = model_to_vals_arrays(model, [sample_data])[0]
        sw_result = stats.shapiro(x0.array)
        sw_result_table = [
            ["p-value", format(sw_result.pvalue, ".5g")],
            ["", pvalue_to_asterisks(sw_result.pvalue)],
            ["statistic", format(sw_result.statistic, ".5g")],
        ]
        return WidgetDataModel(
            value=sw_result_table,
            type=StandardType.TABLE,
            title=f"Shapiro-Wilk Test result of {model.title}",
        )

    return run_shapiro_wilk_test


def kolmogorov_smirnov_test(win: SubWindow) -> Parametric:
    """Run a Kolmogorov-Smirnov test on a table-like data."""

    @configure_gui(
        sample_data={"widget_type": SelectionEdit, "getter": range_getter(win)},
    )
    def run_kolmogorov_smirnov_test(
        sample_data: tuple[tuple[int, int], tuple[int, int]],
        cdf: Literal["norm", "expon", "uniform"] = "norm",
        alternative: ALTERNATIVE = "two-sided",
        method: Literal["auto", "exact", "approx", "asymp"] = "auto",
    ):
        """
        Run Kolmogorov-Smirnov test on a table-like data.

        Parameters
        ----------
        sample_data : tuple[tuple[int, int], tuple[int, int]]
            Range of the sample data.
        cdf : function, optional
            Cumulative distribution function, by default "norm"
        alternative : str, default "two-sided"
            Test alternative hypothesis. This parameteris forwarded to
            `scipy.stats.kstest`.
        method : str, optional
            The method to use to calculate the p-value. This parameteris forwarded to
            `scipy.stats.kstest`. Following is from the documentation of `stats.kstest`:

            * 'auto' : selects one of the other options.
            * 'exact' : uses the exact distribution of test statistic.
            * 'approx' : approximates the two-sided probability with twice the
                one-sided probability
            * 'asymp': uses asymptotic distribution of test statistic, by default "auto"
        """
        model = win.to_model()
        x0 = model_to_vals_arrays(model, [sample_data])[0]
        ks_result = stats.kstest(x0.array, cdf, alternative=alternative, method=method)
        ks_result_table = [
            ["p-value", format(ks_result.pvalue, ".5g")],
            ["", pvalue_to_asterisks(ks_result.pvalue)],
            ["statistic", format(ks_result.statistic, ".5g")],
        ]
        return WidgetDataModel(
            value=ks_result_table,
            type=StandardType.TABLE,
            title=f"Kolmogorov-Smirnov Test result of {model.title}",
        )

    return run_kolmogorov_smirnov_test
