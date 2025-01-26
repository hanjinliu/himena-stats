from __future__ import annotations
from typing import Literal

from himena import Parametric, StandardType, WidgetDataModel
from himena.widgets import SubWindow
from himena.plugins import register_function, configure_gui
from himena.utils.table_selection import range_getter
from himena.qt.magicgui import SelectionEdit

from himena_stats._lazy_import import stats
from himena_stats.consts import MENUS_TEST
from himena_stats.test_tools._utils import pvalue_to_asterisks, values_groups_to_xy


@register_function(
    menus=MENUS_TEST,
    title="T-test ...",
    types=[StandardType.TABLE, StandardType.DATAFRAME, StandardType.EXCEL],
    command_id="himena-stats:test:t-test",
)
def t_test(win: SubWindow) -> Parametric:
    """Run a Student's or Welch's t-test on a table-like data."""

    @configure_gui(
        a={"widget_type": SelectionEdit, "getter": range_getter(win)},
        b={"widget_type": SelectionEdit, "getter": range_getter(win)},
        groups={"widget_type": SelectionEdit, "getter": range_getter(win)},
    )
    def run_t_test(
        a,
        b=None,
        groups=None,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        kind: Literal["Student", "Welch", "F"] = "Student",
        f_threshold: float = 0.05,
    ):
        model = win.to_model()
        x0, y0 = values_groups_to_xy(model, [a, b], groups)
        if kind == "F":
            f_result = stats.f_oneway(x0.array, y0.array)
            if f_result.pvalue < f_threshold:
                kind = "Student"
            else:
                kind = "Welch"
        t_result = stats.ttest_ind(
            x0.array, y0.array, equal_var=kind == "Student", alternative=alternative
        )
        return _ttest_result_to_model(
            t_result,
            title=f"T-test result of {model.title}",
            rows=[["kind", kind], ["comparison", f"{x0.name} vs {y0.name}"]],
        )

    return run_t_test


@register_function(
    menus=MENUS_TEST,
    title="Paired T-test ...",
    types=[StandardType.TABLE, StandardType.DATAFRAME, StandardType.EXCEL],
    command_id="himena-stats:test:paired-t-test",
)
def paired_t_test(win: SubWindow) -> Parametric:
    @configure_gui(
        a={"widget_type": SelectionEdit, "getter": range_getter(win)},
        b={"widget_type": SelectionEdit, "getter": range_getter(win)},
        groups={"widget_type": SelectionEdit, "getter": range_getter(win)},
    )
    def run_t_test(
        a,
        b=None,
        groups=None,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ):
        model = win.to_model()
        x0, y0 = values_groups_to_xy(model, [a, b], groups)
        t_result = stats.ttest_rel(x0.array, y0.array, alternative=alternative)
        return _ttest_result_to_model(
            t_result,
            title=f"Paired T-test result of {model.title}",
            rows=[["comparison", f"{x0.name} vs {y0.name}"]],
        )

    return run_t_test


@register_function(
    menus=MENUS_TEST,
    title="Wilcoxon Test ...",
    types=[StandardType.TABLE, StandardType.DATAFRAME, StandardType.EXCEL],
    command_id="himena-stats:test:wilcoxon-test",
)
def wilcoxon_test(win: SubWindow) -> Parametric:
    @configure_gui(
        a={"widget_type": SelectionEdit, "getter": range_getter(win)},
        b={"widget_type": SelectionEdit, "getter": range_getter(win)},
        groups={"widget_type": SelectionEdit, "getter": range_getter(win)},
    )
    def run_wilcoxon_test(
        a,
        b=None,
        groups=None,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ):
        model = win.to_model()
        x0, y0 = values_groups_to_xy(model, [a, b], groups)
        w_result = stats.wilcoxon(x0.array, y0.array, alternative=alternative)
        w_result_table = [
            ["p-value", format(w_result.pvalue, ".5g")],
            ["", pvalue_to_asterisks(w_result.pvalue)],
            ["statistic", format(w_result.statistic, ".5g")],
        ]
        return WidgetDataModel(
            value=w_result_table,
            type=StandardType.TABLE,
            title=f"Wilcoxon Test result of {model.title}",
        )

    return run_wilcoxon_test


@register_function(
    menus=MENUS_TEST,
    title="Mann-Whitney U Test ...",
    types=[StandardType.TABLE, StandardType.DATAFRAME, StandardType.EXCEL],
    command_id="himena-stats:test:mann-whitney-u-test",
)
def mann_whitney_u_test(win: SubWindow) -> Parametric:
    @configure_gui(
        a={"widget_type": SelectionEdit, "getter": range_getter(win)},
        b={"widget_type": SelectionEdit, "getter": range_getter(win)},
        groups={"widget_type": SelectionEdit, "getter": range_getter(win)},
    )
    def run_mann_whitney_u_test(
        a,
        b=None,
        groups=None,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ):
        model = win.to_model()
        x0, y0 = values_groups_to_xy(model, [a, b], groups)
        u_result = stats.mannwhitneyu(x0.array, y0.array, alternative=alternative)
        u_result_table = [
            ["p-value", format(u_result.pvalue, ".5g")],
            ["", pvalue_to_asterisks(u_result.pvalue)],
            ["U-statistic", format(u_result.statistic, ".5g")],
        ]
        return WidgetDataModel(
            value=u_result_table,
            type=StandardType.TABLE,
            title=f"Mann-Whitney U Test result of {model.title}",
        )

    return run_mann_whitney_u_test


def _ttest_result_to_model(t_result, title: str, rows: list[list[str]] = ()):
    t_result_table = [
        ["p-value", format(t_result.pvalue, ".5g")],
        ["", pvalue_to_asterisks(t_result.pvalue)],
        ["t-statistic", format(t_result.statistic, ".5g")],
        ["degrees of freedom", int(t_result.df)],
    ] + rows
    return WidgetDataModel(
        value=t_result_table,
        type=StandardType.TABLE,
        title=title,
    )
