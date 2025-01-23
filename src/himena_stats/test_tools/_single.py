from __future__ import annotations
from typing import Literal

from himena import Parametric, StandardType, WidgetDataModel
from himena.widgets import SubWindow
from himena.plugins import register_function, configure_gui
from himena.utils.table_selection import model_to_xy_arrays, range_getter
from himena.qt.magicgui import SelectionEdit

from scipy import stats

from himena_stats.test_tools._consts import MENUS
from himena_stats.test_tools._utils import pvalue_to_asterisks


@register_function(
    menus=MENUS,
    title="T-test ...",
    types=[StandardType.TABLE, StandardType.DATAFRAME, StandardType.EXCEL],
    command_id="himena-stats:test:t-test",
)
def t_test(win: SubWindow) -> Parametric:
    """Run a Student's or Welch's t-test on a table-like data."""

    @configure_gui(
        a={"widget_type": SelectionEdit, "getter": range_getter(win)},
        b={"widget_type": SelectionEdit, "getter": range_getter(win)},
    )
    def run_t_test(
        a,
        b,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        kind: Literal["Student", "Welch", "F"] = "Student",
        f_threshold: float = 0.05,
    ):
        model = win.to_model()
        x0, y0 = model_to_xy_arrays(
            model,
            a,
            b,
            allow_empty_x=False,
            allow_multiple_y=False,
            same_size=False,
        )
        if kind == "F":
            f_result = stats.f_oneway(a, b)
            if f_result.pvalue < f_threshold:
                kind = "Student"
            else:
                kind = "Welch"
        t_result = stats.ttest_ind(
            x0.array, y0[0].array, equal_var=kind == "Student", alternative=alternative
        )
        return _ttest_result_to_model(
            t_result,
            title=f"T-test result of {model.title}",
            rows=[["kind", kind], ["comparison", f"{x0.name} vs {y0[0].name}"]],
        )

    return run_t_test


@register_function(
    menus=MENUS,
    title="Paired T-test ...",
    types=[StandardType.TABLE, StandardType.DATAFRAME, StandardType.EXCEL],
    command_id="himena-stats:test:paired-t-test",
)
def paired_t_test(win: SubWindow) -> Parametric:
    @configure_gui(
        a={"widget_type": SelectionEdit, "getter": range_getter(win)},
        b={"widget_type": SelectionEdit, "getter": range_getter(win)},
    )
    def run_t_test(
        a,
        b,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ):
        model = win.to_model()
        x0, y0 = model_to_xy_arrays(
            model,
            a,
            b,
            allow_empty_x=False,
            allow_multiple_y=False,
            same_size=True,
        )
        t_result = stats.ttest_rel(x0.array, y0[0].array, alternative=alternative)
        return _ttest_result_to_model(
            t_result,
            title=f"Paired T-test result of {model.title}",
            rows=[["comparison", f"{x0.name} vs {y0[0].name}"]],
        )

    return run_t_test


@register_function(
    menus=MENUS,
    title="Wilcoxon Test ...",
    types=[StandardType.TABLE, StandardType.DATAFRAME, StandardType.EXCEL],
    command_id="himena-stats:test:wilcoxon-test",
)
def wilcoxon_test(win: SubWindow) -> Parametric:
    @configure_gui(
        a={"widget_type": SelectionEdit, "getter": range_getter(win)},
        b={"widget_type": SelectionEdit, "getter": range_getter(win)},
    )
    def run_wilcoxon_test(
        a,
        b,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ):
        model = win.to_model()
        x0, y0 = model_to_xy_arrays(
            model,
            a,
            b,
            allow_empty_x=False,
            allow_multiple_y=False,
            same_size=False,
        )
        w_result = stats.wilcoxon(x0.array, y0[0].array, alternative=alternative)
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
    menus=MENUS,
    title="Mann-Whitney U Test ...",
    types=[StandardType.TABLE, StandardType.DATAFRAME, StandardType.EXCEL],
    command_id="himena-stats:test:mann-whitney-u-test",
)
def mann_whitney_u_test(win: SubWindow) -> Parametric:
    @configure_gui(
        a={"widget_type": SelectionEdit, "getter": range_getter(win)},
        b={"widget_type": SelectionEdit, "getter": range_getter(win)},
    )
    def run_mann_whitney_u_test(
        a,
        b,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ):
        model = win.to_model()
        x0, y0 = model_to_xy_arrays(
            model,
            a,
            b,
            allow_empty_x=False,
            allow_multiple_y=False,
            same_size=False,
        )
        u_result = stats.mannwhitneyu(x0.array, y0[0].array, alternative=alternative)
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


# @register_function(
#     menus="tools/stats",
#     title="ANOVA ...",
#     types=[StandardType.TABLE, StandardType.DATAFRAME, StandardType.EXCEL],
#     command_id="himena-stats:test:anova",
# )
# def anova(win: SubWindow) -> Parametric:
#     @configure_gui(
#         a={"widget_type": SelectionEdit, "getter": range_getter(win)},
#         b={"widget_type": SelectionEdit, "getter": range_getter(win)},
#     )
#     def run_anova(
#         a,
#         b,
#         f_threshold: float = 0.05,
#     ):
#         model = win.to_model()
#         x0, y0 = model_to_xy_arrays(
#             model, a, b, allow_empty_x=False, allow_multiple_y=True
#         )
#         f_result = stats.f_oneway(*[y.array for y in y0])
#         if f_result.pvalue < f_threshold:
#             return _ttest_result_to_model(f_result, title=f"ANOVA result of {model.title}")
#         else:
#             return WidgetDataModel(
#                 value="The null hypothesis is not rejected.",
#                 type=StandardType.STRING,
#                 title=f"ANOVA result of {model.title}",
#             )
#     return run_anova
