from __future__ import annotations

import numpy as np
from himena import Parametric, StandardType, WidgetDataModel
from himena.widgets import SubWindow
from himena.plugins import register_function, configure_gui
from himena.utils.table_selection import (
    range_getter,
)
from himena.qt.magicgui import SelectionEdit

from himena_stats._lazy_import import stats, scikit_posthocs
from himena_stats.consts import MENUS_TEST, TABLE_LIKE
from himena_stats.test_tools._utils import (
    pvalue_to_asterisks,
    values_groups_to_arrays,
    dropna,
)


@register_function(
    menus=MENUS_TEST,
    title="Steel-Dwass test ...",
    types=TABLE_LIKE,
    command_id="himena-stats:test-multi:steel-dwass",
)
def steel_dwass_test(win: SubWindow) -> Parametric:
    """Run a Steel-Dwass test on a table-like data."""
    selection_opt = {"widget_type": SelectionEdit, "getter": range_getter(win)}

    @configure_gui(
        values={
            "widget_type": "ListEdit",
            "options": selection_opt,
            "value": [None],
            "layout": "vertical",
        },
        groups=selection_opt,
    )
    def run_steel_dwass_test(values: list, groups):
        model = win.to_model()
        arrs = values_groups_to_arrays(model, values, groups)
        result = scikit_posthocs.posthoc_dscf([dropna(a) for a in arrs])
        pvalues = result.to_numpy()
        return WidgetDataModel(
            value=_pval_matrix(pvalues, columns=[a.name for a in arrs]),
            type=StandardType.TABLE,
            title=f"Steel-Dwass test result of {model.title}",
        )

    return run_steel_dwass_test


@register_function(
    menus=MENUS_TEST,
    title="Tukey's HSD test ...",
    types=TABLE_LIKE,
    command_id="himena-stats:test-multi:tukey-hsd",
)
def tukey_hsd_test(win: SubWindow) -> Parametric:
    """Run a Tukey's HSD test on a table-like data."""
    selection_opt = {"widget_type": SelectionEdit, "getter": range_getter(win)}

    @configure_gui(
        values={
            "widget_type": "ListEdit",
            "options": selection_opt,
            "value": [None],
            "layout": "vertical",
        },
        groups=selection_opt,
    )
    def run_tukey_hsd_test(values: list, groups):
        model = win.to_model()
        arrs = values_groups_to_arrays(model, values, groups)
        result = stats.tukey_hsd(*[dropna(a) for a in arrs])
        return WidgetDataModel(
            value=_pval_matrix(result.pvalue, columns=[a.name for a in arrs]),
            type=StandardType.TABLE,
            title=f"Tukey HSD test result of {model.title}",
        )

    return run_tukey_hsd_test


@register_function(
    menus="tools/stats",
    title="ANOVA ...",
    types=[StandardType.TABLE, StandardType.DATAFRAME, StandardType.EXCEL],
    command_id="himena-stats:test:anova",
)
def anova(win: SubWindow) -> Parametric:
    """Run an ANOVA on a table-like data."""
    selection_opt = {"widget_type": SelectionEdit, "getter": range_getter(win)}

    @configure_gui(
        values={
            "widget_type": "ListEdit",
            "options": selection_opt,
            "value": [None],
            "layout": "vertical",
        },
        groups=selection_opt,
    )
    def run_anova(values: list, groups):
        model = win.to_model()
        arrs = values_groups_to_arrays(model, values, groups)
        f_result = stats.f_oneway(*[dropna(a) for a in arrs])
        return WidgetDataModel(
            value=_pval_matrix(f_result.pvalue, columns=[a.name for a in arrs]),
            type=StandardType.TABLE,
            title=f"ANOVA result of {model.title}",
        )

    return run_anova


@register_function(
    menus=MENUS_TEST,
    title="Dunnett's test ...",
    types=TABLE_LIKE,
    command_id="himena-stats:test-multi:dunnett",
)
def dunnett_test(win: SubWindow) -> Parametric:
    """Run a Dunnett's test on a table-like data."""
    selection_opt = {"widget_type": SelectionEdit, "getter": range_getter(win)}

    @configure_gui(
        values={
            "widget_type": "ListEdit",
            "options": selection_opt,
            "value": [None],
            "layout": "vertical",
        },
        groups=selection_opt,
    )
    def run_dunnett_test(values: list, groups, control: str):
        model = win.to_model()
        arrs = values_groups_to_arrays(model, values, groups)
        if control == "":
            idx = 0
        else:
            for idx, each in enumerate(arrs):
                if each.name == control:
                    break
            else:
                raise ValueError(f"No group named {control!r}")
        treatments = [dropna(a) for a in arrs]
        control_arr = treatments.pop(idx)
        columns = [a.name for a in arrs]
        del columns[idx]
        result = scikit_posthocs.posthoc_dunnett(control_arr, treatments)
        pvalues = result.to_numpy()
        return WidgetDataModel(
            value=_pval_matrix(pvalues, columns=columns),
            type=StandardType.TABLE,
            title=f"Dunnett's test result of {model.title}",
        )

    return run_dunnett_test


def _pval_matrix(pvalues: np.ndarray, columns: list[str]):
    size = pvalues.shape[0]
    pvalues_str = np.zeros((size + 1, size + 1), dtype=np.dtypes.StringDType())
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            if i > j:
                pvalues_str[i, j] = pvalue_to_asterisks(pvalues[i - 1, j - 1])
            elif i == j:
                pvalues_str[i, j] = "1.0"
            else:
                pvalues_str[i, j] = format(pvalues[i - 1, j - 1], ".5g")
    pvalues_str[0, 1:] = columns
    pvalues_str[1:, 0] = columns
    return pvalues_str
