from __future__ import annotations

import numpy as np
from himena import Parametric, StandardType, WidgetDataModel
from himena.widgets import SubWindow
from himena.plugins import register_function, configure_gui
from himena.utils.collections import OrderedSet
from himena.utils.table_selection import (
    model_to_col_val_arrays,
    model_to_vals_arrays,
    range_getter,
    NamedArray,
)
from himena.qt.magicgui import SelectionEdit

from scipy import stats
import scikit_posthocs as skp
from himena_stats.test_tools._consts import MENUS, TABLE_LIKE
from himena_stats.test_tools._utils import pvalue_to_asterisks


@register_function(
    menus=MENUS,
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
        arrs = _values_groups_to_arrays(model, values, groups)
        result = skp.posthoc_dscf([a.array for a in arrs])
        pvalues = result.to_numpy()
        return WidgetDataModel(
            value=_pval_matrix(pvalues, columns=[a.name for a in arrs]),
            type=StandardType.TABLE,
            title=f"Steel-Dwass test result of {model.title}",
        )

    return run_steel_dwass_test


@register_function(
    menus=MENUS,
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
        arrs = _values_groups_to_arrays(model, values, groups)
        result = stats.tukey_hsd(*[a.array for a in arrs])
        return WidgetDataModel(
            value=_pval_matrix(result.pvalue, columns=[a.name for a in arrs]),
            type=StandardType.TABLE,
            title=f"Tukey HSD test result of {model.title}",
        )

    return run_tukey_hsd_test


def _values_groups_to_arrays(
    model: WidgetDataModel, values: list, groups
) -> list[NamedArray]:
    if groups is None:
        arrs = model_to_vals_arrays(
            model,
            values,
            same_size=False,
        )
    else:
        if len(values) != 1:
            raise ValueError("If groups are given, values must be a single range.")
        col, val = model_to_col_val_arrays(model, groups, values[0])
        unique_values = OrderedSet(col.array)
        arrs = [
            NamedArray(str(uval), val.array[col.array == uval])
            for uval in unique_values
        ]
    return arrs


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
