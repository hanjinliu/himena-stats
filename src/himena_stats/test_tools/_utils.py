from __future__ import annotations

from himena import WidgetDataModel
from himena.utils.collections import OrderedSet
from himena.utils.table_selection import (
    model_to_col_val_arrays,
    model_to_vals_arrays,
    NamedArray,
)


def pvalue_to_asterisks(pval: float) -> str:
    if pval > 0.05:
        return "n.s."
    elif pval > 0.01:
        return "*"
    elif pval > 0.001:
        return "**"
    elif pval > 0.0001:
        return "***"
    else:
        return "****"


def values_groups_to_arrays(
    model: WidgetDataModel,
    values: list[tuple[tuple[int, int], tuple[int, int]]],
    groups,
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


def values_groups_to_xy(
    model: WidgetDataModel,
    values: list[tuple[tuple[int, int], tuple[int, int]] | None],
    groups,
) -> tuple[NamedArray, NamedArray]:
    values = [val for val in values if val is not None]
    arrs = values_groups_to_arrays(model, values, groups)
    if len(arrs) != 2:
        raise ValueError(f"Expected two groups, found {len(arrs)}")
    return arrs[0], arrs[1]
