from __future__ import annotations

from functools import cache
import json
from pathlib import Path

from himena import StandardType, WidgetDataModel
from himena.plugins import register_reader_plugin, register_writer_plugin
from himena_stats.io._serialize import serialize_rv, deserialize_rv
from himena_stats._lazy_import import stats


@cache
def rv_frozen_type() -> type[stats.rv_frozen]:
    """Return the stats.rv_frozen type (because rv_frozen is not public in runtime)."""
    return type(stats.uniform())


@register_reader_plugin
def read_distribution(path: Path) -> WidgetDataModel:
    with path.open() as f:
        js = json.load(f)
    rv = deserialize_rv(js)
    return WidgetDataModel(
        value=rv,
        type=StandardType.DISTRIBUTION,
        title=path.name,
    )


@read_distribution.mark_matcher
def _(path: Path):
    if path.suffixes == [".dist", ".json"]:
        return StandardType.DISTRIBUTION
    return None


@register_writer_plugin
def write_distribution(model: WidgetDataModel, path: Path):
    js = serialize_rv(model.value)
    with open(path, mode="w") as f:
        json.dump(js, f)


@write_distribution.mark_matcher
def _(model: WidgetDataModel, path: Path):
    return isinstance(model.value, rv_frozen_type())
