from __future__ import annotations

from typing import Any
from himena_stats._lazy_import import stats


def serialize_rv(rv: stats.rv_frozen) -> dict[str, Any]:
    return {"distribution": rv.dist.name, "parameters": rv.kwds.copy()}


def deserialize_rv(rv_dict: dict[str, Any]) -> stats.rv_frozen:
    dist_name = rv_dict["distribution"]
    params = rv_dict["parameters"]
    return getattr(stats, dist_name)(**params)
