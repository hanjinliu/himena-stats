from typing import TYPE_CHECKING


class LazyScipyStats:
    def __getattr__(self, key: str):
        from scipy import stats

        return getattr(stats, key)


class LazyScikitPosthocs:
    def __getattr__(self, key: str):
        import scikit_posthocs

        return getattr(scikit_posthocs, key)


if TYPE_CHECKING:
    from scipy import stats
    import scikit_posthocs
else:
    stats = LazyScipyStats()
    scikit_posthocs = LazyScikitPosthocs()

__all__ = ["stats", "scikit_posthocs"]
