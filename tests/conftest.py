import pytest

@pytest.fixture(scope="session", autouse=True)
def install_plugin(request: pytest.FixtureRequest):
    import himena_stats.distributions  # noqa: F401
    import himena_stats.io  # noqa: F401
    import himena_stats.test_tools  # noqa: F401
