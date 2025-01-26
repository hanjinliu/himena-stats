import pytest
from himena import MainWindow


@pytest.mark.parametrize(
    "command,params",
    [
        ("continuous:norm", {"mu": 3.2, "sigma": 4.5}),
        ("continuous:uniform", {"a": -2, "b": 3.5}),
        ("continuous:expon", {"scale": 2.1}),
        ("continuous:gamma", {"a": 3.0, "scale": 1.2}),
        ("continuous:beta", {"a": 2.0, "b": 2.0}),
        ("continuous:cauchy", {"loc": 1.0, "scale": 2.0}),
        ("continuous:t", {"df": 15}),
        ("discrete:binom", {"n": 20, "p": 0.4}),
        ("discrete:poisson", {"mu": 4.2}),
    ]
)
def test_construction_and_sampling(himena_ui: MainWindow, command: str, params: dict):
    himena_ui.exec_action(
        f"himena-stats:dist-construct:{command}", with_params=params,
    )
    win_dist = himena_ui.current_window
    himena_ui.exec_action("himena-stats:dist:stats")
    himena_ui.exec_action(
        "himena-stats:dist:sample",
        window_context=win_dist,
        with_params={"sample_size": 10, "random_state": 1234}
    )
    win_sample = himena_ui.current_window
    assert win_dist is not None
    assert win_sample is not None

    himena_ui.exec_action(
        "himna-stats:dist-convert:fit-mle",
        model_context=win_dist.to_model(),
        window_context=win_dist,
        with_params={
            "obs": win_sample.to_model(),
            "obs_range": None,
        },
    )
    himena_ui.exec_action(
        "himena-stats:dist-plot:plot",
        window_context=win_dist,
        with_params={"obs": win_sample.to_model(), "obs_range": None}
    )
