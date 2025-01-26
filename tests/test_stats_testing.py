import pytest
from himena import MainWindow, StandardType

table_data_sig = [
    ["a", "b"],
    [1.9, 4.4],
    [2.1, 5.3],
    [1.6, 4.6],
    [2.1, 4.1],
    [1.8, 5.9],
    [1.7, 5.3],
    ["", 5.1],
]


table_data_nonsig = [
    ["a", "b"],
    [4.9, 4.8],
    [5.1, 5.3],
    [4.6, 4.6],
    [5.1, 5.0],
    [5.8, 5.9],
    [5.7, 5.6],
    ["", 5.1],
]

def unpivot(val):
    out = []
    for r0, _ in val[1:]:
        if r0 == "":
            continue
        out.append(["a", r0])
    for _, r1 in val[1:]:
        out.append(["b", r1])
    return out

a_val = ((0, 100), (0, 1))
b_val = ((0, 100), (1, 2))

@pytest.mark.parametrize(
    "command",
    ["t-test", "mann-whitney-u-test"]
)
def test_a_vs_b(himena_ui: MainWindow, command: str):
    win_sig = himena_ui.add_object(table_data_sig, type=StandardType.TABLE)
    win_sig_unpivot = himena_ui.add_object(
        unpivot(table_data_sig), type=StandardType.TABLE
    )
    win_nonsig = himena_ui.add_object(table_data_nonsig, type=StandardType.TABLE)
    win_nonsig_unpivot = himena_ui.add_object(
        unpivot(table_data_nonsig), type=StandardType.TABLE
    )
    himena_ui.exec_action(
        f"himena-stats:test:{command}",
        window_context=win_sig,
        with_params={"a": a_val, "b": b_val}
    )
    assert himena_ui.current_model.value[0, 0] == "p-value"
    assert float(himena_ui.current_model.value[0, 1]) < 0.1

    himena_ui.exec_action(
        f"himena-stats:test:{command}",
        window_context=win_nonsig,
        with_params={"a": a_val, "b": b_val}
    )
    assert himena_ui.current_model.value[0, 0] == "p-value"
    assert float(himena_ui.current_model.value[0, 1]) > 0.3

    himena_ui.exec_action(
        f"himena-stats:test:{command}",
        window_context=win_sig_unpivot,
        with_params={"a": b_val, "groups": a_val}
    )
    assert himena_ui.current_model.value[0, 0] == "p-value"
    assert float(himena_ui.current_model.value[0, 1]) < 0.1

    himena_ui.exec_action(
        f"himena-stats:test:{command}",
        window_context=win_nonsig_unpivot,
        with_params={"a": b_val, "groups": a_val}
    )
    assert himena_ui.current_model.value[0, 0] == "p-value"
    assert float(himena_ui.current_model.value[0, 1]) > 0.3

@pytest.mark.parametrize(
    "command",
    ["paired-t-test", "wilcoxon-test"],
)
def test_a_vs_b_same_shape(himena_ui: MainWindow, command: str):
    win_sig = himena_ui.add_object(table_data_sig[:-1], type=StandardType.TABLE)
    win_sig_unpivot = himena_ui.add_object(
        unpivot(table_data_sig[:-1]), type=StandardType.TABLE
    )
    win_nonsig = himena_ui.add_object(table_data_nonsig[:-1], type=StandardType.TABLE)
    win_nonsig_unpivot = himena_ui.add_object(
        unpivot(table_data_nonsig[:-1]), type=StandardType.TABLE
    )
    himena_ui.exec_action(
        f"himena-stats:test:{command}",
        window_context=win_sig,
        with_params={"a": a_val, "b": b_val}
    )
    assert himena_ui.current_model.value[0, 0] == "p-value"
    assert float(himena_ui.current_model.value[0, 1]) < 0.1

    himena_ui.exec_action(
        f"himena-stats:test:{command}",
        window_context=win_nonsig,
        with_params={"a": a_val, "b": b_val}
    )
    assert himena_ui.current_model.value[0, 0] == "p-value"
    assert float(himena_ui.current_model.value[0, 1]) > 0.3

    himena_ui.exec_action(
        f"himena-stats:test:{command}",
        window_context=win_sig_unpivot,
        with_params={"a": b_val, "groups": a_val}
    )
    assert himena_ui.current_model.value[0, 0] == "p-value"
    assert float(himena_ui.current_model.value[0, 1]) < 0.1

    himena_ui.exec_action(
        f"himena-stats:test:{command}",
        window_context=win_nonsig_unpivot,
        with_params={"a": b_val, "groups": a_val}
    )
    assert himena_ui.current_model.value[0, 0] == "p-value"
    assert float(himena_ui.current_model.value[0, 1]) > 0.3
