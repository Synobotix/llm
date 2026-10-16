import importlib.util
import types

import pytest

from scripts.nemotron import colab_setup as module


@pytest.fixture
def runner(monkeypatch):
    state = types.SimpleNamespace(commands=[], failing=set())

    def run(command, text):
        state.commands.append(command)
        return types.SimpleNamespace(
            returncode=1 if command[-1] in state.failing else 0
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    return state


def test_step_runs_the_module_with_the_current_interpreter(runner):
    module.run_step("1. Test", "package.module")

    assert runner.commands == [[module.sys.executable, "-m", "package.module"]]


def test_failed_step_raises_with_its_name(runner):
    runner.failing.add("package.module")

    with pytest.raises(RuntimeError, match="1. Test"):
        module.run_step("1. Test", "package.module")


def test_main_runs_all_steps_in_order(runner):
    module.main()

    assert [c[-1] for c in runner.commands] == [m for _, m in module.STEPS]


def test_main_stops_at_first_failing_step(runner):
    runner.failing.add("scripts.split_dataset")

    with pytest.raises(RuntimeError):
        module.main()

    assert [c[-1] for c in runner.commands] == [
        "scripts.nemotron.download_and_prepare",
        "scripts.split_dataset",
    ]


def test_pipeline_order_is_download_split_tokenizer_tokenize():
    names = [name.split(".")[0] for name, _ in module.STEPS]

    assert names == ["1", "2", "3", "4"]
    assert module.STEPS[0][1] == "scripts.nemotron.download_and_prepare"
    assert module.STEPS[1][1] == "scripts.split_dataset"
    assert module.STEPS[3][1] == "scripts.tokenize_dataset"


@pytest.mark.parametrize(
    "module_name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "colab_setup référence src.tokenizer.train_tokenizer, "
                    "qui n'existe pas : le module est tokenizer.train_tokenizer."
                ),
            )
            if name == "src.tokenizer.train_tokenizer"
            else (),
        )
        for _, name in module.STEPS
    ],
)
def test_every_pipeline_step_points_to_an_existing_module(module_name):
    parent = module_name.rsplit(".", 1)[0]

    assert importlib.util.find_spec(parent) is not None
    assert importlib.util.find_spec(module_name) is not None
