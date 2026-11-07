"""The CLI must not require native dependencies just to parse arguments."""

from unittest.mock import Mock

import pytest

from pulsing.cli import __main__ as cli


def test_actor_constructor_and_legacy_override(monkeypatch):
    start = Mock()
    monkeypatch.setattr(cli, "actor", start)
    cli.main(
        [
            "actor",
            "my_app.Counter",
            "--name",
            "counter",
            "-D",
            'actor.extra_kwargs={"enabled": true}',
            "--",
            "--initial",
            "10",
        ]
    )
    start.assert_called_once_with(
        actor_type="my_app.Counter",
        addr=None,
        seeds=None,
        name="counter",
        extra_kwargs={"initial": "10", "enabled": True},
    )


def test_inspect_typed_options_and_aliases(monkeypatch):
    observe = Mock()
    monkeypatch.setattr(cli, "inspect", observe)
    cli.main(
        [
            "inspect",
            "actors",
            "--endpoint",
            "127.0.0.1:8000",
            "--top",
            "3",
            "--all-actors",
            "--json",
            "--raw",
            "false",
            "--best_effort",
            "True",
        ]
    )
    options = observe.call_args.kwargs
    assert options["top"] == 3
    assert options["all_actors"] is True
    assert options["best_effort"] is True
    assert options["json"] is True
    assert options["raw"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["examples", "counting_game"],
        ["examples", "--name", "counting_game"],
    ],
)
def test_example_name_forms(monkeypatch, arguments):
    show = Mock()
    monkeypatch.setattr(cli, "examples", show)
    cli.main(arguments)
    show.assert_called_once_with(name="counting_game")


@pytest.mark.parametrize(
    "arguments",
    [
        ["forge"],
        ["bench"],
        ["gui"],
        ["actor", "my_app.Counter", "-D", "other.setting=1"],
        ["actor", "my_app.Counter", "--extra-kwargs", "[]"],
        ["inspect", "actors", "--raw", "not-a-boolean"],
    ],
)
def test_invalid_cli_input_fails(arguments):
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments)
    assert exc.value.code == 2
