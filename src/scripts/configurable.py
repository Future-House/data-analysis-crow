import asyncio
import importlib
import sys
from abc import ABC, abstractmethod
from typing import Self

from .config import ConfigModel, load_arg_dict, load_config


class ConfigurableExpt(ConfigModel, ABC):
    """A base class for configurable experiments.

    Example usage:
    ```py
    expt = DummyExpt()
    await expt.run()  # prints "Hello, world!"

    expt = DummyExpt.from_cli_args(argv=["--who", "friend"])
    await expt.run()  # prints "Hello, friend!"
    ```
    """

    @classmethod
    def from_cli_args(cls, **kwargs) -> Self:
        return load_config(cls, **kwargs)

    @abstractmethod
    async def run(self) -> None:
        """The entry point for the executable."""


class DummyExpt(ConfigurableExpt):
    """For unit tests."""

    who: str = "world"

    # Returning string for unit tests
    async def run(self) -> str:  # type: ignore[override]
        print(f"Hello, {self.who}!")
        return self.who


def _run_expt() -> None:
    """
    Import and run a ConfigurableExpt.

    NOTE: this is not meant to be called from python code, instead it's exposed
    (in pyproject.toml) as `run_expt` command line entry point.
    """

    argv = sys.argv[1:]
    first_arg: str | None = argv[0] if argv else None

    if not first_arg or first_arg in {"-h", "--help"}:
        print("Usage: run_expt <expt_name> [app_args...]")
        return

    # check if expt_name was specified
    if first_arg.startswith("--") or first_arg.endswith(".yaml"):
        # expt_name was not specified in CLI args. Try to infer from remaining args
        parsed_args = load_arg_dict(argv=argv)
        try:
            expt_name = parsed_args["expt"]
        except KeyError:
            # NOTE: not using `raise ValueError` to avoid lengthy traceback
            print(
                "Error: experiment was not specified in CLI args nor in configuration.",
                file=sys.stderr,
            )
            sys.exit(1)

    else:
        expt_name = argv.pop(0)

    # Import the expt
    expt_module = importlib.import_module(name=".".join(expt_name.split(".")[:-1]))
    expt_class = getattr(expt_module, expt_name.split(".")[-1])
    if not issubclass(expt_class, ConfigurableExpt):
        # NOTE: not using `raise TypeError` to avoid lengthy traceback
        print(
            f"Error: {expt_name} is not a subclass of ConfigurableExpt.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Skip 'expt' if it's in the args, since that was just used to infer expt_name
    expt = expt_class.from_cli_args(argv=argv, args_to_exclude=["expt"])
    asyncio.run(expt.run())
