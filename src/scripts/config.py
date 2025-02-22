"""Module for handling yaml config/CLI args and translating them into pydantic configs."""

import contextlib
import inspect
import logging
import os
import shutil
import sys
import textwrap
from argparse import ArgumentParser
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_core import PydanticUndefined

from .logging import configure_logs

logger = logging.getLogger(__name__)


class ConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, populate_by_name=True
    )


TConfig = TypeVar("TConfig", bound=BaseModel)


def load_arg_dict(argv: list[str]) -> dict[str, Any]:
    """Loads arguments from command line and yaml files into a dictionary.

    For example, if the command line args are `--foo.bar 1 --foo.baz 2`, the resulting
    dictionary is {'foo': {'bar': 1, 'baz': 2}}. YAML files are directly parsed as dictionaries.
    """
    parser = ArgumentParser(add_help=False)
    parser.add_argument("config_files", nargs="*", type=str)

    if not any(a.endswith(".yaml") for a in argv):
        # add a dummy arg to avoid argparse error
        argv = ["INVALID.yaml", *argv]
    args, remaining_args = parser.parse_known_args(argv)

    config_acc: dict[str, Any] = {}
    for cfg in args.config_files:
        if cfg == "INVALID.yaml":
            continue
        with open(cfg) as fcfg:
            config = yaml.load(fcfg, Loader=yaml.Loader)  # noqa: S506
        _recursive_update(config_acc, config)

    _parse_cli_args(remaining_args, config_acc)

    return config_acc


def load_config(
    config_cls: type[TConfig],
    verbose: bool = True,
    argv: list[str] | None = None,
    args_to_exclude: Iterable[str] | None = None,
) -> TConfig:
    """Utility function for handling config and command line args supplied via command line.

    Args:
        config_cls: Config class object
        verbose: Boolean indicating extent of logging info
        argv: List of command line args. If not specified (default), will use sys.argv.
        args_to_exclude: Arguments to skip when constructing the config object.

    Returns:
        Config object synthesizing CLI args and supplied yaml.
    """
    if argv is None:
        argv = sys.argv[1:]

    if "-h" in argv or "--help" in argv:
        print(get_config_help_string(config_cls))
        sys.exit(0)

    config_acc = load_arg_dict(argv)
    if args_to_exclude:
        for arg in args_to_exclude:
            config_acc.pop(arg, None)

    config = config_cls(**config_acc)

    if verbose:
        logger.info("\n%s", yaml.dump({config_cls.__name__: config.model_dump()}))

    return config


def _parse_cli_args(remaining_args: list[str], config_acc: dict):
    while remaining_args:
        arg = remaining_args.pop(0)
        if not arg.startswith("--"):
            raise ValueError(f"Invalid argument {arg}")

        arg = arg[2:]
        try:
            value = remaining_args[0]
            if value.startswith("--"):
                # moved on to next arg
                value = "True"
            else:
                # consumed value - remove from args
                remaining_args.pop(0)
        except IndexError:
            # end of args, assume it was a flag
            value = "True"
        value = _resolve_value(value)

        arg_hierarchy = arg.split(".")
        update_dict: dict[str, Any] = {}
        current_dict = update_dict
        for arg in arg_hierarchy[:-1]:
            current_dict[arg] = {}
            current_dict = current_dict[arg]
        current_dict[arg_hierarchy[-1]] = value
        _recursive_update(config_acc, update_dict)


def dump_config(config: BaseModel, path: os.PathLike | str) -> None:
    """Dump the input Pydantic config to a YAML file."""
    path = Path(path)
    if path.is_dir():
        path /= "config.yaml"
    with path.open("w") as f:
        yaml.dump(config.model_dump(), f)


def get_config_help_string(config_cls: type[BaseModel], indent: int = 0) -> str:
    s = (
        textwrap.indent(f"{config_cls.__name__}:", "  " * indent) + "\n"
        if indent == 0
        else ""
    )

    indent += 1
    for key, value in config_cls.model_fields.items():
        annot: Any = value.annotation
        # Removing the description printing for now, since it's just too verbose.
        # TODO: see if we can format it in a more readable way.
        # desc = f"  # {value.description}" if value.description else ""
        desc = ""

        if inspect.isclass(annot):
            if issubclass(annot, BaseModel):
                s += textwrap.indent(f"{key}:{desc}", "  " * indent) + "\n"
                s += get_config_help_string(annot, indent)
                continue

            annot = annot.__name__

        if value.is_required():
            s += textwrap.indent(f"{key}: {annot}{desc}", "  " * indent) + "\n"
        else:
            default = (
                value.default_factory
                if value.default is PydanticUndefined
                else value.default
            )
            s += (
                textwrap.indent(f"{key}: {annot} = {default!r}{desc}", "  " * indent)
                + "\n"
            )

    return s


DEFAULT_OUTPUT_LOG_NAME = "output.log"


def set_up_output_dir(
    directory_path: str | os.PathLike,
    config: BaseModel | None = None,
    log_name: str | None = DEFAULT_OUTPUT_LOG_NAME,
    is_main_process: bool = True,
    remove_existing: bool = False,
) -> Path:
    if remove_existing and is_main_process:
        shutil.rmtree(directory_path, ignore_errors=True)
    directory_path = Path(directory_path)
    directory_path.mkdir(parents=True, exist_ok=True)

    if log_name:
        configure_logs(log_file=directory_path / log_name)

    if config is not None and is_main_process:
        dump_config(config, directory_path)

    return directory_path


def _resolve_value(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    with contextlib.suppress(ValueError):
        return int(value)
    with contextlib.suppress(ValueError):
        return float(value)

    if value == "None":
        return None

    return value


def configure_yaml_multiline() -> None:
    # copied from SWE-agent
    def multiline_representer(dumper, data):
        """Configures yaml for dumping multiline strings.

        Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data.
        """
        if data.count("\n") > 0:  # check for multiline string
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, multiline_representer)


def _recursive_update(d: dict, u: dict) -> dict:
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


CONFIGURATION_ENABLE = {"1", "true", "yes", "on"}
CONFIGURATION_DISABLE = {"0", "false", "no", "off"}
