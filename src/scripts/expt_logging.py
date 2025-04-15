import logging
import os

from ldp.utils import configure_stdout_logs
from lmi import configure_llm_logs

logger = logging.getLogger(__name__)


def configure_logs(
    log_file: str | os.PathLike | None = None,
    stdout_level: int | str | tuple[str, int | str] | None = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """Configure logs.

    Args:
        log_file: Optional log file to add to all loggers.
        stdout_level: If int (default) or str, it's a log level for stdout. If two-tuple
            of str and int, it's a logger name and log level for that logger. Otherwise,
            if None, don't configure stdout logs.
        fmt: Logging format string.
    """
    configure_llm_logs()

    # Set some good default log levels to avoid too much verbosity
    logging.getLogger("dask").setLevel(logging.WARNING)
    logging.getLogger("vcr.cassette").setLevel(logging.WARNING)

    if stdout_level is not None:
        if isinstance(stdout_level, tuple):
            configure_stdout_logs(name=stdout_level[0], level=stdout_level[1], fmt=fmt)
        else:
            configure_stdout_logs(level=stdout_level, fmt=fmt)

    if log_file is not None:
        # Configure all loggers to write to a log file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt))
        logger.info(f"Logging to {log_file}.")

        # apply retroactively to root logger and all existing loggers
        for logger_name in ("root", *logging.root.manager.loggerDict.keys()):
            logging.getLogger(logger_name).addHandler(file_handler)
