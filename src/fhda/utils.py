import asyncio
import base64
import logging
from collections.abc import Iterable
from enum import StrEnum, auto
from typing import TYPE_CHECKING, assert_never
import os

import nbformat
from traitlets.config import Config
from nbconvert import HTMLExporter
from aiodocker.containers import DockerContainer
from aviary.utils import MultipleChoiceQuestion

from . import config as cfg

if TYPE_CHECKING:
    from jupyter_client.asynchronous.client import AsyncKernelClient

logger = logging.getLogger(__name__)

JUPYTER_IMAGE_OUTPUT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
}

JUPYTER_TABLE_OUTPUT_TYPES_TO_IGNORE = {
    "text/latex",
    "text/html",
    "text/markdown",
}


class NBLanguage(StrEnum):
    PYTHON = auto()
    R = auto()

    def make_kernelspec(self) -> dict[str, str]:
        match self:
            # These are the default kernelspecs set by Jupyter and IRkernel respectively.
            case NBLanguage.PYTHON:
                kspec = {"name": "python", "display_name": "Python 3 (ipykernel)"}
            case NBLanguage.R:
                kspec = {"name": "ir", "display_name": "R"}
            case _:
                assert_never(self)

        return kspec | {"language": self.value}


def limit_notebook_output(output: str | list[str]) -> str:
    """Limit notebook output to configured length.

    Args:
        output: String output from notebook cell

    Returns:
        String output, truncated if longer than configured limit with
        indication of truncation
    """
    if isinstance(output, list):
        raise TypeError("Only string output truncation is supported")
    output_length = len(output)
    if output_length < cfg.NB_OUTPUT_LIMIT:
        return output
    cutoff = int(cfg.NB_OUTPUT_LIMIT / 2)
    # Sometimes error tracebacks have important information at the end
    # and at the beginning so important to keep those sections
    return output[:cutoff] + "\n<...output limited...>\n" + output[-cutoff:]


def process_cell_output(
    output, md: list[str], images: list[str], cell_streams: list[str]
) -> None:
    """Process a single output from a notebook cell."""
    if output.output_type == "stream":
        cell_streams.append(output.text)
    elif output.output_type == "execute_result":
        data = output.get("data", {}).get("text/plain", "")
        md.append(limit_notebook_output(data))
    elif output.output_type == "error":
        traceback_str = (
            "\n".join(output.traceback)
            if isinstance(output.traceback, list)
            else output.traceback
        )
        md.append(limit_notebook_output(traceback_str))
    elif output.output_type in {"display_data"}.union(JUPYTER_IMAGE_OUTPUT_TYPES):
        data_type = next(iter(output.data.keys()), "")
        if data_type in JUPYTER_TABLE_OUTPUT_TYPES_TO_IGNORE:
            return
        if data_type == "text/plain":
            md.append(limit_notebook_output(output.data[data_type]))
        elif data_type in JUPYTER_IMAGE_OUTPUT_TYPES:
            md.append(f"<{len(images) + 1}>")
            image_format = data_type.split("/")[-1]
            image_prefix = f"data:image/{image_format};base64,"
            images.append(image_prefix + encode_image_to_base64(output.data[data_type]))
        else:
            logger.warning(f"Unknown data type: {data_type}")
            md.append(limit_notebook_output(output.data[data_type]))


def view_notebook(
    cells: list[nbformat.NotebookNode], language: str
) -> tuple[str, list[str]]:
    """Process notebook cells and convert them to markdown format with images.

    Args:
        cells: List of notebook cells to process
        language: Programming language of the notebook code cells

    Returns:
        tuple containing:
            - Markdown string with cell contents and outputs
            - List of base64 encoded images found in cell outputs
    """
    md: list[str] = []
    images: list[str] = []

    for idx, cell in enumerate(cells):
        md.append(f"### Cell {idx}:")
        if cell.cell_type == "code":
            md.extend((f"```{language}", cell.source, "```"))

            outputs = cell.get("outputs", [])
            if outputs:
                md.extend([f"### Output {idx}:", "```"])
                cell_streams: list[str] = []

                for output in outputs:
                    process_cell_output(output, md, images, cell_streams)

                if cell_streams:
                    combined_stream = "\n".join(cell_streams)
                    md.append(limit_notebook_output(combined_stream))
                md.append("```")
        elif cell.cell_type in {"markdown", "raw"}:
            md.append(cell.source)

    return "\n".join(md), images


def encode_image_to_base64(image: str) -> str:
    decoded_image = base64.b64decode(image)
    return base64.b64encode(decoded_image).decode("utf-8")


async def nbformat_run_notebook(
    cells: Iterable[nbformat.NotebookNode], client: "AsyncKernelClient"
) -> list[str]:
    """Execute notebook cells using a kernel client and collect outputs.

    Args:
        cells: Notebook cell dictionaries to execute sequentially
        client: KernelClient instance to use for code execution

    Raises:
        ValueError: If there is an error executing a cell

    Returns:
        List of error messages from cells that raised an error
    """
    error_messages = []
    try:
        logger.debug("Beginning cell execution")
        for idx, cell in enumerate(cells):
            if cell.cell_type == "code":
                logger.debug(f"Executing code cell {idx}")
                cell.outputs = []  # Initialize empty outputs list
                msg_id = client.execute(cell.source)
                logger.debug(f"Message ID for cell {idx}: {msg_id}")

                while True:
                    msg = await client.get_iopub_msg()
                    logger.debug(f"Received message type: {msg['msg_type']}")

                    if msg["parent_header"].get("msg_id") == msg_id:
                        msg_type = msg["msg_type"]
                        content = msg["content"]

                        if msg_type in {
                            "execute_result",
                            "display_data",
                            "stream",
                        }:
                            if msg_type == "stream":
                                output = nbformat.v4.new_output(
                                    output_type="stream",
                                    name=content["name"],
                                    text=content["text"],
                                )
                            elif msg_type == "execute_result":
                                output = nbformat.v4.new_output(
                                    output_type="execute_result",
                                    data=content.get("data", {}),
                                    metadata=content.get("metadata", {}),
                                    execution_count=content.get("execution_count"),
                                )
                            else:  # display_data
                                output = nbformat.v4.new_output(
                                    output_type="display_data",
                                    data=content.get("data", {}),
                                    metadata=content.get("metadata", {}),
                                )
                            cell.outputs.append(output)
                            logger.debug(
                                f"Added output of type {msg_type} to cell {idx}"
                            )

                        elif msg_type == "error":
                            # Create error output and add it to cell outputs
                            error_output = nbformat.v4.new_output(
                                output_type="error",
                                ename=content.get("ename", ""),
                                evalue=content.get("evalue", ""),
                                traceback=content.get("traceback", []),
                            )
                            cell.outputs.append(error_output)

                            error_msg = (
                                f"Error executing cell {idx}:\n"
                                f"Name: {content.get('ename', 'Unknown')}\n"
                                f"Value: {content.get('evalue', 'No error message')}\n"
                                f"Traceback: {content.get('traceback', [])}"
                            )
                            error_messages.append(
                                f"Cell {idx}: {content.get('evalue', '')}"
                            )
                            logger.error(error_msg)
                            # raise ValueError(error_msg)
                        elif (
                            msg_type == "status"
                            and content["execution_state"] == "idle"
                        ):
                            logger.debug(f"Cell {idx} execution finished")
                            break
    finally:
        logger.debug("Stopping kernel channels")
        client.stop_channels()

    return error_messages


async def exec_cmd(
    container: DockerContainer, exec_command: list[str], timeout: float | None = 300
) -> str:
    """Execute a command in a Docker container and capture output.

    Args:
        container: Docker container instance to execute command in
        exec_command: Command to execute as list of strings
        timeout: Maximum time in seconds to wait for command completion

    Returns:
        tuple containing:
            - Exit code from command execution
            - stdout output as string
            - stderr output as string

    Raises:
        TimeoutError: If command execution exceeds timeout period
    """
    try:
        async with asyncio.timeout(timeout):
            exec_instance = await container.exec(
                cmd=exec_command,
                tty=True,
                privileged=True,
            )

            # Start the execution
            stream = exec_instance.start()
            stdout = ""
            stderr = ""

            while True:
                try:
                    message = await stream.read_out()
                    if message is None:
                        break

                    # Messages come as tuples of (stream_type, data)
                    stream_type, data = message
                    if stream_type == cfg.DOCKER_STREAM_TYPE_STDOUT:  # stdout
                        stdout += data.decode()
                    elif stream_type == cfg.DOCKER_STREAM_TYPE_STDERR:  # stderr
                        stderr += data.decode()

                except EOFError:
                    break

            exit_code = (await exec_instance.inspect())["ExitCode"]
            logger.debug(f"Command output:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
            return exit_code
    except TimeoutError as err:
        raise TimeoutError(
            f"Command execution timed out after {timeout} seconds"
        ) from err


def collect_notebook_stats(nb: nbformat.NotebookNode) -> dict[str, int]:
    """Count lines, cells, outputs, and different language usage in a Jupyter notebook."""
    stats = {
        "code_lines": 0,
        "comment_lines": 0,
        "markdown_lines": 0,
        "code_cells": 0,
        "markdown_cells": 0,
        "images": 0,
        "tables": 0,
        "r_cells": 0,
        "bash_cells": 0,
        "shell_commands": 0,
    }
    for cell in nb.cells:
        # Split cell source into lines and count non-empty lines
        lines = [line for line in cell.source.split("\n") if line.strip()]

        if cell.cell_type == "code":
            stats["code_cells"] += 1

            # Process each line in code cells
            for line in lines:
                line = line.strip()
                # Check if line is a comment (starts with # but not #!)
                if line.startswith("#") and not line.startswith("#!"):
                    stats["comment_lines"] += 1
                else:
                    stats["code_lines"] += 1

            # Check for R and bash cells
            if lines:
                first_line = lines[0].strip()
                if first_line.startswith("%%R"):
                    stats["r_cells"] += 1
                elif first_line.startswith("%%bash"):
                    stats["bash_cells"] += 1

                # Count shell commands (lines starting with !)
                stats["shell_commands"] += sum(
                    1 for line in lines if line.strip().startswith("!")
                )

            # Check outputs for images and tables
            if hasattr(cell, "outputs"):
                for output in cell.outputs:
                    # Check for images
                    if output.get("output_type") in {"display_data", "execute_result"}:
                        if "image/png" in output.get("data", {}):
                            stats["images"] += 1

                        # Check for HTML tables or DataFrame representations
                        if "text/html" in output.get("data", {}):
                            html_content = output["data"]["text/html"]
                            if isinstance(html_content, list):
                                html_content = "".join(html_content)
                            if "<table" in html_content:
                                stats["tables"] += 1

                        # Check for plain text DataFrame representations
                        elif "text/plain" in output.get("data", {}):
                            text_content = output["data"]["text/plain"]
                            if isinstance(text_content, list):
                                text_content = "".join(text_content)
                            if any(
                                marker in text_content
                                for marker in ("DataFrame", "Series")
                            ):
                                stats["tables"] += 1

        elif cell.cell_type == "markdown":
            stats["markdown_lines"] += len(lines)
            stats["markdown_cells"] += 1

            # Count markdown images
            for line in lines:
                if "![" in line or "<img" in line:
                    stats["images"] += 1
    return stats


def load_mcq(
    mcq: dict, open_question: bool = False, question_id: str | None = None
) -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question=mcq["question"],
        options=[
            mcq["ideal_answer"],
            mcq["distractor_1"],
            mcq["distractor_2"],
            mcq["distractor_3"],
        ],
        ideal_answer=mcq["ideal_answer"],
        shuffle_seed=MultipleChoiceQuestion.SEED_USING_QUESTION,
        prompt_without_options=open_question,
        question_id=question_id or "Q",
    )


def nb_to_html(nb: nbformat.NotebookNode) -> str:
    # This configuration is necessary for the HTMLExporter to find the templates on GCP Cloud Jobs
    template_paths = [
        os.path.join(os.path.dirname(__file__), "templates"),
        os.path.join(os.path.dirname(__file__), "templates/base"),
        os.path.join(os.path.dirname(__file__), "templates/lab"),
    ]
    c = Config()
    c.TemplateExporter.template_paths = template_paths
    c.TemplateExporter.template_name = "lab/index.html.j2"

    exporter = HTMLExporter(config=c)
    html, _ = exporter.from_notebook_node(nb)
    return html
