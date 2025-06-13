import os
import uuid
import asyncio
import copy
from typing import Any, Callable, Optional
from os import PathLike
import time
import json
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from . import config as cfg

from futurehouse_client import FutureHouseClient
from futurehouse_client.models import TaskRequest, RuntimeConfig
from futurehouse_client.models.app import AuthType, Stage
import anthropic
import logging
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Step(BaseModel):
    """A step in the agent execution pipeline."""

    name: str = Field(
        description="Name of the job to run (e.g. 'job-futurehouse-data-analysis-crow-high')"
    )
    llm_call: bool = Field(
        default=False, description="Whether to call the LLM for the step"
    )
    include_search_tool: bool = Field(
        default=False, description="Whether to include the search tool in the LLM call"
    )
    model_name: str = Field(
        default=cfg.DEFAULT_MODEL, description="Name of the model to use for the step"
    )
    prompt_template: str = Field(description="Prompt template to use for the step")
    prompt_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments to format the prompt template.",
    )
    input_files: dict[str, str] = Field(
        default_factory=dict, description="Files to upload {'source_path': 'dest_name'}"
    )
    output_files: dict[str, str] = Field(
        default_factory=dict,
        description="Files to download {'source_name': 'dest_path'}",
    )
    step_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Small UID for the step",
    )
    n_replicate_tasks: int = Field(
        default=1, description="Number of parallel tasks to run"
    )
    runtime_config: RuntimeConfig = Field(
        default_factory=RuntimeConfig, description="Configuration for the step"
    )
    post_process: Optional[Callable[[dict[str, Any], str], None]] = Field(
        default=None, description="Function to run after step completion"
    )
    prompt_generator: Optional[Callable[[], list[tuple[str, dict[str, Any]]]]] = Field(
        default=None,
        description="Function to generate prompts and args for parallel tasks based on previous results",
    )
    timeout: int = Field(default=15 * 60, description="Timeout for the step in seconds")

    def format_prompt(self) -> str:
        """Format the prompt template with the provided arguments."""
        final_prompt = self.prompt_template.format(**self.prompt_args)
        return final_prompt


class Tortoise:
    """Runner for multi-step agent pipelines."""

    def __init__(self, api_key: str, environment: str = "PROD"):
        """Initialize the tortoise framework with FutureHouse API key."""
        self.client = FutureHouseClient(
            auth_type=AuthType.API_KEY,
            api_key=api_key,
            verbose_logging=True,
            stage=getattr(Stage, environment.upper(), Stage.PROD),
        )
        self.steps: list[Step] = []
        self.results: dict[str, Any] = {}

    def add_step(self, step: Step) -> None:
        """Add a step to the pipeline."""
        self.steps.append(step)

    def save_results(self, output_dir: str | PathLike = "output") -> None:
        """Save the results to a JSON file."""
        results_path = f"{output_dir}/results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        logger.info(f"Saving all results to {results_path}")
        try:
            os.makedirs(output_dir, exist_ok=True)
            serializable_results = {}
            for step_id, step_result in self.results.items():
                serializable_results[step_id] = dict(step_result)

            with open(results_path, "w") as f:
                json.dump(serializable_results, f, indent=2, default=str)
            logger.info(f"Results successfully saved to {results_path}")
        except Exception as e:
            logger.error(f"Error saving results to {results_path}: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _upload_file_with_retry(
        self, job_name: str, file_path: str, upload_id: str
    ) -> None:
        """Upload a file with retry logic."""
        self.client.upload_file(job_name, file_path=file_path, upload_id=upload_id)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _download_file_with_retry(
        self, job_name: str, trajectory_id: str, file_path: str, destination_path: str
    ) -> None:
        """Download a file with retry logic."""
        self.client.download_file(
            job_name,
            trajectory_id=trajectory_id,
            file_path=file_path,
            destination_path=destination_path,
        )

    def _create_task_requests(
        self, step: Step, runtime_config: RuntimeConfig
    ) -> list[TaskRequest]:
        """Create task requests with either identical or dynamic prompts.

        Args:
            step: The step configuration
            runtime_config: The runtime configuration for the task

        Returns:
            List of task requests to be executed
        """
        task_requests = []
        task_count = max(step.n_replicate_tasks, 1)

        if step.model_name:
            agent_config = cfg.get_custom_agent_config(step.model_name)
            runtime_config.agent = agent_config

        if step.runtime_config.continued_job_id:
            task_ids = self.results[str(step.runtime_config.continued_job_id)][
                "task_ids"
            ]
            if len(task_ids) > 1:
                logger.warning(
                    f"Continued job {step.runtime_config.continued_job_id} has multiple task ids, using the first one"
                )
            runtime_config.continued_job_id = str(task_ids[0])

        if step.prompt_generator and task_count > 1:
            # Generate dynamic prompts based on previous results
            prompt_pairs = step.prompt_generator()
            # Create a task request for each generated prompt
            for prompt_text, prompt_args in prompt_pairs[
                :task_count
            ]:  # Limit to requested parallel count
                step_copy = copy.deepcopy(step)
                step_copy.prompt_template = prompt_text
                step_copy.prompt_args = prompt_args
                query = step_copy.format_prompt()
                task_requests.append(
                    TaskRequest(
                        name=step.name,
                        query=query,
                        runtime_config=runtime_config,
                    )
                )
        else:
            # Default behavior: use the same prompt for all tasks
            query = step.format_prompt()
            task_requests = [
                TaskRequest(
                    name=step.name,
                    query=query,
                    runtime_config=runtime_config,
                )
            ] * task_count

        return task_requests

    async def call_llm(self, step: Step) -> list:
        """Call the LLM for the step."""
        anthropic_client = anthropic.Anthropic()
        # TODO: This is a hack to get the model name without the provider prefix
        model_name = step.model_name.replace("anthropic/", "")
        if step.include_search_tool:
            tools = [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                }
            ]
        else:
            tools = []
        response = anthropic_client.messages.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": step.prompt_template,
                }
            ],
            tools=tools,
            max_tokens=8192,
        )
        result = "\n".join([r.text for r in response.content if hasattr(r, "text")])
        return [result]

    async def _run_tasks_with_retry(
        self, task_requests, progress_bar, verbose, timeout
    ):
        """Run tasks with retry logic."""
        return await self.client.arun_tasks_until_done(
            task_requests,
            progress_bar=progress_bar,
            verbose=verbose,
            timeout=timeout,
            concurrency=1,  # Reduce concurrency to avoid overwhelming the server
        )

    async def run_pipeline(
        self, output_dir: str | PathLike = "output"
    ) -> dict[str, Any]:
        """Run the entire pipeline of steps."""
        os.makedirs(output_dir, exist_ok=True)

        for i, step in enumerate(self.steps):
            logger.info(f"Running step {i + 1}/{len(self.steps)}: {step.name}")
            if not step.runtime_config.upload_id:
                step.runtime_config.upload_id = step.step_id

            for source_path, dest_name in step.input_files.items():
                logger.info(f"Uploading file {source_path} as {dest_name}")
                try:
                    self._upload_file_with_retry(
                        step.name,
                        file_path=source_path,
                        upload_id=step.runtime_config.upload_id,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to upload file {source_path} after multiple retries: {e}"
                    )
                    raise

            if step.llm_call:
                task_responses = await self.call_llm(step)
                task_ids = [f"llm_{str(uuid.uuid4())[:8]}"]
                success_rate = 1
            else:
                task_requests = self._create_task_requests(step, step.runtime_config)

                logger.info(
                    f"Running {len(task_requests)} task{'s' if len(task_requests) > 1 else ''}"
                )
                try:
                    task_responses = await self._run_tasks_with_retry(
                        task_requests,
                        progress_bar=True,
                        verbose=False,
                        timeout=step.timeout,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to run tasks for step {step.step_id} after multiple retries: {e}"
                    )
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    # Create an error result entry and continue to the next step
                    self.results[step.step_id] = {
                        "task_ids": [],
                        "task_responses": [],
                        "success_rate": 0,
                        "error": str(e),
                    }
                    continue

                task_ids = [str(task.task_id) for task in task_responses]
                success_rate = sum(
                    [task.status == "success" for task in task_responses]
                ) / len(task_responses)
                logger.info(f"Task success rate: {success_rate * 100}%")

            self.results[step.step_id] = {
                "task_ids": task_ids,
                "task_responses": task_responses,
                "success_rate": success_rate,
            }

            os.makedirs(f"{output_dir}/{step.step_id}", exist_ok=True)

            for idx, task_id in enumerate(task_ids):
                for source_name, dest_path in step.output_files.items():
                    try:
                        # Add index suffix only when there are multiple tasks
                        path_suffix = f"_{idx}" if len(task_ids) > 1 else ""
                        if "." in dest_path:
                            base, ext = os.path.splitext(dest_path)
                            dest_path_with_idx = f"{base}{path_suffix}{ext}"
                        else:
                            dest_path_with_idx = f"{dest_path}{path_suffix}"

                        path = f"{output_dir}/{step.step_id}/{dest_path_with_idx}"
                        os.makedirs(
                            os.path.dirname(os.path.abspath(path)), exist_ok=True
                        )
                        logger.info(f"Downloading file {source_name} to {path}")
                        try:
                            self._download_file_with_retry(
                                step.name,
                                trajectory_id=task_id,
                                file_path=source_name,
                                destination_path=path,
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to download {source_name} from task {task_id} after multiple retries: {e}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error downloading {source_name} from task {task_id}: {e}"
                        )

            if step.post_process:
                logger.info(f"Running post-processing for step {step.step_id}")
                step.post_process(
                    self.results[step.step_id], f"{output_dir}/{step.step_id}"
                )

            logger.info(f"Completed step {i + 1}/{len(self.steps)}")

        self.save_results(output_dir)
        return self.results

    def run(self, output_dir: str | PathLike = "output") -> dict[str, Any]:
        """Synchronous version of run_pipeline."""
        return asyncio.run(self.run_pipeline(output_dir))
