import os
import uuid
import asyncio
import copy
from typing import Any, Callable, Optional
from os import PathLike
import time
import json
from pydantic import BaseModel, Field
from . import prompts

from futurehouse_client import FutureHouseClient
from futurehouse_client.models import TaskRequest, RuntimeConfig
from futurehouse_client.models.app import AuthType


class StepConfig(BaseModel):
    """Agent runtime configuration."""

    language: str = Field(
        default="PYTHON", description="Language for execution environment"
    )
    max_steps: int = Field(
        default=30, description="Maximum number of steps for the agent"
    )
    timeout: int = Field(default=15 * 60, description="Timeout for the step in seconds")
    eval: bool = Field(
        default=True,
        description="For Finch, this indicates whether this is an API call or UI call. Setting it to True removes the automatic CoT additions.",
    )


class Step(BaseModel):
    """A step in the agent execution pipeline."""

    name: str = Field(
        description="Name of the job to run (e.g. 'job-futurehouse-data-analysis-crow-high')"
    )
    prompt_template: str = Field(description="Prompt template to use for the step")
    cot_prompt: bool = Field(
        default=False, description="Whether to augment the query with COT prompting"
    )
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
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Small UID for the step",
    )
    upload_id: Optional[str] = Field(default=None, description="Upload ID for GCS")
    parallel: int = Field(default=1, description="Number of parallel tasks to run")
    config: StepConfig = Field(
        default_factory=StepConfig, description="Configuration for the step"
    )
    post_process: Optional[Callable[[dict[str, Any], str], None]] = Field(
        default=None, description="Function to run after step completion"
    )
    prompt_generator: Optional[Callable[[], list[tuple[str, dict[str, Any]]]]] = Field(
        default=None,
        description="Function to generate prompts and args for parallel tasks based on previous results",
    )

    def cot_prompting(self, query: str, language: str) -> str:
        """Apply chain-of-thought prompting to the query."""
        guidelines = prompts.GENERAL_NOTEBOOK_GUIDELINES.format(language=language)
        if language == "R":
            guidelines = prompts.R_SPECIFIC_GUIDELINES.format(language=language)
        return (
            f"{prompts.CHAIN_OF_THOUGHT_AGNOSTIC.format(language=language)}\n"
            f"{guidelines}"
            f"Here is the research question to address:\n"
            f"<query>\n"
            f"{query}\n"
            f"</query>\n"
        )

    def format_prompt(self) -> str:
        """Format the prompt template with the provided arguments."""
        final_prompt = self.prompt_template.format(**self.prompt_args)
        if self.cot_prompt:
            final_prompt = self.cot_prompting(final_prompt, self.config.language)
        return final_prompt


class Tortoise:
    """Runner for multi-step agent pipelines."""

    def __init__(self, api_key: str):
        """Initialize the tortoise framework with FutureHouse API key."""
        self.client = FutureHouseClient(auth_type=AuthType.API_KEY, api_key=api_key)
        self.steps: list[Step] = []
        self.results: dict[str, Any] = {}

    def add_step(self, step: Step) -> None:
        """Add a step to the pipeline."""
        self.steps.append(step)

    def save_results(self, output_dir: str | PathLike = "output") -> None:
        """Save the results to a JSON file."""
        results_path = f"{output_dir}/results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        print(f"Saving all results to {results_path}")
        try:
            os.makedirs(output_dir, exist_ok=True)
            serializable_results = {}
            for step_id, step_result in self.results.items():
                serializable_results[step_id] = dict(step_result)

            with open(results_path, "w") as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Results successfully saved to {results_path}")
        except Exception as e:
            print(f"Error saving results to {results_path}: {e}")

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
        task_count = max(step.parallel, 1)

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

    async def run_pipeline(
        self, output_dir: str | PathLike = "output"
    ) -> dict[str, Any]:
        """Run the entire pipeline of steps."""
        os.makedirs(output_dir, exist_ok=True)

        for i, step in enumerate(self.steps):
            print(f"Running step {i + 1}/{len(self.steps)}: {step.name}")
            if not step.upload_id:
                step.upload_id = f"{step.name}_{step.step_id}"

            for source_path, dest_name in step.input_files.items():
                print(f"Uploading file {source_path} as {dest_name}")
                self.client.upload_file(
                    step.name, file_path=source_path, upload_id=step.upload_id
                )

            if step.config:
                runtime_config = RuntimeConfig(
                    max_steps=step.config.max_steps,
                    upload_id=step.upload_id,
                    environment_config={
                        "eval": step.config.eval,
                        "language": step.config.language,
                    },
                )
            else:
                runtime_config = None

            task_requests = self._create_task_requests(step, runtime_config)

            print(
                f"Running {len(task_requests)} task{'s' if len(task_requests) > 1 else ''}"
            )
            task_responses = await self.client.arun_tasks_until_done(
                task_requests,
                progress_bar=True,
                verbose=False,
                timeout=step.config.timeout,
            )

            task_ids = [str(task.task_id) for task in task_responses]
            success_rate = sum(
                [task.status == "success" for task in task_responses]
            ) / len(task_responses)
            print(f"Task success rate: {success_rate * 100}%")

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
                        print(f"Downloading file {source_name} to {path}")
                        self.client.download_file(
                            step.name,
                            trajectory_id=task_id,
                            file_path=source_name,
                            destination_path=path,
                        )
                    except Exception as e:
                        print(
                            f"Error downloading {source_name} from task {task_id}: {e}"
                        )

            if step.post_process:
                print(f"Running post-processing for step {step.step_id}")
                step.post_process(
                    self.results[step.step_id], f"{output_dir}/{step.step_id}"
                )

            print(f"Completed step {i + 1}/{len(self.steps)}")

        self.save_results(output_dir)
        return self.results

    def run(self, output_dir: str | PathLike = "output") -> dict[str, Any]:
        """Synchronous version of run_pipeline."""
        return asyncio.run(self.run_pipeline(output_dir))
