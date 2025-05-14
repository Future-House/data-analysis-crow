import logging
import shutil
from typing import Any, cast
import time
from aviary.core import (
    EvalAnswerMode,
    Frame,
    Message,
    Messages,
    Tool,
)

from lmi.cost_tracker import GLOBAL_COST_TRACKER, enable_cost_tracking
from futurehouse_client.models import TaskRequest, AuthType
from futurehouse_client import FutureHouseClient

from .notebook_env import NBEnvironment
from .utils import NBLanguage, MultipleChoiceQuestion
from . import prompts
from . import config as cfg

logger = logging.getLogger(__name__)

CORRECT_MSG = "Correct answer!"
INCORRECT_MSG = "Incorrect answer."


class DataAnalysisEnv(NBEnvironment):
    def __init__(
        self,
        *,
        problem_id: str,
        problem: str,
        answer: str | int | float | None = None,  # noqa: PYI041
        system_prompt: str | None = None,
        correct_reward: float = 1.0,
        eval_mode: EvalAnswerMode | None = None,
        metadata: dict[str, Any] | None = None,  # used for NBEvalExpt
        mcqs: list[MultipleChoiceQuestion] | None = None,
        # Exclude list_workdir and query_literature tools by default
        exclude_tools: list[str] | None = ["list_workdir", "query_literature"],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.problem_id = problem_id
        self.problem = problem
        self.mcqs = mcqs
        self.answer = answer
        self.eval_mode = eval_mode
        self.correct_reward = correct_reward
        self.system_prompt = system_prompt
        self.metadata = metadata
        self.question_rewards: dict[str, int] = {}
        self.exclude_tools = exclude_tools

    async def reset(self) -> tuple[Messages, list[Tool]]:
        # Discard base class's init_obs and make our own with the problem statement
        _, tools = await super().reset()

        tools.append(Tool.from_function(self.query_literature))

        if self.exclude_tools:
            tools = [
                tool
                for tool in tools
                if tool._tool_fn.__name__ not in self.exclude_tools
            ]

        messages = [
            Message(content=self.problem),
            self.get_env_state_msg(),
        ]
        # If the list_workdir tool is excluded, add the content of the working directory to the initial message
        if self.exclude_tools is not None and "list_workdir" in self.exclude_tools:
            messages.append(
                Message(
                    content=f"Here is the content of your working directory:\n{self.list_workdir()}"
                )
            )

        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        init_obs = cast(
            Messages,
            messages,
        )

        return init_obs, tools

    # DA Specific Tools

    async def query_literature(self, query: str) -> str:
        """Query the scientific literature. Produces a succinct answer citing the scientific literature.

        Args:
            query: The scientific question to answer
        """

        if cfg.PLATFORM_API_KEY is None:
            raise Exception("Platform API key is not set")

        logger.info("Running PQA query")
        client = FutureHouseClient(
            stage=cfg.CROW_STAGE,
            auth_type=AuthType.API_KEY,
            api_key=cfg.PLATFORM_API_KEY,
        )

        job_data = TaskRequest(
            name="job-futurehouse-paperqa2",
            query=query,
        )
        job_id = client.create_task(job_data)
        status = "in progress"
        while status in ["in progress", "queued"]:
            logger.info(
                "Waiting for pqa task to complete... checking again in 5 seconds"
            )
            time.sleep(5)
            status = client.get_task(job_id).status

        if status == "failed":
            raise Exception("PaperQA platform job failed")

        job_result = client.get_task(job_id, verbose=True)
        answer = job_result.environment_frame["state"]["state"]["response"]["answer"][
            "answer"
        ]
        return answer

    async def submit_answer(self, answer: str) -> str:  # type: ignore[override]
        """Submit an answer to the problem.

        Note that this tool may only be called once and ends the episode.

        Args:
            answer: The answer to the problem
        """
        # TODO: support various eval modes
        self.state.answer = answer
        self.state.done = True
        logger.info("Submitting answer and closing environment")
        await self.close()
        logger.info("Answer: %s", answer)
        return answer

    def export_frame(self) -> Frame:
        return Frame(
            state={
                "last_action": self.state.actions[-1] if self.state.actions else None,
                "answer": self.state.answer,
                "done": self.state.done,
                "total_reward": self.state.total_reward,
                "nb_state": self.state.nb,
                # "nb_state_html": nb_to_html(self.state.nb), # temporarily disabled
                "nb_runtime_errors": self.state.notebook_runtime_errors,
            },
            info={
                "eval_mode": self.eval_mode,
                "language": self.state.language,
                "problem": self.problem,
                "problem_id": self.problem_id,
                "cost": GLOBAL_COST_TRACKER.lifetime_cost_usd,
                "work_dir": self.work_dir,
            },
        )

    @classmethod
    def from_task(
        cls,
        task: str,
        gcs_artifact_path: str | None = None,
        trajectory_id: str | None = None,
        user_id: str | None = None,
        environment_config: dict[str, Any] | None = None,
        continued_trajectory_id: str | None = None,
    ) -> "DataAnalysisEnv":
        """
        Perform data analysis on a user query.

        Args:
            task: The user query
            gcs_artifact_path: The path to the GCS artifact – required for evaluation on crow jobs
            environment_config: A JSON string of environment configuration
        """
        logger.info("User task: %s", task[:50])
        logger.info("GCS artifact path: %s", gcs_artifact_path)
        logger.info("environment_config: %s", environment_config)
        logger.info("trajectory_id: %s", trajectory_id)
        logger.info("user_id: %s", user_id)
        logger.info("continued_trajectory_id: %s", continued_trajectory_id)
        enable_cost_tracking()

        if (
            (not gcs_artifact_path) and not continued_trajectory_id
        ):  # Platform jobs should always be associated with data from a GCS bucket
            raise NotImplementedError(
                "Running crow jobs without gcs_artifact_path is not supported"
            )

        if user_id is None:
            logger.warning("No user_id provided, using default_user")
            user_id = "default_user"
        if trajectory_id is None:
            logger.warning("No trajectory_id provided, using time-based id")
            trajectory_id = f"{gcs_artifact_path}-{time.time()}"
        if environment_config:
            kwargs = {
                k: v
                for k, v in environment_config.items()
                if k in cfg.VALID_FROM_TASK_KWARGS
            }
        else:
            kwargs = {}
            environment_config = {}
        # Always create a new directory for the trajectory
        trajectory_path = (
            cfg.DATA_STORAGE_PATH / "user_trajectories" / user_id / trajectory_id
        )
        if continued_trajectory_id:
            data_path = (
                cfg.DATA_STORAGE_PATH
                / "user_trajectories"
                / user_id
                / continued_trajectory_id
            )
            logger.info("Continuing trajectory from %s", continued_trajectory_id)
        elif environment_config.get("gcs_override", False):
            data_path = cfg.DATA_STORAGE_PATH / gcs_artifact_path  # type: ignore
        else:
            data_path = (
                cfg.DATA_STORAGE_PATH / "user_data" / user_id / gcs_artifact_path  # type: ignore
            )
        logger.info("Trajectory path: %s", trajectory_path)
        logger.info("Data path: %s", data_path)
        trajectory_path.mkdir(parents=True, exist_ok=True)
        for item in data_path.iterdir():
            if item.is_file():
                shutil.copy2(item, trajectory_path)
            elif item.is_dir():
                shutil.copytree(item, trajectory_path / item.name, dirs_exist_ok=True)
        logger.info("Filtered kwargs: %s", kwargs)

        language = getattr(NBLanguage, environment_config.get("language", "PYTHON"))
        # Overwrite the language in the kwargs with NBLanguage enum
        kwargs["language"] = language
        logger.info("Language: %s", language.name)

        if not environment_config.get("eval", False):
            logger.info(
                "Platform job detected, augmenting user query with CoT instructions"
            )
            # If running via the platform, augment incoming user query with CoT instructions
            task = (
                f"{prompts.CHAIN_OF_THOUGHT_AGNOSTIC.format(language=kwargs.get('language', 'PYTHON'))}\n"
                f"{prompts.GENERAL_NOTEBOOK_GUIDELINES.format(language=kwargs.get('language', 'PYTHON'))}"
                f"Here is the research question to address:\n"
                f"<query>\n"
                f"{task}\n"
                f"</query>\n"
            )
        nb_path = trajectory_path / NBEnvironment.NOTEBOOK_NAME
        logger.info("NB path: %s", nb_path)

        if trajectory_path.exists():
            files = list(trajectory_path.iterdir())
            logger.info("Files in directory: %s", [f.name for f in files])
            if not files:
                raise ValueError(
                    f"No files found in trajectory path: {trajectory_path}"
                )
        else:
            raise ValueError(f"Trajectory path does not exist: {trajectory_path}")

        return cls(
            problem_id=f"data-analysis-task-{trajectory_id}",
            problem=task,
            eval_mode=EvalAnswerMode.LLM,
            nb_path=nb_path,
            work_dir=trajectory_path,
            system_prompt=environment_config.get(
                "system_prompt", prompts.CAPSULE_SYSTEM_PROMPT_QUERY
            ),
            use_tmp_work_dir=False,
            **kwargs,
        )
