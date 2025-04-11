import hashlib
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

from llmclient import GLOBAL_COST_TRACKER, enable_cost_tracking

from .notebook_env import NBEnvironment
from .utils import NBLanguage, MultipleChoiceQuestion, nb_to_html
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

    async def reset(self) -> tuple[Messages, list[Tool]]:
        # Discard base class's init_obs and make our own with the problem statement
        _, tools = await super().reset()
        messages = [
            Message(content=self.problem),
            self.get_env_state_msg(),
        ]
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        init_obs = cast(
            Messages,
            messages,
        )

        return init_obs, tools

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
                "nb_state_html": nb_to_html(self.state.nb),
                "nb_runtime_errors": self.state.notebook_runtime_errors,
            },
            info={
                "eval_mode": self.eval_mode,
                "language": self.state.language,
                "problem": self.problem,
                "problem_id": self.problem_id,
                "cost": GLOBAL_COST_TRACKER.lifetime_cost_usd,
            },
        )

    @classmethod
    def from_task(
        cls,
        task: str,
        gcs_artifact_path: str | None = None,
        environment_config: dict[str, Any] | None = None,
    ) -> "DataAnalysisEnv":
        """
        Perform data analysis on a user query.

        Args:
            task: The user query
            gcs_artifact_path: The path to the GCS artifact – required for evaluation on crow jobs
            environment_config: A JSON string of environment configuration
        """
        logger.info("User task: %s", task)
        logger.info("GCS artifact path: %s", gcs_artifact_path)
        logger.info("environment_config: %s", environment_config)
        # Track cost of running the environment
        enable_cost_tracking()
        if (
            not gcs_artifact_path
        ):  # Platform jobs should always be associated with data from a GCS bucket
            raise NotImplementedError(
                "Running crow jobs without gcs_artifact_path is not supported"
            )

        if environment_config:
            kwargs = {
                k: v
                for k, v in environment_config.items()
                if k in cfg.VALID_FROM_TASK_KWARGS
            }
        else:
            kwargs = {}
        logger.info("Filtered kwargs: %s", kwargs)
        task_hash = hashlib.sha256(task.encode()).hexdigest()
        if kwargs.get("eval", False):
            logger.info("Eval mode is True")
            # Create a temporary directory in GCP mounted storage volume
            trajectory_path = cfg.DATA_STORAGE_PATH / f"{task_hash}-{time.time()}"
            trajectory_path.mkdir(parents=True, exist_ok=True)
            for item in (cfg.DATA_STORAGE_PATH / gcs_artifact_path).iterdir():
                if item.is_file():
                    shutil.copy2(item, trajectory_path)
                elif item.is_dir():
                    shutil.copytree(
                        item, trajectory_path / item.name, dirs_exist_ok=True
                    )
        else:
            logger.info("Eval mode is False")
            # Use the GCP folder created when uploading the data via the platform
            trajectory_path = cfg.DATA_STORAGE_PATH / gcs_artifact_path
            # Augment incoming user query with CoT instructions
            task = (
                f"Here is the user query to address:\n"
                f"<query>\n"
                f"{task}\n"
                f"</query>\n"
                f"{prompts.CHAIN_OF_THOUGHT_AGNOSTIC}\n"
                f"{prompts.GENERAL_NOTEBOOK_GUIDELINES}"
            )
        logger.info("Trajectory path: %s", trajectory_path)
        nb_path = trajectory_path / NBEnvironment.NOTEBOOK_NAME
        logger.info("NB path: %s", nb_path)
        language = NBLanguage.PYTHON  # In future, this should be a hyperparameter
        if language == NBLanguage.R:
            task += f"\n{prompts.R_OUTPUT_RECOMMENDATION_PROMPT}"

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
            problem_id=f"data-analysis-task-{task_hash}",
            problem=task,
            eval_mode=EvalAnswerMode.LLM,
            nb_path=nb_path,
            work_dir=trajectory_path,
            language=language,
            system_prompt=prompts.CAPSULE_SYSTEM_PROMPT_QUERY,
            use_tmp_work_dir=False,
            **kwargs,
        )
