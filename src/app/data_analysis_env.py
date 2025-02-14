import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, cast
import time
from aviary.core import (
    EvalAnswerMode,
    Frame,
    Message,
    Messages,
    Tool,
    eval_answer,
)

from .notebook_env import NBEnvironment
from .utils import NBLanguage, MultipleChoiceQuestion
from . import prompts

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
        eval_mode: EvalAnswerMode,
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

    async def submit_answer(self, answer: str | float | dict[str, Any] | None) -> str:  # type: ignore[override]
        """Submit an answer to the problem.

        Note that this tool may only be called once and ends the episode.

        Args:
            answer: The answer to the problem
        """
        # TODO: support various eval modes
        self.state.answer = answer
        self.state.done = True
        correct = False
        logger.info("Answer: %s", answer)

        if isinstance(self.answer, int):
            try:
                answer = int(answer)  # type: ignore[arg-type]
            except ValueError:
                pass
            else:
                correct = answer == self.answer

        elif isinstance(self.answer, float):
            try:
                answer = float(answer)  # type: ignore[arg-type]
            except ValueError:
                pass
            else:
                correct = abs(answer - self.answer) < 1e-4 * self.answer

        elif isinstance(self.answer, str):
            correct = bool(
                await eval_answer(
                    proposed=str(answer),
                    correct=str(self.answer),
                    question=self.problem,
                    eval_mode=self.eval_mode,
                )
            )
        elif isinstance(self.answer, dict):  # This is for mcqs and open questions
            # Check if answer is a json string
            if isinstance(answer, str):  # type: ignore[unreachable]
                # Process json into dictionary
                try:
                    processed_answer = json.loads(answer)
                except json.JSONDecodeError:
                    return INCORRECT_MSG
            else:
                processed_answer = answer if isinstance(answer, dict) else {}

            # Loop through each question and answer
            for question_id, agent_answer in processed_answer.items():
                try:
                    ideal_answer = self.answer[question_id]
                    question = next(
                        q
                        for q in self.mcqs
                        if q.question_id.lower() == question_id.lower()
                    )
                    correct = bool(
                        await eval_answer(
                            proposed=str(agent_answer),
                            correct=str(ideal_answer),
                            question=question,
                            eval_mode=self.eval_mode,
                        )
                    )
                    self.question_rewards[question_id] = correct
                except KeyError:
                    self.question_rewards[question_id] = 0
                average_reward = sum(self.question_rewards.values()) / len(self.mcqs)
            correct = round(average_reward) == 1.0

        if correct:
            self.state.total_reward += self.correct_reward
            return CORRECT_MSG
        return INCORRECT_MSG

    @classmethod
    def from_task(cls, task: str) -> "DataAnalysisEnv":
        """
        Perform data analysis on a user query.

        Args:
            task: The user query structured as <data_path> | <query>

        eg "CaspuleFolder-a7812fg | How many genes are differentially expressed between the two conditions?"
        """

        language = NBLanguage.PYTHON  # In future, this should be a hyperparameter
        # Hash the task to get a unique identifier
        task_hash = hashlib.sha256(task.encode()).hexdigest()
        # Extract data path and query from task
        data_path, query = task.split("|")
        # Augment incoming task with CoT instructions
        augmented_task = f"""\
Here is the user query to address:

<query>
{query}
</query>

{prompts.CHAIN_OF_THOUGHT_AGNOSTIC}
{prompts.GENERAL_NOTEBOOK_GUIDELINES}"""

        if language == NBLanguage.R:
            augmented_task += f"\n{prompts.R_OUTPUT_RECOMMENDATION_PROMPT}"
        # Create temporary directory in GCP mounted storage volume
        trajectory_path = Path("storage") / f"{task_hash}-{time.time()}"
        trajectory_path.mkdir(parents=True, exist_ok=True)
        notebook_name = NBEnvironment.NOTEBOOK_NAME
        nb_path = trajectory_path / notebook_name
        # Copy task data to trajectory path
        for item in (Path("storage") / data_path).iterdir():
            if item.is_file():
                shutil.copy2(item, trajectory_path)
            elif item.is_dir():
                shutil.copytree(item, trajectory_path / item.name, dirs_exist_ok=True)

        return cls(
            problem_id=f"data-analysis-task-{task_hash}",
            problem=augmented_task,
            eval_mode=EvalAnswerMode.LLM,
            nb_path=nb_path,
            work_dir=trajectory_path,
            language=language,
            system_prompt=prompts.CAPSULE_SYSTEM_PROMPT_QUERY,
            use_tmp_work_dir=False,
        )

    def export_frame(self) -> Frame:
        return Frame(
            state={
                "last_action": self.state.actions[-1],
                "answer": self.state.answer,
                "done": self.state.done,
                "total_reward": self.state.total_reward,
                "nb_state": self.state.nb,
            },
            info={
                "eval_mode": self.eval_mode,
                "language": self.state.language,
                "problem": self.problem,
                "problem_id": self.problem_id,
            },
        )
