import json
import shutil
from pathlib import Path
from tempfile import mkdtemp

from pydantic import Field

from aviary.core import EvalAnswerMode, TaskDataset
from .storage import DataRepo
from .data_analysis_env import DataAnalysisEnv
from .utils import NBLanguage, load_mcq
from . import prompts
from .models import ConfigModel
from .notebook_env import NBEnvironment
import logging


logger = logging.getLogger(__name__)


class CapsuleDatasetConfig(ConfigModel):
    repo: DataRepo = Field(
        default_factory=lambda: DataRepo(name="baseline-envs/data-analysis/v3.1"),
        description="The hosted repo to use for the dataset.",
    )

    local_repo_path: str | None = Field(
        default=None,
        description="If provided, will source the data from this local path instead of the hosted repo.",
    )

    local_output_path: str | None = Field(
        default=None,
        description="If provided, will save the output to this local path instead of the hosted repo.",
    )

    capsule_mode: str | None = Field(
        default="mcq",
        description="Determines whether the agent is to answer MCQs, open questions or whether a hypothesis is supported by the data",
    )

    eval_mode: EvalAnswerMode = Field(
        default=EvalAnswerMode.LLM,
        description="If exact, the target will be 'answer' in the metadata json (i.e. T/F) "
        "If llm, the target will be 'result'. Contains/score not supported",
    )

    avoid_images: bool = Field(
        default=False,
        description="If True, the agent will be prompted to avoid using images in its notebook.",
    )

    preload_notebook: bool = Field(
        default=False,
        description=(
            "If False, the agent will have to start from a virgin notebook. "
            "If True, the agent environment will be preloaded with a notebook "
            "containing a portion of the capsule problem already completed "
            "eg package & data loading."
        ),
    )

    prompt_template_key: str = Field(
        default="v1.3.1",
        description="The key of the prompt template from the CAPSULE_PROMPT_TEMPLATES dict to use for the problem.",
    )


class CapsuleDataset(TaskDataset[DataAnalysisEnv]):
    """A dataset of tasks derived from data analysis capsules."""

    def __init__(self, config: CapsuleDatasetConfig):
        # Load dataset from local path or hosted repo
        if config.local_repo_path:
            repo_path = config.local_repo_path
        else:
            config.repo.pull(progress=True)
            repo_path = config.repo.local_path
        self.capsules = list(Path(repo_path).rglob("CapsuleFolder*"))

        # Load prompt template
        self.prompt = prompts.CAPSULE_PROMPT_TEMPLATES[config.prompt_template_key]
        self.config = config

    def get_new_env_by_idx(self, idx: int) -> DataAnalysisEnv:
        capsule_path = self.capsules[idx]
        metadata = json.load((capsule_path / "metadata.json").open())

        notebook_name = NBEnvironment.NOTEBOOK_NAME
        # Define local capsule directory
        if self.config.local_output_path:
            problem_dir = Path(self.config.local_output_path) / capsule_path.name
        else:
            problem_dir = Path(mkdtemp())
        problem_dir.mkdir(parents=True, exist_ok=True)

        # Copy capsule contents to local directory
        for item in capsule_path.iterdir():
            if self.config.preload_notebook and str(item).endswith("_stripped.ipynb"):
                shutil.copy(item, problem_dir)
            elif str(item).endswith((".ipynb", "metadata.json", "checksum")):
                continue
            elif item.is_dir():
                shutil.copytree(item, problem_dir / item.name)
            else:
                shutil.copy(item, problem_dir)

        nb_path = problem_dir / notebook_name

        # Define system prompt and problem
        if self.config.capsule_mode == "hypothesis":
            system_prompt = prompts.CAPSULE_SYSTEM_PROMPT_HYPOTHESIS
            problem = self.prompt.replace("{{hypothesis}}", metadata["hypothesis"])
            answer = metadata["answer"]
            processed_questions = None
        elif self.config.capsule_mode == "mcq":
            raw_mcqs = metadata["notebook_questions"]["questions"]
            processed_questions = [
                load_mcq(i, open_question=False, question_id=i["id"]) for i in raw_mcqs
            ]
            system_prompt = prompts.CAPSULE_SYSTEM_PROMPT_MCQ
            problem = self.prompt.format(
                questions="\n-------\n".join(
                    [i.question_prompt for i in processed_questions]
                )
            )
            answer = {i.question_id: i.ideal_answer for i in processed_questions}
        elif self.config.capsule_mode == "open":
            system_prompt = prompts.CAPSULE_SYSTEM_PROMPT_OPEN
            raw_open_questions = metadata["notebook_questions"]["questions"]
            processed_questions = [
                load_mcq(i, open_question=True, question_id=i["id"])
                for i in raw_open_questions
            ]
            problem = self.prompt.format(
                questions="\n-------\n".join(
                    [i.question_prompt for i in processed_questions]
                )
            )
            answer = {i.question_id: i.ideal_answer for i in processed_questions}
        else:
            raise ValueError(f"Invalid capsule mode: {self.config.capsule_mode}")

        if self.config.avoid_images:
            problem += prompts.AVOID_IMAGES

        # Temporarily hard code language to python, but can also use R
        language = NBLanguage.PYTHON
        return DataAnalysisEnv(
            problem_id=capsule_path.name,
            problem=problem,
            eval_mode=self.config.eval_mode,
            nb_path=nb_path,
            work_dir=problem_dir,
            language=language,
            system_prompt=system_prompt,
            metadata=metadata,
            answer=answer,
            mcqs=processed_questions,
        )

    def __len__(self) -> int:
        return len(self.capsules)
