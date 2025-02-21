import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Self, cast

import litellm
from aviary.core import EvalAnswerMode, TaskDatasetClient
from scripts.config import ConfigModel, set_up_output_dir
from scripts.configurable import ConfigurableExpt
from app.utils import NBLanguage
from app.storage import DataRepo
from ldp.agent import Agent, AgentConfig
from ldp.alg import Evaluator, EvaluatorConfig, TrajectoryFileCallback
from ldp.alg.callbacks import Callback
from ldp.alg.rollout import RolloutManager
from ldp.data_structures import Transition
from llmclient.cost_tracker import enable_cost_tracking
from pydantic import Field, model_validator

from app.data_analysis_env import DataAnalysisEnv

from .client import TaskDatasetSplit
from .common import (
    LoggingCallback,
    SaveWorkspaceCallback,
    VerboseCallback,
)

logger = logging.getLogger(__name__)


class EnvServerConfig(ConfigModel):
    split: TaskDatasetSplit
    host: str = "localhost"
    port: int
    request_timeout: float | None = 300.0


class NBEvalExpt(ConfigurableExpt):
    output_repo: DataRepo
    comment: str = ""
    overwrite: bool = False

    env: EnvServerConfig

    agent: AgentConfig
    evaluator: EvaluatorConfig = Field(
        default_factory=lambda: EvaluatorConfig(num_eval_iterations=25)
    )

    async def make_dataset(self) -> TaskDatasetClient:
        base_dataset = await TaskDatasetClient.create(
            server_url=f"http://{self.env.host}:{self.env.port}",
            request_timeout=self.env.request_timeout,
        )
        return self.env.split.get_random_split(base_dataset)

    @model_validator(mode="after")
    def post_init(self) -> Self:
        if self.overwrite:
            shutil.rmtree(self.output_repo.local_path, ignore_errors=True)
        self.output_repo.mkdir()
        return self

    async def run(self) -> None:
        set_up_output_dir(self.output_repo.local_path, config=self)
        dataset = await self.make_dataset()
        agent = self.agent.construct_agent()
        callbacks: list[Callback] = [
            TrajectoryFileCallback(self.output_repo.local_path),
            LoggingCallback(self.output_repo),
            SaveWorkspaceCallback(
                dataset_client=dataset,
                workspace_repo=DataRepo(name=f"{self.output_repo.name}-workspaces"),
            ),
        ]
        if self.evaluator.batch_size == 1:
            callbacks.append(VerboseCallback())
        litellm.drop_params = True
        enable_cost_tracking(enabled=True)
        evaluator = Evaluator(
            config=self.evaluator,
            agent=agent,
            dataset=dataset,
            callbacks=callbacks,
        )
        await evaluator.run()

        self.output_repo.push(progress=True)


class AdHocExptCallback(Callback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    async def after_transition(
        self,
        traj_id: str,
        agent: Agent,
        env: DataAnalysisEnv,  # type: ignore[override]
        transition: Transition,
    ) -> None:
        if transition.done or transition.truncated or transition.failed:
            target_dir = self.output_dir / env.problem_id
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(env.state.work_dir, target_dir)

            if transition.action:
                action = transition.action.value
                submitted_answers = [
                    tc.function.arguments["answer"]
                    for tc in action.tool_calls
                    if tc.function.name == "submit_answer"
                ]
                with (self.output_dir / (env.problem_id + "-answer.json")).open(
                    "w"
                ) as f:
                    json.dump(submitted_answers, f, indent=2)


class AdHocExpt(ConfigurableExpt):
    problem: str = Field(description="Problem to solve.")
    problem_id: str = Field(
        default_factory=lambda: f"analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        description="Arbitrary problem ID - outputs will be stored with this name. "
        "Auto-assigned with timestamp if not provided.",
    )

    input_dir: str = Field(description="Directory containing input data.")
    input_repo: DataRepo | None = Field(
        default=None,
        description="If provided, will set `input_dir` to `input_repo.local_path`.",
    )

    output_dir: str | None = Field(
        default=None,
        description="Directory to save output notebooks. "
        "If not provided, will use `input_dir`.",
    )
    output_repo: DataRepo | None = Field(
        default=None,
        description="If provided, will set `output_dir` to `output_repo.local_path`.",
    )

    agent: AgentConfig
    max_rollout_steps: int | None = None
    verbose_callback: bool = True
    copy_workspace_callback: bool = True
    language: str = "python"

    async def run(self) -> None:
        output_path = Path(cast(str, self.output_dir))
        agent = self.agent.construct_agent()

        # Sanity check to prevent misconfiguration for now - may revisit
        if not getattr(agent, "hide_old_env_states", True):
            raise RuntimeError(
                "It is strongly recommended that hide_old_env_states=True "
                "if the agent provides this option."
            )

        callbacks: list[Callback] = []
        if self.verbose_callback:
            callbacks.append(VerboseCallback())
        if self.copy_workspace_callback:
            callbacks.append(AdHocExptCallback(output_path))

        rollout = RolloutManager(agent=agent, callbacks=callbacks)

        language = NBLanguage.PYTHON if self.language == "python" else NBLanguage.R

        input_path = Path(self.input_dir)
        env = DataAnalysisEnv(
            problem_id=self.problem_id,
            problem=self.problem,
            # doesn't really matter, since there's no answer
            eval_mode=EvalAnswerMode.EXACT,
            # use_tmp_work_dir=True by default, so self.data_dir will be copied
            nb_path=(input_path / "analysis.ipynb"),
            work_dir=input_path,
            language=language,
        )

        await rollout.sample_trajectories(
            environments=[env], max_steps=self.max_rollout_steps
        )

        await env.close()

        if self.output_repo is not None:
            self.output_repo.push(progress=True)

    @model_validator(mode="before")
    @classmethod
    def set_dirs(cls, data):
        if isinstance(data, dict):
            for pfx in ("input", "output"):
                if f"{pfx}_repo" in data:
                    assert f"{pfx}_dir" not in data, (
                        f"Cannot provide both {pfx}_dir and {pfx}_repo"
                    )
                    data[f"{pfx}_repo"] = DataRepo(**data[f"{pfx}_repo"])
                    data[f"{pfx}_dir"] = data[f"{pfx}_repo"].local_path
        return data

    @model_validator(mode="after")
    def post_init(self) -> Self:
        if self.input_repo is not None:
            self.input_repo.pull(progress=True)

        if self.output_repo is not None:
            self.output_repo.mkdir()

        if self.output_dir is None:
            self.output_dir = self.input_dir

        return self
