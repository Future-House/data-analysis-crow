"""Utilities to run TaskDatasetServers on various notebook task datasets."""

import json
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from aviary.core import TaskDataset, TaskDatasetServer
from fhda.storage import DataRepo
from fhda.utils import collect_notebook_stats
from fhda.data_analysis_env import DataAnalysisEnv
from fhda.dataset import CapsuleDataset, CapsuleDatasetConfig
from scripts.configurable import ConfigurableExpt
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class SaveWorkspaceRequest(BaseModel):
    env_id: str
    traj_id: str
    workspace_repo: DataRepo
    exception: bool
    cost: float
    time: float


class NBTaskDatasetServer(TaskDatasetServer[DataAnalysisEnv]):
    def _setup_routes(self) -> None:
        super()._setup_routes()

        @self.app.post("/save_workspace")
        async def save_workspace(req: SaveWorkspaceRequest):
            async with self.lock:
                env = self._get_env(req.env_id)

            problem_id = env.problem_id
            this_workspace_repo = DataRepo(
                name=f"{req.workspace_repo.name}/{problem_id.replace('/', '-')}-{req.traj_id}"
            )
            this_workspace_repo.mkdir()
            out_dir = Path(this_workspace_repo.local_path)
            logger.info(f"Saving workspace to {this_workspace_repo.name}")

            # # Copy the full output directory
            for file in Path(env.state.work_dir).glob("**/*"):
                if file.suffix in {".ipynb", ".json"}:
                    dest = out_dir / file.relative_to(env.state.work_dir)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, dest)
            res = {
                "problem_id": problem_id,
                "traj_id": req.traj_id,
                "reward": env.state.total_reward,
                "agent_answer": env.state.answer,
                "ideal_answer": env.answer,
                "problem": env.problem,
                "mcq_options": [q.options for q in env.mcqs] if env.mcqs else [],
                "mcq_question": [q.question for q in env.mcqs] if env.mcqs else [],
                "question_rewards": env.question_rewards,
                "cost": req.cost,
                "exception": req.exception,
                "notebook_stats": collect_notebook_stats(env.state.nb),
                "time": req.time,
                "actions": env.state.actions,
                "run_id": req.workspace_repo.name,
                "metadata": env.metadata,
                "insufficient_options": {
                    q.question_id: q.unsure_answer_letter for q in (env.mcqs or [])
                },
            }
            with (out_dir / "metadata.json").open("w") as f:
                json.dump(
                    res,
                    f,
                    indent=4,
                )

            # Push just this specific workspace, not the whole workspace repo
            this_workspace_repo.push(progress=True)
            # # Delete the workspace directory after pushing
            shutil.rmtree(out_dir)


TDataset = TypeVar("TDataset", bound=TaskDataset)


class DatasetServer(ConfigurableExpt, ABC, Generic[TDataset]):
    port: int

    @abstractmethod
    def make_dataset(self) -> TDataset:
        pass

    async def run(self) -> None:
        dataset = self.make_dataset()
        logger.info(f"Starting {dataset.__class__.__name__} server on port {self.port}")
        server = NBTaskDatasetServer(dataset, port=self.port)
        await server.astart()


class CapsuleDatasetServer(DatasetServer[CapsuleDataset]):
    dataset: CapsuleDatasetConfig = Field(default_factory=CapsuleDatasetConfig)

    def make_dataset(self) -> CapsuleDataset:
        return CapsuleDataset(config=self.dataset)
