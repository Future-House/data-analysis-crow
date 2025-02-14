import logging
import random
from collections.abc import Mapping, Sequence

from aviary.core import (
    TaskDatasetClient,
)
from aviary_internal import utils
from aviary_internal.agent import DQNAgentVariant
from aviary_internal.agent.dqn_agent import LLMSamplingMode
from aviary_internal.alg.optimizer.dqn import DQNOptimizer
from aviary_internal.nn.sft_optimizer import LocalLLMSFTOptimizer
from aviary_internal.serialization import disable_serialization_backend
from cloning.expts.local_sft import CloningOnlineLocalTrainingExpt
from gsm8k.expts.dqn.online import GSM8kDQNOnlineTrainingExpt
from ldp.alg.callbacks import Callback
from ldp.alg.runners import OnlineTrainerConfig
from ldp.data_structures import Trajectory

from .client import TaskDatasetSplit
from .common import SaveWorkspaceCallback, prev_choice_rep_fn

logger = logging.getLogger(__name__)


class EnvServerConfig(utils.ConfigModel):
    host: str
    port: int
    request_timeout: float | None = 300.0

    async def make_datasets(self) -> dict[str, TaskDatasetClient]:
        base_dataset = await TaskDatasetClient.create(
            server_url=f"http://{self.host}:{self.port}",
            request_timeout=self.request_timeout,
        )
        return {
            "train_dataset": TaskDatasetSplit.TRAIN.get_random_split(base_dataset),
            "eval_dataset": TaskDatasetSplit.EVAL.get_random_split(base_dataset),
        }


class NBDQNOnlineTrainingExpt(GSM8kDQNOnlineTrainingExpt):
    env: EnvServerConfig

    async def make_datasets(self) -> dict[str, TaskDatasetClient]:
        return await self.env.make_datasets()

    def make_callbacks(
        self,
        agent: DQNAgentVariant,
        optimizer: DQNOptimizer,
        datasets: Mapping[str, TaskDatasetClient],
    ) -> list[Callback]:
        callbacks = super().make_callbacks(agent, optimizer, datasets)
        callbacks.append(
            SaveWorkspaceCallback(
                dataset_client=datasets["train_dataset"],
                workspace_repo=utils.DataRepo(
                    name=f"{self.output_repo.name}-workspaces"
                ),
            )
        )
        return callbacks

    def make_agent(self, **kwargs) -> DQNAgentVariant:
        if self.agent.llm_sampling_mode == LLMSamplingMode.SEQUENTIAL:
            self.agent.llm_kwargs["prev_choice_rep_fn"] = prev_choice_rep_fn
        return super().make_agent(**kwargs)


class NBOnlineTrainingConfig(OnlineTrainerConfig):
    save_all_checkpoints: bool = True
    num_val_trajs: int
    num_train_trajs: int | None = None


class NBOnlineLocalTrainingExpt(CloningOnlineLocalTrainingExpt):
    env: EnvServerConfig
    trainer: NBOnlineTrainingConfig

    async def _get_demonstration_examples(
        self, opt: LocalLLMSFTOptimizer
    ) -> tuple[list[dict], list[dict]]:
        backend = await self.make_backend()
        trajectories = await backend.get_trajectories()

        random.Random(self.data_seed).shuffle(trajectories)
        val_trajs = self._filter_trajectories(
            trajectories[: self.trainer.num_val_trajs], opt
        )
        train_trajs = self._filter_trajectories(
            trajectories[self.trainer.num_val_trajs :][: self.trainer.num_train_trajs],
            opt,
        )
        logger.info(
            f"Loaded {len(train_trajs)} ({len(val_trajs)}) train (val) trajectories."
        )

        # Disable the backend so we don't accidentally overwrite input data
        disable_serialization_backend()

        # convert to examples
        train_examples = self._trajs_to_examples(train_trajs, opt)
        val_examples = self._trajs_to_examples(val_trajs, opt)
        return train_examples, val_examples

    def _filter_trajectories(
        self, trajectories: Sequence[Trajectory], opt: LocalLLMSFTOptimizer
    ):
        return [t for t in trajectories if opt.trajectory_passes(t)]
