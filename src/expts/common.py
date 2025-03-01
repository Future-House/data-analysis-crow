import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from aviary.core import (
    Environment,
    Message,
    Messages,
    TaskDatasetClient,
    TaskEnvironmentClient,
    ToolRequestMessage,
)

# from aviary_internal import utils
# from aviary_internal.graph.multiple_completion_op import (
#     SequentialMultipleCompletionLLMCallOp,
# )
from ldp.agent import Agent
from ldp.alg import Callback
from ldp.data_structures import Trajectory, Transition
from llmclient.cost_tracker import GLOBAL_COST_TRACKER
from fhda.storage import DataRepo

logger = logging.getLogger(__name__)


class VerboseCallback(Callback):
    """Callback to visualize notebook state before each transition."""

    async def before_transition(
        self,
        traj_id: str,
        agent: Agent,
        env: Environment,
        agent_state: Any,
        obs: list[Message],
    ) -> None:
        for msg in obs:
            if msg.content:
                logger.info("VerboseCallback:\n%s", msg.content)


class SaveWorkspaceCallback(Callback):
    def __init__(self, dataset_client: TaskDatasetClient, workspace_repo: DataRepo):
        self.dataset_client = dataset_client
        self.workspace_repo = workspace_repo

    async def before_transition(
        self,
        traj_id: str,
        agent: Agent,
        env: Environment,
        agent_state,
        obs: list[Message],
    ) -> None:
        self.start = time.time()

    async def after_transition(
        self,
        traj_id: str,
        agent: Agent,
        env: TaskEnvironmentClient,  # type: ignore[override]
        transition: Transition,
    ) -> None:
        if not any((transition.done, transition.truncated)):
            # only save if the trajectory is over
            return

        # TODO: figure out how to support overwrite flag
        async with self.dataset_client.get_http_client() as client:
            response = await client.post(
                "/save_workspace",
                json={
                    "env_id": env.state.env_id,
                    "traj_id": traj_id,
                    "workspace_repo": self.workspace_repo.model_dump(),
                    "exception": transition.failed,
                    "cost": GLOBAL_COST_TRACKER.lifetime_cost_usd,
                    "time": time.time() - self.start,
                },
            )
            if not response.is_success:
                logger.error(f"Failed to save workspace: {response.content!r}")


class LoggingCallback(Callback):
    def __init__(self, output_repo: DataRepo):
        self.output_repo = output_repo
        self.rewards: list[float] = []

    async def after_eval_step(self, trajectories: Sequence[Trajectory]) -> None:
        this_batch_rewards = [
            sum(step.reward for step in traj.steps) for traj in trajectories
        ]
        self.rewards += this_batch_rewards
        self.reward_mean, self.reward_stde = self._compute_summary_stats(self.rewards)
        # NOTE: assumes that positive reward implies success
        self.acc_mean, self.acc_stde = self._compute_summary_stats(
            [r > 0 for r in self.rewards]
        )

        print(flush=True)
        logger.info(
            f"Accuracy={self.acc_mean:.2f}±{self.acc_stde:.2f}; "
            f"Rewards={self.reward_mean:.2f}±{self.reward_stde:.2f}"
        )

    async def after_eval_loop(self) -> None:
        results = {
            "reward_mean": self.reward_mean,
            "reward_stde": self.reward_stde,
            "acc_mean": self.acc_mean,
            "acc_stde": self.acc_stde,
        }

        with open(Path(self.output_repo.local_path) / "results.json", "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"These are the results: {results}")
        with open(Path(self.output_repo.local_path) / "rewards.json", "w") as f:
            json.dump(self.rewards, f)

    def _compute_summary_stats(self, metrics: list) -> tuple[float, float]:
        return np.mean(metrics), np.std(metrics) / np.sqrt(len(metrics) + 1)


def prev_choice_rep_fn(output_messages: Messages) -> str:
    rep = ""
    for i, msg in enumerate(output_messages):
        assert isinstance(msg, ToolRequestMessage)
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]

        match tc.function.name:
            case "submit_answer":
                rep += f"Option {i + 1}: Submitting solution."

            case "list_workdir":
                rep += f"Option {i + 1}: Listing workdir contents."

            case "edit_cell":
                idx = tc.function.arguments.get("idx", None)
                if idx is None:
                    rep += f"Option {i + 1}: Adding cell:\n```\n"
                else:
                    rep += f"Option {i + 1}: Editing cell {idx}:\n```\n"
                rep += tc.function.arguments["contents"] + "\n```\n"

            case _:
                # Don't throw error for now, since there may be a case I haven't considered
                # But eventually this should be an exception.
                logger.error(f"Unexpected tool call: {tc.function.name}")

        rep += "\n"

    return rep
