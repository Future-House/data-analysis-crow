import random
import typing
from enum import StrEnum, auto

from aviary.core import (
    TaskDatasetClient,
    TaskEnvironmentClient,
)
from ldp.alg.callbacks import ComputeTrajectoryMetricsMixin


class TaskDatasetSubsetClient(TaskDatasetClient, ComputeTrajectoryMetricsMixin):
    """Convenience class to subset a dataset using a single server."""

    def __init__(self, client: TaskDatasetClient, task_idcs: list[int]) -> None:
        super().__init__(
            server_url=client.server_url, request_timeout=client.request_timeout
        )
        self.idcs = task_idcs

    def __len__(self) -> int:
        return len(self.idcs)

    def get_new_env_by_idx(self, idx: int) -> TaskEnvironmentClient:
        return super().get_new_env_by_idx(self.idcs[idx])


class TaskDatasetSplit(StrEnum):
    TRAIN = auto()
    EVAL = auto()
    TEST = auto()
    ALL = auto()

    def get_random_split(
        self, dataset_client: TaskDatasetClient, seed: int = 0
    ) -> TaskDatasetClient:
        if self == TaskDatasetSplit.ALL:
            return dataset_client

        # Slightly hacky way to make a split for now
        # Split the dataset into a 80/10/10 split using a deterministic seed
        n_total = len(dataset_client)
        all_idcs = random.Random(seed).sample(range(n_total), n_total)

        match self:
            case TaskDatasetSplit.TRAIN:
                idcs = all_idcs[: int(0.8 * n_total)]
            case TaskDatasetSplit.EVAL:
                idcs = all_idcs[int(0.8 * n_total) : int(0.9 * n_total)]
            case TaskDatasetSplit.TEST:
                idcs = all_idcs[int(0.9 * n_total) :]

            case _:
                typing.assert_never(self)

        return TaskDatasetSubsetClient(dataset_client, idcs)
