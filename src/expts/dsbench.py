import json
import logging
from datetime import datetime
from pathlib import Path

from aviary_internal import __version__, utils
from pydantic import Field
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GetKaggleInfo(utils.ConfigurableExpt):
    dataset_repo: utils.DataRepo = Field(
        default_factory=lambda: utils.DataRepo(
            name="baseline-envs/dsbench/data_modeling"
        )
    )

    async def run(self) -> None:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            raise ImportError(
                "Please `pip install kaggle` and set up authentication."
            ) from None

        api = KaggleApi()
        # Will raise if user is not authenticated
        api.authenticate()

        src_dir = Path(self.dataset_repo.local_path)
        competitions = sorted([d.name for d in (src_dir / "data_resplit").glob("*")])
        kaggle_info: dict[str, dict[str, float | bool | list[float]]] = {}

        for comp in tqdm(competitions, desc="Querying Kaggle", ncols=0):
            # Bit ugly: to determine if 'best' is max or min, we get the GT result and compare
            # to the actual submissions. I can't find any documentation saying the leaderboard
            # is ordered.

            try:
                target_result = float(
                    (src_dir / "save_performance/GT" / comp / "result.txt").read_text()
                )
            except FileNotFoundError:
                logger.error(f"Could not find GT result file for {comp} - skipping.")
                continue

            leaderboard = api.competition_leaderboard_view(comp)
            scores = [float(entry.score) for entry in leaderboard if entry.hasScore]
            if not scores:
                logger.error(f"No scores found for {comp} - skipping.")
                continue

            max_score, min_score = max(scores), min(scores)

            if min_score >= target_result:
                # smaller is better
                kaggle_info[comp] = {
                    "best_score": min_score,
                    "max_is_best": False,
                    "scores": scores,
                }

            elif max_score <= target_result:
                # larger is better
                kaggle_info[comp] = {
                    "best_score": max_score,
                    "max_is_best": True,
                    "scores": scores,
                }

            else:
                raise RuntimeError(f"Could not determine best score for {comp}.")

        with (src_dir / "kaggle_submissions.json").open("w") as f:
            json.dump(
                {
                    "metadata": {
                        "description": "Created by data_analysis.expts.dsbench.GetKaggleInfo.",
                        "timestamp": datetime.now().isoformat(),
                        "aviary_internal": __version__,
                    },
                    "kaggle_info": kaggle_info,
                },
                f,
                indent=2,
            )

        self.dataset_repo.push(progress=True)
