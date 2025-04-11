import time
import os
import asyncio
import json
import ast
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import logging
from pathlib import Path
from crow_client import CrowClient
from crow_client.models import AuthType, Stage, JobResponse
from aviary.utils import MultipleChoiceQuestion, eval_answer, EvalAnswerMode


# Configure logging
logger = logging.getLogger(__name__)

ENV = "PROD"


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure logging"""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )


def create_client(
    api_key: Optional[str] = None,
    stage: Stage = getattr(Stage, ENV),
    organization: str = "FutureHouse",
) -> CrowClient:
    """Create and return a CrowClient instance."""
    return CrowClient(
        stage=stage,
        organization=organization,
        auth_type=AuthType.API_KEY,
        api_key=api_key or os.environ[f"CROW_API_KEY_{ENV}"],
    )


def load_job_data(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load Job data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of job data records
    """
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loaded %s records from %s", len(data), file_path.name)
    if data:
        logger.info("First record job_id: %s", data[0]["job_id"])

    return data


async def fetch_jobs_batch(
    client: CrowClient, job_ids: List[str], batch_size: int = 10
) -> List[Dict[str, Any]]:
    """Fetch jobs in batches to avoid memory issues.

    Args:
        client: CrowClient instance
        job_ids: List of job IDs to fetch
        batch_size: Number of jobs to fetch in each batch

    Returns:
        List of fetched jobs
    """

    async def get_job_async(job_id: str) -> JobResponse:
        return await asyncio.to_thread(
            client.get_job, job_id, False, True
        )  # False for history, True for verbose

    results = []

    for i in range(0, len(job_ids), batch_size):
        batch = job_ids[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(job_ids) + batch_size - 1) // batch_size
        logger.info(
            "Processing batch %s/%s: %s jobs", batch_num, total_batches, len(batch)
        )

        tasks = [asyncio.create_task(get_job_async(job_id)) for job_id in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        if i + batch_size < len(job_ids):
            await asyncio.sleep(0.5)
    results = [i.model_dump() for i in results]
    return results


def merge_data(
    bixbench_data: List[Dict[str, Any]], fetched_jobs: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Merge BixBench data with fetched jobs.

    Args:
        bixbench_data: Original BixBench data
        fetched_jobs: Fetched job trajectories

    Returns:
        Merged DataFrame
    """
    results = []
    for capsule, trajectory in zip(bixbench_data, fetched_jobs):
        results.append({**capsule, **trajectory})

    return pd.DataFrame(results)


def parse_questions(row: pd.Series) -> Optional[Dict[str, Any]]:
    """Extract question details for a specific question key."""
    idx = next(
        (
            i
            for i, q in enumerate(row["questions"])
            if q["question_id"] == row["question_keys"]
        ),
        None,
    )
    if idx is not None:
        return row["questions"][idx]
    else:
        logger.warning("Index %s out of range for row %s", idx, row)
        logger.warning("Questions length: %s", len(row["questions"]))
        return None


def parse_answer(row: pd.Series) -> str:
    """Parse the answer for a specific question key."""
    answer = row["answer"]
    try:
        if isinstance(answer, dict):
            return answer[row["question_keys"]]
        else:
            return answer
    except Exception:
        return "No answer"


def fetch_answer(frame: Dict[str, Any]) -> Any:
    """Extract answer from environment frame."""
    try:
        return ast.literal_eval(frame["state"]["state"]["answer"])
    except Exception:
        return "No answer"


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the DataFrame for evaluation.

    Args:
        df: Input DataFrame

    Returns:
        Processed DataFrame ready for evaluation
    """
    print(df.head())
    df["answer"] = df["environment_frame"].apply(fetch_answer)
    df["question_keys"] = df["questions"].apply(lambda x: [i["question_id"] for i in x])
    exploded = df.explode("question_keys")
    exploded["answer"] = exploded.apply(parse_answer, axis=1)
    exploded["question"] = exploded.apply(parse_questions, axis=1)
    exploded["question"] = exploded["question"].apply(
        lambda x: MultipleChoiceQuestion.model_validate(x)  # type: ignore
    )

    return exploded


async def grade_single(row: pd.Series) -> float:
    """Grade a single answer."""
    grade_result = await eval_answer(
        row["answer"],
        row["question"].ideal_answer,
        row["question"].question_prompt,
        EvalAnswerMode.LLM,
    )
    return grade_result


async def grade_all_questions_concurrent(exploded: pd.DataFrame) -> pd.DataFrame:
    """Grade all questions concurrently.

    Args:
        exploded: DataFrame with questions to grade

    Returns:
        DataFrame with grading results
    """
    tasks = [asyncio.create_task(grade_single(row)) for _, row in exploded.iterrows()]
    results = await asyncio.gather(*tasks)

    result_df = exploded.copy()
    result_df["grade_result"] = results
    return result_df


def print_results(graded_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """Print evaluation results."""
    success_count = (graded_df["status"] == "success").sum()
    total_count = len(graded_df)
    success_percentage = success_count / total_count if total_count > 0 else 0

    logger.info(
        "Success count: %s out of %s (%.2f%%)",
        success_count,
        total_count,
        success_percentage * 100,
    )
    logger.info("Average accuracy: %.2f", graded_df["grade_result"].mean())

    return {
        "success_count": success_count,
        "total_count": total_count,
        "success_percentage": success_percentage,
        "average_accuracy": graded_df["grade_result"].mean(),
    }


def save_results(graded_df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """Save graded results to a file.

    Args:
        graded_df: DataFrame with grading results
        output_path: Path to save the results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graded_df.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)


async def main(
    job_file_path: Union[str, Path],
    output_path: Union[str, Path],
    job_request_batch_size: int = 10,
    api_key: Optional[str] = None,
    stage: Stage = getattr(Stage, ENV),
    organization: str = "FutureHouse",
    log_level: int = logging.INFO,
) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
    """Main function to run the evaluation pipeline.

    Args:
        job_file_path: Path to Job data file with all the job IDs
        output_path: Path to save results
        job_request_batch_size: Batch size for job requests
        api_key: API key for CrowClient
        stage: Stage for CrowClient
        organization: Organization for CrowClient
        log_level: Logging level

    Returns:
        Tuple containing the graded DataFrame and evaluation results summary
    """
    # Setup logging
    setup_logging(log_level)

    # Start timing
    start_time = time.time()

    logger.info("Starting evaluation with job file: %s", job_file_path)
    logger.info("Results will be saved to: %s", output_path)

    # Create client
    client = create_client(api_key, stage, organization)

    # Load data
    job_data = load_job_data(job_file_path)

    # Get job IDs
    job_ids = [i["job_id"] for i in job_data]
    logger.info("Processing %s job IDs", len(job_ids))

    # Fetch jobs
    fetched_jobs = await fetch_jobs_batch(client, job_ids, job_request_batch_size)

    # Merge data
    df = merge_data(job_data, fetched_jobs)
    logger.info("Created DataFrame with %s rows", len(df))

    # Prepare DataFrame
    exploded = prepare_dataframe(df)
    logger.info("Exploded DataFrame has %s rows", len(exploded))

    # Grade questions
    logger.info("Starting grading process...")
    graded_df = await grade_all_questions_concurrent(exploded)
    logger.info("Grading completed")

    # Print results
    results = print_results(graded_df)

    # Save results
    save_results(graded_df, output_path)

    # Calculate and log elapsed time
    elapsed_time = time.time() - start_time
    logger.info(
        "Evaluation completed successfully in %.2f seconds (%.2f minutes)",
        elapsed_time,
        elapsed_time / 60,
    )

    return graded_df, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate BixBench data")
    parser.add_argument(
        "--job-file-path",
        type=str,
        default="local/bixbench_runs/baseline-3.7-single-cell-run2-20250325-065452.json",
        help="Path to Job data file with all the job IDs",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="local/bixbench_runs/",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for job requests"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for CrowClient (defaults to CROW_API_KEY env var)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Convert string log level to actual logging level
    log_level = getattr(logging, args.log_level)

    job_file_path = Path(args.job_file_path)
    output_path = Path(args.output_path)
    output_path = output_path / f"{job_file_path.name.replace('.json', '')}-eval.csv"

    asyncio.run(
        main(
            job_file_path,
            output_path,
            args.batch_size,
            args.api_key,
            log_level=log_level,
        )
    )
