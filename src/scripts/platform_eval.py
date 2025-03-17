import asyncio
import json
import logging
import os
import uuid
from typing import Any
import ast
import time

import datasets
from ldp.agent import AgentConfig
from aviary.core import MultipleChoiceQuestion
from crow_client import CrowClient
from crow_client.models import Stage, JobRequest, RuntimeConfig
from crow_client.models.app import AuthType
import src.fhda.prompts as prompts

logger = logging.getLogger(__name__)

JOB_NAME = "job-futurehouse-bixbench-crow-dev"
CROW_STAGE = Stage.LOCAL
API_KEY = os.environ.get("CROW_API_KEY")
RUN_UUID = str(uuid.uuid4())
GCS_ARTIFACT_PATH = "bixbench_data/"
HF_REPO = "futurehouse/bixbench"
MODEL = "claude-3-7-sonnet-20250219"
TEMPERATURE = 1
NUM_RETRIES = 3
MAX_STEPS = 30
AVOID_IMAGES = True
KEEP_NOTEBOOKS = False
NUM_ITERATIONS = 1
RUN_NAME = "baseline-3.7"
RESULTS_FILE = f"local/bixbench_runs/{RUN_NAME}-{time.strftime('%Y%m%d-%H%M%S')}.json"
RUNTIME_PARAMS = {
    "model": MODEL,
    "temperature": TEMPERATURE,
    "num_retries": NUM_RETRIES,
    "max_steps": MAX_STEPS,
    "avoid_images": AVOID_IMAGES,
    "run_name": RUN_NAME,
}


async def prepare_job(capsule: dict[str, Any]) -> JobRequest:
    """
    Prepare a job for a capsule.
    """

    formatted_question = "\n-------\n".join(
        [i.question_prompt for i in capsule["questions"]]
    )

    task = f"""\
            Here is the user query to address:

            <query>
            {formatted_question}
            </query>

            {prompts.CHAIN_OF_THOUGHT_AGNOSTIC}
            {prompts.SUBMIT_ANSWER_OPEN}
            {prompts.GENERAL_NOTEBOOK_GUIDELINES}"""

    if AVOID_IMAGES:
        task += prompts.AVOID_IMAGES
    agent = AgentConfig(
        agent_type="ReActAgent",
        agent_kwargs={
            "llm_model": {
                "model": MODEL,
                "temperature": TEMPERATURE,
                "num_retries": NUM_RETRIES,
            },
            "hide_old_env_states": True,
            "runtime_params": RUNTIME_PARAMS,  # type: ignore
        },
    )
    job_data = JobRequest(
        name=JOB_NAME,
        query=task,
        runtime_config=RuntimeConfig(
            agent=agent, max_steps=MAX_STEPS, upload_id=capsule["data_folder"]
        ),
    )
    return job_data


async def load_bixbench_data(
    open_question: bool = True,
) -> list[dict[str, Any]]:
    """Load the BixBench dataset."""
    data = datasets.load_dataset(HF_REPO, split="train").to_list()
    processed_dataset = []
    for capsule in data:
        raw_questions = ast.literal_eval(capsule["questions"])
        processed_questions = [
            MultipleChoiceQuestion(
                question=i["question"],
                options=[
                    i["ideal_answer"],
                    i["distractor_1"],
                    i["distractor_2"],
                    i["distractor_3"],
                ],
                ideal_answer=i["ideal_answer"],
                shuffle_seed=MultipleChoiceQuestion.SEED_USING_QUESTION,
                prompt_without_options=open_question,
                question_id=i["id"],
            )
            for i in raw_questions
        ]
        processed_dataset.append(
            {
                "data_folder": GCS_ARTIFACT_PATH
                + capsule["data_folder"].replace(".zip", ""),
                "short_id": capsule["short_id"],
                "uuid": capsule["uuid"],
                "questions": processed_questions,
            }
        )
    return processed_dataset


async def submit_jobs(
    data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Submit a question to the Crow service and wait for the answer.

    Args:
        client: The CrowJobClient instance
        questions: The MultipleChoiceQuestions to submit
        timeout: Maximum time to wait for an answer in seconds

    Returns:
        The answer string from the agent
    """

    client = CrowClient(
        stage=CROW_STAGE,
        auth_type=AuthType.API_KEY,
        api_key=API_KEY,
    )

    jobs = []
    for iteration in range(1, NUM_ITERATIONS + 1):
        for capsule in data[:1]:
            job_request = await prepare_job(capsule)
            job_id = client.create_job(job_request)
            logger.info(
                "Submitted job %s with question: %s", job_id, capsule["short_id"]
            )
            job_metadata = {
                **job_request.model_dump(),
                **capsule,
                "job_id": job_id,
                "iteration": iteration,
            }
            job_metadata["questions"] = [
                i.model_dump() for i in job_metadata["questions"]
            ]
            jobs.append(job_metadata)

    return jobs


async def save_results(jobs: list[dict[str, Any]], output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=4)


async def main():
    data = await load_bixbench_data()
    jobs = await submit_jobs(data)
    await save_results(jobs, RESULTS_FILE)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    asyncio.run(main())
