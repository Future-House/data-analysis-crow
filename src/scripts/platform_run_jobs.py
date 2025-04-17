import asyncio
import json
import logging
import os
from typing import Any
import ast
import time

import datasets
from ldp.agent import AgentConfig
from aviary.core import MultipleChoiceQuestion
from futurehouse_client import FutureHouseClient
from futurehouse_client.models import Stage, JobRequest, RuntimeConfig
from futurehouse_client.models.app import AuthType
import src.fhda.prompts as prompts

logger = logging.getLogger(__name__)

ENV = "PROD"
JOB_NAME = "job-futurehouse-data-analysis-crow-high"
CROW_STAGE = getattr(Stage, ENV)
API_KEY = os.environ.get(f"CROW_API_KEY_{ENV}")
DATASET_NAME = "bb50k"
if DATASET_NAME == "bixbench":
    GCS_ARTIFACT_PATH = "bixbench_data/"
    HF_REPO = "futurehouse/bixbench"
    SUBMIT_ANSWER_PROMPT = prompts.SUBMIT_ANSWER_OPEN
elif DATASET_NAME == "bb50k":
    BB50K_PATH = "local/bb50k/ngs_analysis_rna_seq_dge_dataset_0_qa_metadata_questions_20250404_210834.json"
    GCS_ARTIFACT_PATH = "bb50k/"
    SUBMIT_ANSWER_PROMPT = prompts.SUBMIT_ANSWER_SINGLE
else:
    raise ValueError(f"Dataset {DATASET_NAME} not supported")
NB_LANGUAGE = "PYTHON"
MODEL = "claude-3-7-sonnet-latest"
TEMPERATURE = 1
NUM_RETRIES = 3
MAX_STEPS = 50
AVOID_IMAGES = True
NUM_ITERATIONS = 2
RUN_NAME = "bb50k_v1"
RESULTS_FILE = f"local/bixbench_runs/{RUN_NAME}-{time.strftime('%Y%m%d-%H%M%S')}.json"
RUNTIME_PARAMS = {
    "model": MODEL,
    "temperature": TEMPERATURE,
    "num_retries": NUM_RETRIES,
    "max_steps": MAX_STEPS,
    "avoid_images": AVOID_IMAGES,
    "run_name": RUN_NAME,
}
MINI_MODE = False
MINUTES = 60
SLEEP_TIME = 0.5 * MINUTES


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

            {prompts.CHAIN_OF_THOUGHT_AGNOSTIC.format(language=NB_LANGUAGE)}
            {SUBMIT_ANSWER_PROMPT}
            {prompts.GENERAL_NOTEBOOK_GUIDELINES.format(language=NB_LANGUAGE)}"""

    if AVOID_IMAGES:
        task += prompts.AVOID_IMAGES
    runtime_params = RUNTIME_PARAMS.copy()
    runtime_params["categories"] = capsule["categories"]
    agent = AgentConfig(
        agent_type="ReActAgent",
        agent_kwargs={
            "llm_model": {
                "name": MODEL,
                "temperature": TEMPERATURE,
                "num_retries": NUM_RETRIES,
            },
            "hide_old_env_states": True,
            "runtime_params": runtime_params,  # type: ignore
        },
    )
    job_data = JobRequest(
        name=JOB_NAME,
        query=task,
        runtime_config=RuntimeConfig(
            agent=agent,
            max_steps=MAX_STEPS,
            upload_id=capsule["data_folder"],
            environment_config={
                "run_notebook_on_edit": False,
                "eval": True,
                "language": NB_LANGUAGE,
            },
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
                "categories": capsule["categories"],
                "uuid": capsule["uuid"],
                "questions": processed_questions,
            }
        )
    return processed_dataset


async def load_bb50k_data(
    open_question: bool = True,
) -> list[dict[str, Any]]:
    """Load the BixBench dataset."""
    data = json.load(open("local/bb50k/single_dataset_per_wf.json"))
    processed_data = []
    for i in data:
        processed_data.append(
            {
                "data_folder": f"{GCS_ARTIFACT_PATH}/{i['workflow']}/{i['dataset'].replace('dataset_', '')}",
                "short_id": i["qa_id"],
                "generator_class": i["generator_class"],
                "uuid": i["qa_id"],
                "domain": i["domain"],
                "workflow": i["workflow"],
                "dataset": i["dataset"],
                "source_node": i["source_node"],
                "node_execution_order": i["node_execution_order"],
                "answer_type": i["answer_type"],
                "template": i["template"],
                "questions": [
                    MultipleChoiceQuestion(
                        question=i["question"],
                        options=[],
                        ideal_answer=str(i["answer_value"]),
                        shuffle_seed=MultipleChoiceQuestion.SEED_USING_QUESTION,
                        prompt_without_options=open_question,
                    )
                ],
            }
        )
    return processed_data


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

    client = FutureHouseClient(
        stage=CROW_STAGE,
        auth_type=AuthType.API_KEY,
        api_key=API_KEY,
    )

    jobs = []
    for iteration in range(1, NUM_ITERATIONS + 1):
        logger.info("Iteration %s", iteration)
        for capsule in data:
            job_request = await prepare_job(capsule)
            try:
                job_id = client.create_job(job_request)
                logger.info(
                    "Submitted job %s with capsule: %s", job_id, capsule["short_id"]
                )
            except Exception as e:
                logger.error(
                    "Error submitting job %s with capsule: %s",
                    job_id,
                    capsule["short_id"],
                )
                logger.error(e)
                job_id = "FAILED"
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
        # This is necessary when running with local backend
        if SLEEP_TIME and iteration < NUM_ITERATIONS:
            logger.info("Sleeping for %s seconds", SLEEP_TIME)
            time.sleep(SLEEP_TIME)

    return jobs


async def save_results(jobs: list[dict[str, Any]], output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=4)


async def main():
    if DATASET_NAME == "bixbench":
        data = await load_bixbench_data()
    elif DATASET_NAME == "bb50k":
        data = await load_bb50k_data()
    else:
        raise ValueError(f"Dataset {DATASET_NAME} not supported")

    if MINI_MODE:
        data = data[:2]

    jobs = await submit_jobs(data)
    await save_results(jobs, RESULTS_FILE)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    asyncio.run(main())
