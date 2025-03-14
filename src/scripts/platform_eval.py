import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any
import ast

import datasets
from ldp.agent import AgentConfig
from aviary.core import MultipleChoiceEvaluation, MultipleChoiceQuestion
from crow_client import CrowClient
from crow_client.models import Stage, JobRequest, RuntimeConfig
from crow_client.models.app import AuthType
import src.fhda.prompts as prompts

logger = logging.getLogger(__name__)

JOB_NAME = "job-futurehouse-bixbench-crow-dev"
CROW_STAGE = Stage.DEV
RUN_UUID = str(uuid.uuid4())
GCS_ARTIFACT_PATH = "bixbench_data/"
HF_REPO = "futurehouse/bixbench"
MODEL = "claude-3-7-sonnet-20250219"
TEMPERATURE = 1
NUM_RETRIES = 3
MAX_STEPS = 30
AVOID_IMAGES = True
KEEP_NOTEBOOKS = False
POLL_INTERVAL = 60


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


async def submit_questions_and_get_answers(
    client: CrowClient,
    data: list[dict[str, Any]],
    timeout: int = 3600,
) -> dict[str, str]:
    """
    Submit a question to the Crow service and wait for the answer.

    Args:
        client: The CrowJobClient instance
        questions: The MultipleChoiceQuestions to submit
        timeout: Maximum time to wait for an answer in seconds

    Returns:
        The answer string from the agent
    """
    job_answers = {}

    for capsule in data:
        job_id = client.create_job(await prepare_job(capsule))
        logger.info("Submitted job %s with question: %s", job_id, capsule["short_id"])
        job_answers[job_id] = {
            "capsule_id": f"{capsule['short_id']}",
            "answer": None,
            "notebook": None,
            "status": None,
            "question": capsule["questions"],
        }

    start_time = time.time()
    is_complete = False
    while (time.time() - start_time < timeout) and not is_complete:
        for job_id, response in job_answers.items():
            if not response.get("answer"):
                job_status = client.get_job(job_id)
                job_answers[job_id]["status"] = job_status["status"]
                if job_status["status"].lower() == "success":
                    frame = job_status.get("environment_frame", {})
                    answer = frame.get("state", {}).get("state", {}).get("answer", None)
                    if answer:
                        job_answers[job_id]["answer"] = answer
                    if KEEP_NOTEBOOKS:
                        notebook = (
                            frame.get("state", {})
                            .get("state", {})
                            .get("nb_state", None)
                        )
                        if notebook:
                            job_answers[job_id]["notebook"] = notebook

                        logger.info(
                            "Received answer for job %s: %s...", job_id, answer[:20]
                        )
                elif job_status["status"].lower() == "failed":
                    logger.error("Job %s failed.", job_id)
        await asyncio.sleep(POLL_INTERVAL)
        is_complete = all(j["status"] != "in progress" for j in job_answers.values())

    # remove the incomplete jobs
    for job_id, response in job_answers.items():
        if not response["answer"]:
            logger.error(
                "Job %s did not complete in time. (%s seconds)", job_id, timeout
            )
            job_answers["job_id"]["status"] = "timeout"

    return job_answers


async def load_bixbench_data(
    open_question: bool = True,
) -> dict[str, Any]:
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


async def evaluate_dataset(
    data: list[dict[str, Any]],
    runs: int = 1,
    api_key: str | None = None,
    stage: Stage = CROW_STAGE,
) -> dict[str, Any]:
    """
    Evaluate a BixBench dataset by submitting questions and grading responses.

    Args:
        data: The BixBench dataset to evaluate
        runs: The number of times to run the evaluation
        api_key: Authentication token for the Crow service
        stage: The deployment stage to use

    Returns:
        A dictionary of evaluation metrics
    """

    if not api_key:
        api_key = os.environ.get("CROW_API_KEY")

    client = CrowClient(
        stage=stage,
        auth_type=AuthType.API_KEY,
        api_key=api_key,
    )

    # Lists to store results
    all_results = []
    all_evaluations = []
    questions = []

    answers = await submit_questions_and_get_answers(client, data)

    # Save answers to a JSON file for reference
    answers_output_path = f"bixbench_answers_{RUN_UUID}.json"
    with open(answers_output_path, "w") as f:
        json.dump(answers, f, indent=2)

    # for question in questions:
    #     try:
    #         evaluation, extracted_answer = await question.grade(
    #             answers[question.question_id]
    #         )
    #         all_results.append({
    #             "question_id": question.question_id,
    #             "question": question.question,
    #             "ideal_answer": question.ideal_answer,
    #             "options": question.options,
    #             "submitted_answer": answers.get(question.question_id),
    #             "extracted_answer": extracted_answer,
    #             "evaluation": evaluation,
    #         })
    #         all_evaluations.append(evaluation)
    #         logger.info("Question %s: %s", question.question_id, evaluation)
    #     except Exception:
    #         logger.exception("Error processing question %s", question.question_id)

    # accuracy, precision = MultipleChoiceEvaluation.calculate_accuracy_precision(
    #     all_evaluations
    # )

    # metrics = {
    #     "total_questions": len(all_results),
    #     "correct_count": sum(
    #         1 for e in all_evaluations if e == MultipleChoiceEvaluation.CORRECT
    #     ),
    #     "incorrect_count": sum(
    #         1 for e in all_evaluations if e == MultipleChoiceEvaluation.INCORRECT
    #     ),
    #     "unsure_count": sum(
    #         1 for e in all_evaluations if e == MultipleChoiceEvaluation.UNSURE
    #     ),
    #     "accuracy": accuracy,
    #     "precision": precision,
    #     "results": all_results,
    # }

    # logger.info("Evaluation complete. Accuracy: %.2f, Precision: %.2f", accuracy, precision)

    # return metrics


def save_results(
    metrics: dict[str, Any],
    output_path: str = f"bixbench_evaluation_results_{RUN_UUID}.json",
):
    """Save evaluation results to a file."""
    serializable_metrics = {k: v for k, v in metrics.items() if k != "results"}
    serializable_metrics["results"] = []

    for result in metrics["results"]:
        serializable_result = {
            k: str(v) if isinstance(v, (MultipleChoiceEvaluation)) else v
            for k, v in result.items()
        }
        if "options" in serializable_result and isinstance(
            serializable_result["options"], list
        ):
            serializable_result["options"] = [
                str(opt) for opt in serializable_result["options"]
            ]

        serializable_metrics["results"].append(serializable_result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=2)

    logger.info("Results saved to %s", output_path)


async def main():
    parser = argparse.ArgumentParser(description="Evaluate BixBench dataset with Crow")
    parser.add_argument(
        "--output",
        type=str,
        default=f"bixbench_evaluation_results_{RUN_UUID}.json",
        help="Output file path for results",
    )
    args = parser.parse_args()

    data = await load_bixbench_data()
    metrics = await evaluate_dataset(data)

    # save_results(metrics, args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    asyncio.run(main())
