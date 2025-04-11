import os
from pathlib import Path

from crow_client import CrowClient
from crow_client.models import (
    CrowDeploymentConfig,
    DockerContainerConfiguration,
    Stage,
    FramePath,
    AuthType,
)
from crow_client.models.app import TaskQueuesConfig

HIGH = True

ENV_VARS = {
    "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
    "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
    "USE_R": "false",
    "USE_DOCKER": "false",
    "STAGE": "PROD",
}

CONTAINER_CONFIG = DockerContainerConfiguration(cpu="2", memory="4Gi")

frame_paths = [
    FramePath(path="info.cost", type="text"),
    FramePath(path="state.answer", type="markdown"),
    FramePath(path="state.nb_state_html", type="notebook"),
]

CROWS_TO_DEPLOY = [
    CrowDeploymentConfig(
        requirements_path=Path("pyproject.toml"),
        path=Path("src"),
        name="data-analysis-crow-high" if HIGH else "data-analysis-crow",
        environment="src.fhda.data_analysis_env.DataAnalysisEnv",
        environment_variables=ENV_VARS,
        agent="ldp.agent.ReActAgent",
        container_config=CONTAINER_CONFIG,
        force=True,
        frame_paths=frame_paths,
        timeout=3600,
        task_queues_config=TaskQueuesConfig(
            name="data-analysis-crow",
            max_running_jobs=300,
        ),
    ),
]


def rename_dockerfile(path: Path, new_name: str):
    if path.exists():
        path.rename(path.parent / new_name)
        print(f"Renamed {path} to {new_name}")
    else:
        print(f"Warning: {path} does not exist")


if __name__ == "__main__":
    client = CrowClient(
        # stage=Stage.from_string(os.environ.get("CROW_ENV", ENV_VARS["STAGE"])),
        stage=Stage.from_string(os.environ.get("CROW_ENV", "LOCAL")),
        organization="FutureHouse",
        auth_type=AuthType.API_KEY,
        api_key=os.environ[f"CROW_API_KEY_{ENV_VARS['STAGE']}"],
    )

    if not HIGH:
        dockerfile_path = Path("src/fhda/Dockerfile.custom_deployment")
        rename_dockerfile(dockerfile_path, "Dockerfile_skip.custom_deployment")

    for crow in CROWS_TO_DEPLOY:
        try:
            client.create_crow(crow)
            print(f"Deploying {crow.name}: {client.get_build_status()}")
        except Exception as e:
            print(f"Error deploying {crow.name}: {e}")

    if not HIGH:
        dockerfile_path = Path("src/fhda/Dockerfile_skip.custom_deployment")
        rename_dockerfile(dockerfile_path, "Dockerfile.custom_deployment")
