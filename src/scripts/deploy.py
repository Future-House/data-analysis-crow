import os
from pathlib import Path

from crow_client import CrowClient
from crow_client.models import (
    CrowDeploymentConfig,
    DockerContainerConfiguration,
    Stage,
)

ENV_VARS = {
    "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
    "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
    "USE_R": "false",
    "USE_DOCKER": "false",
}

CONTAINER_CONFIG = DockerContainerConfiguration(cpu="2", memory="4Gi")

CROWS_TO_DEPLOY = [
    CrowDeploymentConfig(
        requirements_path=Path("pyproject.toml"),
        path=Path("src"),
        name="data-analysis-crow",
        environment="src.app.data_analysis_env.DataAnalysisEnv",
        environment_variables=ENV_VARS,
        agent="ldp.agent.ReActAgent",
        container_config=CONTAINER_CONFIG,
        force=True,
    ),
]

if __name__ == "__main__":
    client = CrowClient(
        stage=Stage.from_string(os.environ["CROW_ENV"]), organization="FutureHouse"
    )
    for crow in CROWS_TO_DEPLOY:
        try:
            client.create_crow(crow)
            print(f"Deploying {crow.name}: {client.get_build_status()}")
        except Exception as e:
            print(f"Error deploying {crow.name}: {e}")
