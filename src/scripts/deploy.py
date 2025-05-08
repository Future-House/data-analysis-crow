import os
from pathlib import Path

from ldp.agent import AgentConfig
from futurehouse_client import FutureHouseClient
from futurehouse_client.models import (
    JobDeploymentConfig,
    DockerContainerConfiguration,
    Stage,
    FramePath,
    AuthType,
)
from futurehouse_client.models.app import TaskQueuesConfig

HIGH = False
ENVIRONMENT = "DEV"

ENV_VARS = {
    # "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
    # "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
    "USE_DOCKER": "false",
    "STAGE": ENVIRONMENT,
    "ENVIRONMENT": ENVIRONMENT,
    "API_KEY": os.environ[f"CROW_API_KEY_{ENVIRONMENT}"],
}

CONTAINER_CONFIG = DockerContainerConfiguration(cpu="8", memory="16Gi")

frame_paths = [
    FramePath(path="info.cost", type="text"),
    FramePath(path="state.answer", type="markdown"),
    FramePath(path="state.nb_state_html", type="notebook"),
]

MODEL = "claude-3-7-sonnet-latest"
TEMPERATURE = 1
NUM_RETRIES = 3

# agent = AgentConfig(
#     agent_type="ReActAgent",
#     agent_kwargs={
#         "llm_model": {
#             "name": MODEL,
#             "temperature": TEMPERATURE,
#             "num_retries": NUM_RETRIES,
#         },
#         "hide_old_env_states": True,
#     },
# )

AGENT_MODEL_LIST = [
    {
        "model_name": "anthropic/claude-3-7-sonnet-20250219",
        "litellm_params": {
            "model": "anthropic/claude-3-7-sonnet-20250219",
            "api_key": os.environ["ANTHROPIC_API_KEY"],
        },
    },
    {
        "model_name": "openai/gpt-4.1-2025-04-14",
        "litellm_params": {
            "model": "openai/gpt-4.1-2025-04-14",
            "api_key": os.environ["OPENAI_API_KEY"],
        },
    },
    {
        "model_name": "anthropic/claude-3-5-sonnet-20241022",
        "litellm_params": {
            "model": "anthropic/claude-3-5-sonnet-20241022",
            "api_key": os.environ["ANTHROPIC_API_KEY"],
        },
    },
    {
        "model_name": "openai/gpt-4o-2024-11-20",
        "litellm_params": {
            "model": "openai/gpt-4o-2024-11-20",
            "api_key": os.environ["OPENAI_API_KEY"],
        },
    },
]

AGENT_ROUTER_KWARGS = {
    "set_verbose": True,
    # fallback in list order if the main key fails
    "fallbacks": [
        {
            "openai/gpt-4.1-2025-04-14": [
                "anthropic/claude-3-7-sonnet-20250219",
                "anthropic/claude-3-5-sonnet-20241022",
                "openai/gpt-4o-2024-11-20",
            ]
        }
    ],
}

AGENT_CONFIG = {
    "agent_type": "ReActAgent",
    "agent_kwargs": {
        "llm_model": {
            "name": "anthropic/claude-3-7-sonnet-20250219",
            "config": {
                "model_list": AGENT_MODEL_LIST,
                "router_kwargs": AGENT_ROUTER_KWARGS,
                "fallbacks": [
                    {
                        "openai/gpt-4.1-2025-04-14": [
                            "anthropic/claude-3-7-sonnet-20250219",
                            "anthropic/claude-3-5-sonnet-20241022",
                            "openai/gpt-4o-2024-11-20",
                        ]
                    }
                ],
            },
        },
        "hide_old_env_states": True,
    },
}

CROWS_TO_DEPLOY = [
    JobDeploymentConfig(
        requirements_path=Path("pyproject.toml"),
        path=Path("src"),
        name="data-analysis-crow-high" if HIGH else "data-analysis-crow",
        environment="src.fhda.data_analysis_env.DataAnalysisEnv",
        environment_variables=ENV_VARS,
        agent=AgentConfig(**AGENT_CONFIG),  # type: ignore
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
    client = FutureHouseClient(
        stage=Stage.from_string(os.environ.get("CROW_ENV", ENV_VARS["STAGE"])),
        # stage=Stage.from_string(os.environ.get("CROW_ENV", "LOCAL")),
        organization="FutureHouse",
        auth_type=AuthType.API_KEY,
        api_key=os.environ[f"CROW_API_KEY_{ENV_VARS['STAGE']}"],
    )
    if HIGH:
        print("Using custom deployment Dockerfile")
    else:
        dockerfile_path = Path("src/fhda/Dockerfile.custom_deployment")
        rename_dockerfile(dockerfile_path, "Dockerfile_skip.custom_deployment")

    for crow in CROWS_TO_DEPLOY:
        try:
            client.create_job(crow)
            print(f"Deploying {crow.name}: {client.get_build_status()}")
        except Exception as e:
            print(f"Error deploying {crow.name}: {e}")

    if not HIGH:
        dockerfile_path = Path("src/fhda/Dockerfile_skip.custom_deployment")
        rename_dockerfile(dockerfile_path, "Dockerfile.custom_deployment")
