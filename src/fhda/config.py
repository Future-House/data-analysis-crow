import os
from pathlib import Path
from futurehouse_client.models import Stage

USE_DOCKER = bool(os.getenv("USE_DOCKER", "true").lower() == "true")
NB_ENVIRONMENT_DOCKER_IMAGE = os.getenv(
    "NB_ENVIRONMENT_DOCKER_IMAGE", "futurehouse/bixbench:aviary-notebook-env"
)

# Some R error messages can be 100,000 of characters
NB_OUTPUT_LIMIT = 3000  # chars
# Streams from a docker container. Don't set to `sys.stdout.fileno()`
# because we want to differentiate from file I/O
DOCKER_STREAM_TYPE_STDOUT = 1
DOCKER_STREAM_TYPE_STDERR = 2

STAGE = os.getenv("STAGE", "local")
if STAGE == "local":
    DATA_STORAGE_PATH = Path("storage")
else:
    DATA_STORAGE_PATH = Path("/storage")

VALID_FROM_TASK_KWARGS = ["run_notebook_on_edit", "exclude_tools"]

# FutureHosue client config
ENVIRONMENT = os.getenv("ENVIRONMENT", "prod")
CROW_STAGE = getattr(Stage, ENVIRONMENT.upper(), Stage.PROD)
PLATFORM_API_KEY = os.getenv("CROW_API_KEY", None)
