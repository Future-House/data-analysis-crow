import os
from pathlib import Path

USE_DOCKER = bool(os.getenv("USE_DOCKER", "true").lower() == "true")
USE_R = bool(os.getenv("USE_R", "false").lower() == "true")
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

EVAL = bool(os.getenv("EVAL", "false").lower() == "true")

VALID_FROM_TASK_KWARGS = ["run_notebook_on_edit"]
