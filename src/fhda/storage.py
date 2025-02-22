"""Module containing storage utilities for Google Cloud Platform (GCP) and Google Cloud Storage (GCS)."""

import asyncio
import base64
import concurrent.futures
import logging
import os
import re
import shutil
from typing import Self

import aiofiles
import google.api_core.exceptions
import google.auth
import httpx
from google.cloud import secretmanager
from google.cloud.storage import Client
from google_crc32c import Checksum
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from requests.adapters import HTTPAdapter
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "aviary-storage"
DEFAULT_KEY = os.path.expanduser("~/.keys/aviary-storage-service.json")
DEFAULT_STORAGE_PATH = os.path.expanduser("~/aviary_data/")
DEFAULT_GCP_PROJECT_ID = "362315315966"  # Corresponds to "paperqa" project
MAX_THREADS = 100


def validate_google_app_creds() -> None:
    """Validate we have a google application credential set.

    Priority order:
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable
    2. Default key path
    3. Fetch key from Secret Manager (and cache in default key path)
    """
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        # This code path is mostly meant for CI, which uses a GitHub secret,
        # not a key file.
        return

    if os.path.exists(DEFAULT_KEY):
        return

    logger.info("aviary-storage-service account key not found, attempting to fetch...")
    client = secretmanager.SecretManagerServiceClient()
    try:
        response = client.access_secret_version(
            request={
                "name": f"projects/{DEFAULT_GCP_PROJECT_ID}/secrets/AVIARY-STORAGE-SERVICE-KEY/versions/latest"
            }
        )
    except google.api_core.exceptions.RetryError as e:
        # Could use better error handling here, but it's a little confusing how they chain exceptions
        raise RuntimeError(
            "Failed to fetch 'aviary-storage-service' key from Secret Manager. "
            "Confirm that you are authenticated by running `gcloud auth application-default login`"
        ) from e

    payload = response.payload.data.decode("UTF-8")
    os.makedirs(os.path.dirname(DEFAULT_KEY), exist_ok=True)
    with open(DEFAULT_KEY, "w") as f:  # noqa: FURB103
        f.write(payload)
    logger.info(
        f"Successfully stored aviary-storage-service account key in {DEFAULT_KEY}."
    )


def auth_required(func):
    """Decorator to ensure that the user is authenticated with GCP before calling."""

    def wrapper(*args, **kwargs):
        validate_google_app_creds()
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", DEFAULT_KEY)
        google.auth.default()  # Check authentication
        return func(*args, **kwargs)

    return wrapper


class DataRepo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description=(
            "Subpath to the target directory within the cloud bucket `bucket`,"
            " something like 'relative/path/to/sub/bucket'. Set to empty string to use"
            " the root of the bucket."
        ),
    )

    local_path: str = Field(
        default="UNSET",
        description=(
            "Set to the target directory to mirror files. If left as the default of"
            " 'UNSET', it will be set to be <DEFAULT_STORAGE_PATH>/<self.name>."
        ),
    )
    bucket: str = Field(
        default=DEFAULT_BUCKET,
        description=(
            "Cloud bucket name like 'aviary-storage'. An analogy with a local"
            " filesystem is the drive (e.g. 'C:' on Windows)."
        ),
    )

    validate_gcs_auth: bool = Field(
        default=True,
        description=(
            "Set True (default) to validate GCS authentication at construction time."
        ),
    )

    def __bool__(self) -> bool:
        """Determines truthiness based on whether the name and local_path are set."""
        return bool(self.name and self.local_path and self.local_path != "UNSET")

    @staticmethod
    def get_local_storage_path() -> str:
        return os.getenv("AVIARY_LOCAL_STORAGE", DEFAULT_STORAGE_PATH)

    @field_validator("name")
    @classmethod
    def _remove_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def gcs_name(self) -> str:
        return f"{self.name}/"

    @model_validator(mode="after")
    def set_local_path(self) -> Self:
        if self.local_path == "UNSET":
            self.local_path = os.path.join(self.get_local_storage_path(), self.name)
        return self

    def mkdir(self, remove_existing: bool = False):
        if remove_existing:
            shutil.rmtree(self.local_path, ignore_errors=True)
        os.makedirs(self.local_path, exist_ok=True)

    @auth_required
    def push(
        self,
        overwrite: bool = False,
        include: re.Pattern | str | None = None,
        exclude: re.Pattern | str | None = None,
        progress: bool = False,
    ) -> None:
        logger.info(f"Pushing data repo: {self.name}")
        bucket = _get_gcs_client().get_bucket(self.bucket)

        include = _resolve_pattern(include)
        exclude = _resolve_pattern(exclude)

        # If overwrite is True, delete the contents of the bucket directory
        if overwrite:
            blobs = bucket.list_blobs(prefix=self.gcs_name)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS)
            for blob in blobs:
                executor.submit(lambda b: b.delete(), blob)
            executor.shutdown(wait=True)

        def upload(local_path: str, blob_path: str):
            blob = bucket.blob(blob_path)

            # Check if the blob already exists and has the same hash
            if blob.exists():
                blob.reload()  # Ensure that the blob's metadata is up-to-date
                if blob.crc32c == compute_crc32c(local_path):
                    pbar.update()
                    return

            # Upload the file
            logger.debug(f"Pushing {local_path} to gcs://{blob_path}")
            blob.upload_from_filename(local_path)
            blob.patch()  # Save metadata changes to GCS
            pbar.update()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS)
        pbar = tqdm(
            disable=not progress, desc=f"Push [{self.name}]", unit="files", ncols=0
        )

        # Walk through the local directory and upload each file
        count = 0
        for root, _, files in os.walk(self.local_path):
            for file in files:
                if file.endswith(".checksum"):
                    continue

                local_path = os.path.join(root, file)
                blob_path = os.path.join(
                    self.name, os.path.relpath(local_path, self.local_path)
                )

                if not _passes_filters(include, exclude, local_path):
                    continue

                executor.submit(upload, local_path, blob_path)
                count += 1

        pbar.total = count
        executor.shutdown(wait=True)
        pbar.close()

    @auth_required
    def pull(
        self,
        overwrite: bool = False,
        include: re.Pattern | str | None = None,
        exclude: re.Pattern | str | None = None,
        progress: bool = False,
    ):
        logger.info(f"Pulling data repo: {self.name}")
        bucket = _get_gcs_client().get_bucket(self.bucket)

        include = _resolve_pattern(include)
        exclude = _resolve_pattern(exclude)

        # If overwrite is True, delete the contents of the local directory
        if overwrite:
            shutil.rmtree(self.local_path)
        self.mkdir()

        def download(blob, local_path: str):
            blob.reload()
            if os.path.exists(local_path) and blob.crc32c == compute_crc32c(local_path):
                # print(f"Skipping {local_path}; no changes detected.")
                pbar.update()
                return

            local_dir_path = os.path.dirname(local_path)
            if not os.path.exists(local_dir_path):
                os.makedirs(local_dir_path)

            logger.debug(f"Pulling gcs://{blob.name} to {local_path}")
            blob.download_to_filename(local_path)
            with open(f"{local_path}.checksum", "w") as f:  # noqa: FURB103
                f.write(blob.crc32c)
            pbar.update()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS)
        pbar = tqdm(
            disable=not progress, desc=f"Pull [{self.name}]", unit=" files", ncols=0
        )

        # Walk through the bucket directory and download each file
        blobs = bucket.list_blobs(prefix=self.gcs_name)
        count = 0
        n_name = len(self.gcs_name)
        for blob in blobs:
            local_path = os.path.join(self.local_path, blob.name[n_name:])
            if local_path.endswith(".checksum"):
                # ???
                continue

            if not _passes_filters(include, exclude, local_path):
                continue

            executor.submit(download, blob, local_path)
            count += 1

        pbar.total = count
        executor.shutdown(wait=True)
        pbar.close()

    @auth_required
    def remote_exists(self) -> bool:
        bucket = Client().get_bucket(self.bucket)
        return any(True for _ in bucket.list_blobs(prefix=self.gcs_name))

    @model_validator(mode="after")
    def check_auth(self) -> Self:
        if self.validate_gcs_auth:
            self.remote_exists()  # Validate we can connect to GCS
        return self

    def local_exists(self) -> bool:
        return os.path.exists(self.local_path)


def compute_crc32c(path: str):
    checksum_path = f"{path}.checksum"
    if os.path.exists(checksum_path) and os.path.getmtime(
        checksum_path
    ) > os.path.getmtime(path):
        with open(checksum_path) as f:  # noqa: FURB101
            return f.read()
    else:
        if os.path.getsize(path) > 500 * 1024 * 1024:
            logger.info(f"Computing checksum of {path}...")
        with open(path, "rb") as f:  # noqa: FURB101
            data = f.read()
        crc32c = Checksum()
        crc32c.update(data)
        checksum = base64.b64encode(crc32c.digest()).decode("utf-8")
        with open(checksum_path, "w") as f:  # noqa: FURB103
            f.write(checksum)
        return checksum


def _resolve_pattern(pat: str | re.Pattern | None) -> re.Pattern | None:
    if isinstance(pat, str):
        try:
            pat = re.compile(pat)
        except re.error as e:
            raise ValueError(f'Invalid regex pattern "{pat}"') from e
    return pat


def _passes_filters(
    include: re.Pattern | None, exclude: re.Pattern | None, string: str
) -> bool:
    if include is not None and not include.match(string):
        return False
    return not (exclude is not None and exclude.match(string))


def _get_gcs_client() -> Client:
    # patch in a HTTPAdapter with a larger pool size
    # from https://stackoverflow.com/a/77740153
    client = Client()
    adapter = HTTPAdapter(pool_connections=MAX_THREADS, pool_maxsize=MAX_THREADS)
    client._http.mount("https://", adapter)
    client._http._auth_request.session.mount("https://", adapter)
    return client


async def download_file(
    client: httpx.AsyncClient,
    download_url: str,
    local_path: str | os.PathLike,
    file_name: str,
    headers: dict[str, str],
    timeout: float | None,
) -> None:
    """Download a single file.

    Args:
        client: httpx.AsyncClient
        download_url: URL to download.
        local_path: Local path to download file.
        file_name: Name of file to download.
        headers: Dictionary of headers.
        timeout: Timeout.

    """
    response = await client.get(download_url, headers=headers, timeout=timeout)
    response.raise_for_status()

    file_path = os.path.join(local_path, file_name)
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(response.content)
    print(f"Downloaded {file_path}")


async def download_github_subdirectory(
    client: httpx.AsyncClient,
    repo_owner: str,
    repo_name: str,
    branch: str,
    subdirectory: str,
    local_path: str | os.PathLike,
    timeout: float | None,
) -> None:
    """Download a specific subdirectory from a GitHub repository.

    Args:
        client: httpx.AsyncClient
        repo_owner: GitHub repository owner.
        repo_name: GitHub repository name.
        branch: GitHub branch.
        subdirectory: Subdirectory to download.
        local_path: Local path to download to.
        timeout: Timeout.
    """
    # Headers with the API version and authentication (optional)
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token " + os.environ["GITHUB_TOKEN"],
    }
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{subdirectory}?ref={branch}"
    response = await client.get(api_url, headers=headers, timeout=timeout)
    response.raise_for_status()  # Check for HTTP errors
    items = response.json()

    if not os.path.exists(local_path):
        os.makedirs(local_path)

    coroutines = []
    for item in items:
        if item["type"] == "file":
            coroutines.append(
                download_file(
                    client,
                    item["download_url"],
                    local_path,
                    item["name"],
                    headers,
                    timeout,
                )
            )
        elif item["type"] == "dir":
            new_subdir = os.path.join(subdirectory, item["name"])
            new_local_path = os.path.join(local_path, item["name"])
            coroutines.append(
                download_github_subdirectory(
                    client,
                    repo_owner,
                    repo_name,
                    branch,
                    new_subdir,
                    new_local_path,
                    timeout,
                )
            )

    await asyncio.gather(*coroutines)
