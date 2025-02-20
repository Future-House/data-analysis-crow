import argparse
import asyncio
import inspect
import logging
import os
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from pydoc import locate
from typing import Any, cast
from uuid import UUID

import cloudpickle
import dotenv
from aviary.core import Environment
from aviary.functional import EnvironmentBuilder
from aviary.message import Message
from aviary.tools.base import Tool, ToolRequestMessage
from crow_client import CrowJobClient
from crow_client.models import Stage, Step
from crow_client.models.client import (
    ASVState,
    BeforeTransitionState,
    EnvResetState,
    EnvStepState,
    InitialState,
    TransitionState,
)
from google.auth import default as google_default_auth_handler
from google.auth.transport import requests
from google.oauth2.credentials import Credentials
from google.oauth2.id_token import fetch_id_token
from ldp.agent import Agent
from ldp.alg import Callback, RolloutManager
from ldp.data_structures import Transition
from ldp.graph.ops import OpResult
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

import os
os.chdir("src/app/")


class Crow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str | type[Agent] | None = None
    environment: str | type[Environment] | Callable[..., Environment] | None = None

    def get_environment(self, query: str) -> Environment:
        logger.debug(f"Getting environment for query: {query}")
        if self.environment is None or isinstance(self.environment, str):
            logger.error("Environment not properly initialized")
            raise ValueError("Environment must be provided")
        if inspect.isclass(self.environment):
            if hasattr(self.environment, "from_task"):
                logger.debug("Creating environment using from_task method")
                return self.environment.from_task(query)
            logger.debug("Creating environment using default constructor")
            return self.environment()
        if callable(self.environment):
            logger.debug("Creating environment using callable")
            return self.environment(query)  # type: ignore[call-arg]
        raise TypeError(
            "Environment must be a callable or a class with 'from_task' method"
        )

    def get_agent(self) -> Agent:
        logger.debug("Getting agent instance")
        if self.agent is None or isinstance(self.agent, str):
            logger.error("Agent not properly initialized")
            raise ValueError("Agent must be provided")
        try:
            return self.agent()
        except Exception:
            logger.exception("Failed to create agent instance")
            raise

    @field_validator("agent", mode="before")
    @classmethod
    def validate_agent(cls, v: str | type[Agent] | None) -> type[Agent]:
        logger.debug(f"Validating agent: {v}")
        if v is None:
            v = os.getenv("CROW_AGENT")
            logger.debug(f"Using CROW_AGENT from environment: {v}")

        if isinstance(v, str):
            try:
                v = cast(type[Agent] | None, locate(v))
                logger.debug(f"Located agent class: {v}")
            except Exception:
                logger.exception(f"Failed to locate agent class: {v}")
                raise

        if v is None:
            logger.error("Agent validation failed: no agent found")
            raise ValueError("Agent can not be found.")

        return v

    @model_validator(mode="after")
    @classmethod
    def validate_fields(cls, values):
        logger.debug("Validating Crow fields")
        try:
            values.agent = cls.validate_agent(values.agent)
            values.environment = cls.validate_environment(values.environment)
        except Exception:
            logger.exception("Field validation failed on agent or environment")
            raise
        return values

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(
        cls, v: str | type[Environment] | None
    ) -> type[Environment] | EnvironmentBuilder:
        logger.debug(f"Validating environment: {v}")
        if v is None:
            v = os.getenv("CROW_ENVIRONMENT")
            logger.debug(f"Using CROW_ENVIRONMENT from environment: {v}")

        if v is None:
            logger.error("No environment specified and CROW_ENVIRONMENT not set")
            raise ValueError(
                "environment is None and CROW_ENVIRONMENT env variable is not set"
            )

        if isinstance(v, str):
            env_path = f"{v}/environment.pkl"
            if os.path.isfile(env_path):
                logger.debug(f"Loading environment from pickle: {env_path}")
                try:
                    with open(env_path, "rb") as f:
                        environment = cloudpickle.load(f)
                except Exception:
                    logger.exception(f"Failed to load environment from {env_path}")
                    raise

                if not isinstance(environment, EnvironmentBuilder):
                    logger.error(f"Invalid environment type loaded from {env_path}")
                    raise ValueError(
                        f"Loaded object from {v} is not an instance of EnvironmentBuilder."
                    )
            else:
                logger.debug(f"Attempting to locate environment: {v}")
                environment = locate(v)
                if environment is None:
                    logger.error(f"Could not locate environment: {v}")
                    raise ValueError(f"Environment can not be found: {v}")
                if not (
                    (
                        inspect.isclass(environment)
                        and issubclass(environment, Environment)
                    )
                    or isinstance(environment, EnvironmentBuilder)
                ):
                    logger.error(f"Invalid environment type: {type(environment)}")
                    raise ValueError(
                        f"Resolved environment is not a valid Environment subclass: {environment}"
                    )
        else:
            environment = v

        if inspect.isclass(environment) and issubclass(environment, Environment):
            return environment
        if callable(environment):
            return environment
        error_msg = (
            "Environment must be either a subclass of Environment or a callable "
            "that returns an Environment instance."
        )
        logger.error(f"Invalid environment type: {type(environment)}. {error_msg} ")
        raise ValueError(error_msg)


class AgentStatus(StrEnum):
    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    FAIL = "fail"
    SUCCESS = "success"
    TRUNCATED = "truncated"


class CrowCallback(Callback):
    def __init__(
        self,
        query: str,
        agent: str,
        environment: str,
        stage: Stage = Stage.LOCAL,
        auth_token: str | None = None,
        trajectory_id: str | UUID | None = None,
    ):
        logger.info(
            f"Initializing CrowCallback for agent {agent}, environment {environment}"
        )
        self.agent = agent
        self.environment = environment
        self.query = query
        self.auth_token = auth_token

        try:
            if auth_token is None:
                logger.debug("No auth token provided, fetching from credentials")
                credentials, _ = google_default_auth_handler()
                auth_req = requests.Request()
                if isinstance(credentials, Credentials):
                    logger.debug("Using user account credentials")
                    credentials.refresh(auth_req)
                    self.auth_token = credentials.id_token
                else:
                    logger.debug(f"Using service account credentials for stage {stage}")
                    self.auth_token = fetch_id_token(auth_req, stage)

            if self.auth_token is None:
                raise ValueError("Failed to fetch auth token")

            self.client = CrowJobClient(
                base_uri=stage,
                agent=self.agent,
                auth_token=self.auth_token,
                environment=self.environment,
                trajectory_id=trajectory_id,
            )
            logger.debug("Successfully initialized CrowJobClient")
        except Exception:
            logger.exception("Failed to initialize CrowCallback")
            raise

    async def finalize_environment(self, status: str):
        logger.info(f"Finalizing environment with status: {status}")
        try:
            await self.client.finalize_environment(status=status)
        except Exception:
            logger.exception("Failed to finalize environment")
            raise

    async def before_transition(
        self,
        traj_id: UUID | str,
        agent: Agent,
        env: Environment,
        agent_state: Any,
        obs: list[Message],
    ) -> None:
        logger.debug(f"Before transition for trajectory {traj_id}")
        try:
            state = BeforeTransitionState(current_state=agent_state, observations=obs)
            step = Step.BEFORE_TRANSITION.value
            await self.client.store_agent_state(step=step, state=state)
        except Exception:
            logger.exception("Failed in before_transition callback")
            raise

    async def after_agent_init_state(self, traj_id: str, init_state: Any) -> None:
        logger.info(f"After agent init state for trajectory {traj_id}")
        try:
            state = InitialState(initial_state=init_state)
            step = Step.AFTER_AGENT_INIT_STATE.value
            await self.client.store_agent_state(step=step, state=state)
        except Exception:
            logger.exception("Failed in after_agent_init_state callback")
            raise

    async def after_agent_get_asv(
        self,
        traj_id: str,
        action: OpResult[ToolRequestMessage],
        next_agent_state: Any,
        value: float,
    ) -> None:
        logger.debug(f"After agent get ASV for trajectory {traj_id}")
        try:
            state = cast(
                ASVState[ToolRequestMessage],
                ASVState(action=action, next_state=next_agent_state, value=value),
            )
            step = Step.AFTER_AGENT_GET_ASV.value
            await self.client.store_agent_state(step=step, state=state)
        except Exception:
            logger.exception("Failed in after_agent_get_asv callback")
            raise

    async def after_env_reset(
        self, traj_id: str, obs: list[Message], tools: list[Tool]
    ) -> None:
        logger.info(f"After environment reset for trajectory {traj_id}")
        try:
            state = EnvResetState(observations=obs, tools=tools)
            step = Step.AFTER_ENV_RESET.value
            await self.client.store_agent_state(step=step, state=state)
        except Exception:
            logger.exception("Failed in after_env_reset callback")
            raise

    async def after_env_step(
        self, traj_id: str, obs: list[Message], reward: float, done: bool, trunc: bool
    ) -> None:
        logger.debug(
            f"After environment step for trajectory {traj_id}, "
            f"reward: {reward}, done: {done}, truncated: {trunc}"
        )
        try:
            state = EnvStepState(
                observations=obs, reward=reward, done=done, trunc=trunc
            )
            step = Step.AFTER_ENV_STEP.value
            await self.client.store_agent_state(step=step, state=state)
        except Exception:
            logger.exception("Failed in after_env_step callback")
            raise

    async def after_transition(
        self, traj_id: str, agent: Agent, env: Environment, transition: Transition
    ) -> None:
        logger.info(f"After transition for trajectory {traj_id}")
        try:
            state = TransitionState(transition=transition)
            step = Step.AFTER_TRANSITION.value
            await self.client.store_agent_state(step=step, state=state)

            frame = env.export_frame()
            await self.client.store_environment_frame(frame)
        except Exception:
            logger.exception("Failed in after_transition callback")
            raise


async def crow_rollout(
    query: str,
    agent: str,
    environment: str,
    auth_token: str,
    trajectory_id: str | UUID | None = None,
    timeout: float | None = None,
    max_steps: int | None = None,
    stage: Stage = Stage.LOCAL,
) -> AgentStatus:
    logger.info(
        f"Starting crow rollout for query: {query}, "
        f"agent: {agent}, environment: {environment}"
    )
    try:
        job = Crow(agent=agent, environment=environment)
        agent_str = os.getenv("CROW_AGENT", agent)
        environment_str = os.getenv("CROW_ENVIRONMENT", environment)

        logger.debug("Initializing CrowCallback")
        crow_callback = CrowCallback(
            query=query,
            agent=agent_str,
            environment=environment_str,
            trajectory_id=trajectory_id,
            stage=stage,
            auth_token=auth_token,
        )

        try:
            logger.info(f"Starting rollout with timeout: {timeout}")
            async with asyncio.timeout(timeout):
                rollout_manager = RolloutManager(
                    job.get_agent(), callbacks=[crow_callback]
                )
                await rollout_manager.sample_trajectories(
                    environments=[job.get_environment(query)], max_steps=max_steps
                )
                status = AgentStatus.SUCCESS
                logger.info("Rollout completed successfully")
        except TimeoutError:
            logger.warning(f"Agent timeout after {timeout}-sec, truncating execution")
            status = AgentStatus.TRUNCATED
        except Exception:
            logger.exception("Rollout failed with unexpected error")
            status = AgentStatus.FAIL

        logger.info(f"Finalizing environment with status: {status}")
        await crow_callback.finalize_environment(status)
    except Exception:
        logger.exception("Critical error in crow_rollout")
        raise
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True, help="User ID for the job")
    parser.add_argument("--trajectory-id", help="Trajectory ID for the job")
    parser.add_argument(
        "--stage",
        help="Target deployment stage",
        choices=["localdocker", "local", "dev", "prod"],
        default="dev",
    )
    parser.add_argument("--query", required=True, help="Query to prompt the crow.")
    parser.add_argument("--agent", required=False, help="LDP agent module to use.")
    parser.add_argument(
        "--environment", required=False, help="Aviary environment to use."
    )
    parser.add_argument(
        "--timeout", default=None, help="Timeout (seconds) for the job rollout."
    )
    parser.add_argument(
        "--max_steps", default=None, help="Max steps for the job rollout."
    )
    parser.add_argument(
        "--auth-jwt",
        default=None,
        help="A firebase jwt token to use for auth. If None is provided, this fallsback to an application default credentials token.",
    )
    args = parser.parse_args()
    logger.info(f"Received args: {args}")
    stage = Stage.from_string(args.stage)
    status = asyncio.run(
        crow_rollout(
            args.query,
            trajectory_id=args.trajectory_id,
            agent=args.agent,
            environment=args.environment,
            timeout=args.timeout,
            max_steps=args.max_steps,
            stage=stage,
            auth_token=args.auth_jwt,
        )
    )
    logger.info(f"Final status: {status}")
    Path(str(args.trajectory_id)).mkdir(exist_ok=True)
    Path(f"{args.trajectory_id}/status.txt").write_text(status)


if __name__ == "__main__":
    main()
