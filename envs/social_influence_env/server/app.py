"""FastAPI server entrypoint for Social Influence Arena."""

from __future__ import annotations

import os
from functools import partial

from openenv.core.env_server.http_server import create_app

from social_influence_env.models import ArenaAction, ArenaObservation
from social_influence_env.server.arena_env import SocialInfluenceEnvironment

_DEFAULT_TASK = os.environ.get("ARENA_DEFAULT_TASK", "resist_pressure")
_DIFFICULTY = int(os.environ.get("ARENA_DIFFICULTY", "1"))
_PORT = int(os.environ.get("ARENA_PORT", "8000"))
_MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    partial(
        SocialInfluenceEnvironment,
        default_task=_DEFAULT_TASK,
        difficulty=_DIFFICULTY,
    ),
    ArenaAction,
    ArenaObservation,
    env_name="social_influence_env",
    max_concurrent_envs=_MAX_CONCURRENT,
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=_PORT)


if __name__ == "__main__":
    main()
