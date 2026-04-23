"""Typed client for the Social Influence Arena."""

from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import ArenaAction, ArenaObservation, ArenaState, BeliefState, DialogTurn


class SocialInfluenceEnv(EnvClient[ArenaAction, ArenaObservation, ArenaState]):
    """HTTP client. Use ``reset(task_id=...)`` to select a task per episode."""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> StepResult[ArenaObservation]:
        kwargs: Dict[str, Any] = {}
        if task_id is not None:
            kwargs["task_id"] = task_id
        if domain is not None:
            kwargs["domain"] = domain
        return super().reset(seed=seed, episode_id=episode_id, **kwargs)

    # -----------------------------------------------------------------
    # Payload / response shaping
    # -----------------------------------------------------------------

    def _step_payload(self, action: ArenaAction) -> Dict[str, Any]:
        return {
            "belief": action.belief.model_dump(),
            "public_response": action.public_response,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ArenaObservation]:
        obs_data = payload.get("observation", {})
        observation = ArenaObservation(
            turn=obs_data.get("turn", 0),
            task_id=obs_data.get("task_id"),
            scenario=obs_data.get("scenario", "BASELINE"),
            attacker_persona=obs_data.get("attacker_persona", "NEUTRAL"),
            attacker_message=obs_data.get("attacker_message", ""),
            question=obs_data.get("question", ""),
            dialog_history=[
                DialogTurn(**t) for t in obs_data.get("dialog_history", [])
            ],
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            ground_truth=obs_data.get("ground_truth"),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ArenaState:
        return ArenaState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id"),
            question_id=payload.get("question_id"),
            ground_truth=payload.get("ground_truth"),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )


__all__ = ["SocialInfluenceEnv"]
