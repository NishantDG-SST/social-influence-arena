"""Generate DPO preference pairs by sampling K trajectories per episode.

For each episode we sample ``--k`` rollouts with the policy at non-zero
temperature, score each through the rubric, and emit the best/worst pair
as a single DPO example::

    {"prompt": "...", "chosen": "...", "rejected": "..."}

Output is JSONL at ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List
from uuid import uuid4

# Make the env package importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "envs"))

from social_influence_env.client import SocialInfluenceEnv  # noqa: E402
from social_influence_env.inference import (  # noqa: E402
    SYSTEM_PROMPT,
    _build_policy,
    _format_assistant,
)
from social_influence_env.models import ArenaAction, DialogTurn  # noqa: E402


def rollout_once(env, task_id: str, policy, seed: int) -> tuple[list[dict], float, list[DialogTurn]]:
    """Return (turn-level records, total_reward, final history)."""
    result = env.reset(task_id=task_id, seed=seed)
    obs = result.observation
    history: List[DialogTurn] = []
    transcript: list[dict] = []
    total = 0.0
    while True:
        obs_dict = obs.model_dump()
        belief, public = policy(obs_dict, history)
        transcript.append({
            "attacker_message": obs.attacker_message,
            "belief": belief.model_dump(),
            "public_response": public,
        })
        step = env.step(ArenaAction(belief=belief, public_response=public))
        next_obs = step.observation
        history = list(next_obs.dialog_history)
        obs = next_obs
        if step.done:
            total = float(step.reward or 0.0)
            break
    return transcript, total, history


def _chatml(messages: list[dict]) -> str:
    """Simple serialization of a chat transcript for DPO pairs."""
    parts = [f"<|system|>\n{SYSTEM_PROMPT}"]
    for m in messages:
        parts.append(f"<|{m['role']}|>\n{m['content']}")
    return "\n".join(parts)


def build_pair(transcript: list[dict], history: list[DialogTurn]) -> tuple[str, str]:
    """Return (prompt_prefix, assistant_completion) for the last turn."""
    messages = []
    for turn in history[:-1]:
        messages.append({"role": "user", "content": turn.attacker_message})
        messages.append({"role": "assistant", "content": _format_assistant(turn)})
    last = history[-1]
    messages.append({"role": "user", "content": last.attacker_message})
    prompt = _chatml(messages)
    completion = f"<belief>{json.dumps(last.agent_belief.model_dump())}</belief>\n<public>{last.agent_public_response}</public>"
    return prompt, completion


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model-id", default="scripted://always_truthful")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--k", type=int, default=4, help="Number of samples per episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--task",
        nargs="+",
        choices=["resist_pressure", "consistency_memory", "evidence_update"],
        default=["resist_pressure", "consistency_memory", "evidence_update"],
    )
    p.add_argument("--out", type=Path, default=Path("runs/dpo_pairs.jsonl"))
    args = p.parse_args(argv)

    env = SocialInfluenceEnv(base_url=args.base_url)
    policy = _build_policy(args.model_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.out.open("w") as fh:
        for ep in range(args.episodes):
            for task in args.task:
                scored: list[tuple[float, list[DialogTurn]]] = []
                for k in range(args.k):
                    seed = args.seed + ep * 1000 + k  # k differs -> diversity
                    _, total, history = rollout_once(env, task, policy, seed)
                    scored.append((total, history))
                scored.sort(key=lambda x: x[0])
                worst_hist = scored[0][1]
                best_hist = scored[-1][1]
                if scored[0][0] == scored[-1][0]:
                    continue  # flat — skip
                prompt_best, chosen = build_pair([], best_hist)
                prompt_worst, rejected = build_pair([], worst_hist)
                # For DPO we need matched prompts; best/worst share the same
                # attacker schedule but may have diverged. Use the best
                # trajectory's prompt as the reference.
                fh.write(json.dumps({
                    "id": str(uuid4()),
                    "task_id": task,
                    "prompt": prompt_best,
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_reward": scored[-1][0],
                    "rejected_reward": scored[0][0],
                }) + "\n")
                n_written += 1
    print(f"Wrote {n_written} DPO pairs to {args.out}")


if __name__ == "__main__":
    main()
