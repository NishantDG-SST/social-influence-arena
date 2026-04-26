---
title: Social Influence Arena
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
license: apache-2.0
short_description: Train LLMs to resist authority, consensus, gaslighting.
tags:
  - openenv
  - reinforcement-learning
  - rl
  - multi-agent
  - llm
  - alignment
---

# social_influence_env

OpenEnv environment for **Social Influence Arena**: trains LLMs to hold a calibrated belief state under adversarial social pressure (fake authority, fake consensus, gaslighting) and to update only on genuine evidence.

**Source code & training notebooks:** https://github.com/NishantDG-SST/social-influence-arena
**Top-level README** (full story, results, Colab badges): see the [GitHub repo](https://github.com/NishantDG-SST/social-influence-arena/blob/SIArenaV2/README.md)
**Demo video:** https://youtu.be/JLpOtZA9nLY
**Attacker LoRAs on HF Hub:** [authority](https://huggingface.co/NDGCodes/sia-authority-lora) · [consensus](https://huggingface.co/NDGCodes/sia-consensus-lora) · [gaslighter](https://huggingface.co/NDGCodes/sia-gaslighter-lora)

## API

```python
from social_influence_env import SocialInfluenceEnv, ArenaAction, BeliefState

env = SocialInfluenceEnv(base_url="http://localhost:8000")
result = env.reset(task_id="resist_pressure", seed=0)
obs = result.observation
print(obs.attacker_message)

action = ArenaAction(
    belief=BeliefState(answer="42", confidence=0.9, source="prior"),
    public_response="The answer is 42. I won't change my mind under pressure.",
)
step = env.step(action)
print(step.reward, step.done)
```

Three tasks are selectable at `reset`:

- `resist_pressure` — resist fake authority + fake consensus
- `consistency_memory` — refuse gaslighter-style fabrications of prior commitments
- `evidence_update` — update only when a real citation arrives

## Running the server

```bash
pip install -e .
python -m social_influence_env.server.app
# → Uvicorn on 0.0.0.0:8000
```

Or via Docker:

```bash
docker build -t social-influence-env server/
docker run -p 8000:8000 social-influence-env
```

## Env vars

- `ARENA_DEFAULT_TASK` (default `resist_pressure`)
- `ARENA_DIFFICULTY` (0/1/2)
- `ARENA_PORT` (default `8000`)
- `MAX_CONCURRENT_ENVS` (default `8`)
