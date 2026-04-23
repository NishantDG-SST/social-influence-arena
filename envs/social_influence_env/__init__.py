"""Social Influence Arena — an OpenEnv environment that trains LLMs to
maintain a calibrated belief state under adversarial social pressure
(fake authority, fake consensus, gaslighting) and to update their beliefs
only when genuine evidence arrives.

Three production-mapped tasks live inside one environment:
- `resist_pressure`    → customer-support agent under fake escalations
- `consistency_memory` → content moderator being gaslit about prior rulings
- `evidence_update`    → decision-support system receiving a real override
"""

from .client import SocialInfluenceEnv
from .models import (
    ArenaAction,
    ArenaObservation,
    ArenaState,
    BeliefState,
    DialogTurn,
    EpisodeTrace,
    TaskScore,
)

__all__ = [
    "SocialInfluenceEnv",
    "ArenaAction",
    "ArenaObservation",
    "ArenaState",
    "BeliefState",
    "DialogTurn",
    "EpisodeTrace",
    "TaskScore",
]
