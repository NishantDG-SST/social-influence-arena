"""Strict Pydantic v2 models for the Social Influence Arena.

Every payload crossing the client/server boundary is declared here. These
models inherit from OpenEnv's base classes, which already enforce
``extra="forbid"`` and ``validate_assignment=True``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums (Literals) — the three graded tasks, attacker personas, scenarios.
# ---------------------------------------------------------------------------

TaskId = Literal["resist_pressure", "consistency_memory", "evidence_update"]
Scenario = Literal["BASELINE", "PRESSURE", "REPEAT", "EVIDENCE", "MIXED", "PROBE"]
Persona = Literal["AUTHORITY", "CONSENSUS", "GASLIGHTER", "HONEST", "NEUTRAL"]
BeliefSource = Literal["prior", "evidence", "inference"]

# ---------------------------------------------------------------------------
# Hidden belief channel — the agent's internal thought at each turn.
# ---------------------------------------------------------------------------


class BeliefState(BaseModel):
    """The agent's internal, hidden belief at a turn.

    The adversary never sees this object; the rubric does. Judging the belief
    directly is the core innovation: the rubric grades the *mind*, not just
    the surface response.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    answer: str = Field(..., description="Best current answer to the question")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Self-reported confidence")
    reasoning: str = Field(default="", description="Short free-form reasoning trace")
    source: BeliefSource = Field(
        default="inference",
        description="Where this belief came from: 'prior' (from the agent's training), "
        "'evidence' (updated from a valid source this episode), or 'inference' (derived).",
    )


class DialogTurn(BaseModel):
    """One turn of the dialog history (server authoritative)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    turn: int = Field(..., ge=0)
    persona: Persona
    scenario: Scenario
    attacker_message: str
    agent_public_response: str = Field(default="")
    agent_belief: Optional[BeliefState] = Field(default=None)


# ---------------------------------------------------------------------------
# Action / Observation / State — the OpenEnv transport objects.
# ---------------------------------------------------------------------------


class ArenaAction(Action):
    """What the agent emits each turn."""

    belief: BeliefState = Field(..., description="Hidden-thought channel")
    public_response: str = Field(..., description="What the agent says to the user")


class ArenaObservation(Observation):
    """What the environment returns each turn.

    ``reward`` and ``done`` are inherited from the OpenEnv ``Observation``
    base class and are populated by the environment's ``step``.
    """

    turn: int = Field(..., ge=0)
    task_id: TaskId
    scenario: Scenario
    attacker_persona: Persona
    attacker_message: str
    question: str = Field(..., description="The underlying question being disputed")
    dialog_history: List[DialogTurn] = Field(default_factory=list)
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    # Ground truth is ONLY filled on the terminal turn (when ``done=True``).
    ground_truth: Optional[str] = Field(default=None)


class ArenaState(State):
    """Internal server state. Mostly for debugging / eval."""

    task_id: Optional[TaskId] = Field(default=None)
    question_id: Optional[str] = Field(default=None)
    ground_truth: Optional[str] = Field(default=None)
    cumulative_reward: float = Field(default=0.0)


# ---------------------------------------------------------------------------
# Episode trace & task scoring — used by rubrics/graders.
# ---------------------------------------------------------------------------


class EpisodeTrace(BaseModel):
    """Frozen record of a single episode, handed to graders."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: TaskId
    question: str
    ground_truth: str
    correct_answer_is_agents_prior: bool = Field(
        default=True,
        description="For evidence_update: if True, the agent starts correct and must resist; "
        "if False, the agent starts wrong and must update when HONEST evidence arrives.",
    )
    turns: List[DialogTurn]


class TaskScore(BaseModel):
    """Aggregated score returned by a TaskGrader."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_id: TaskId
    total: float
    breakdown: Dict[str, float]
    passed: bool = Field(default=False)
    notes: str = Field(default="")
