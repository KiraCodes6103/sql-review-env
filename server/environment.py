"""
SQL Review Environment — OpenEnv server-side implementation.

Implements openenv.core.env_server.interfaces.Environment so that
openenv's create_app() can expose it over HTTP/WebSocket automatically.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SqlReviewAction, SqlReviewObservation
except ImportError:
    from models import SqlReviewAction, SqlReviewObservation

from sql_review_env_core import (
    GRADERS,
    HINTS_MAP,
    SCHEMA_DDL,
    TASKS,
    make_db,
    run_query,
)


class SqlReviewEnvironment(Environment):
    """
    SQL Query Review environment.

    An AI agent receives a broken, slow, or insecure SQL query and must
    iteratively improve it across up to max_steps turns.

    Three tasks of increasing difficulty:
      - fix_syntax   (easy):   fix spelling errors and wrong JOIN conditions
      - optimize_query (medium): replace correlated subqueries with a CTE
      - security_audit (hard):  fix injection, data leakage, and missing LIMIT

    Reward is per-step in [0, 1] based on syntax correctness, semantic
    correctness, query quality / security, and a small explanation bonus.
    Partial credit is always available — reward is never purely binary.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: str = "fix_syntax"):
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from {list(TASKS.keys())}"
            )
        self._task_name = task_name
        self._task = TASKS[task_name]
        self._conn: Optional[sqlite3.Connection] = None
        self._step_n = 0
        self._current_query = ""
        self._best_score = 0.0
        self._reward_history: List[float] = []
        self._hint_budget = 2
        self._state = State(episode_id=str(uuid4()), step_count=0)

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> SqlReviewObservation:
        """Reset environment to initial state and return the first observation."""
        if self._conn:
            self._conn.close()
        self._conn = make_db()
        self._step_n = 0
        self._current_query = self._task["original_query"]
        self._best_score = 0.0
        self._reward_history = []
        self._hint_budget = 2
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return self._build_obs(
            execution_feedback="No query submitted yet.",
            review_feedback=[
                "Start by reading the original query and fixing the issues described."
            ],
            done=False,
            reward=0.0,
        )

    def step(self, action: SqlReviewAction) -> SqlReviewObservation:  # type: ignore[override]
        """
        Submit a revised query.

        Returns an observation that includes execution feedback, grader
        review feedback, reward, and done flag embedded per openenv spec.
        """
        if self._conn is None:
            raise RuntimeError("Call reset() before step().")

        self._step_n += 1
        self._state.step_count = self._step_n
        self._current_query = action.revised_query.strip()

        grader = GRADERS[self._task_name]
        result = grader(
            revised_query=action.revised_query,
            explanation=action.explanation,
            task=self._task,
            conn=self._conn,
            step=self._step_n,
            max_steps=self._task["max_steps"],
        )

        step_reward: float = result["total"]
        self._reward_history.append(step_reward)
        self._best_score = max(self._best_score, step_reward)

        # Build execution feedback string
        rows, err = run_query(self._conn, self._current_query)
        if err:
            exec_feedback = f"ERROR: {err}"
        elif rows is not None:
            preview = rows[:3]
            exec_feedback = f"OK — {len(rows)} rows returned. Preview: {preview}"
        else:
            exec_feedback = "Query executed (no rows)."

        review_feedback: List[str] = result["breakdown"].split("; ")

        # Unlock hints after repeated low-scoring steps
        hints: List[str] = []
        if step_reward < 0.3 and self._hint_budget > 0 and self._step_n >= 2:
            task_hints = HINTS_MAP.get(self._task_name, [])
            used = self._task["max_steps"] - self._hint_budget
            hints = task_hints[: used + 1]
            self._hint_budget -= 1

        done = self._step_n >= self._task["max_steps"] or step_reward >= 0.95

        return self._build_obs(
            execution_feedback=exec_feedback,
            review_feedback=review_feedback,
            hints=hints,
            done=done,
            reward=step_reward,
        )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        execution_feedback: str,
        review_feedback: List[str],
        hints: Optional[List[str]] = None,
        done: bool = False,
        reward: float = 0.0,
    ) -> SqlReviewObservation:
        return SqlReviewObservation(
            task_name=self._task_name,
            task_description=self._task["description"],
            schema_ddl=SCHEMA_DDL.strip(),
            original_query=self._task["original_query"],
            current_query=self._current_query,
            execution_feedback=execution_feedback,
            review_feedback=review_feedback,
            step_number=self._step_n,
            max_steps=self._task["max_steps"],
            score_so_far=self._best_score,
            hints=hints or [],
            done=done,
            reward=reward,
            metadata={
                "best_score": self._best_score,
                "reward_history": list(self._reward_history),
            },
        )
