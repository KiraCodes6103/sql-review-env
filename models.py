"""
Data models for the SQL Review Env environment.

Uses openenv.core base types so the framework can auto-generate
/schema, /reset, /step, /state endpoints.
"""

from typing import List
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SqlReviewAction(Action):
    """Action submitted by the agent each step."""

    revised_query: str = Field(
        ...,
        description=(
            "The improved SQL query. For tasks 1 and 2 this must be a complete, "
            "runnable SQL statement. For task 3 (security_audit) show the safe "
            "version using :param placeholders."
        ),
    )
    explanation: str = Field(
        default="",
        description=(
            "Brief explanation of what was changed and why. "
            "Providing a good explanation earns a small reward bonus."
        ),
    )


class SqlReviewObservation(Observation):
    """What the agent sees after each step."""

    task_name: str = Field(description="Current task identifier")
    task_description: str = Field(description="Plain-English description of the objective")
    schema_ddl: str = Field(description="CREATE TABLE statements for all tables in scope")
    original_query: str = Field(description="The original (possibly broken) SQL query")
    current_query: str = Field(description="Agent's most recent submitted query")
    execution_feedback: str = Field(description="Results or error from running current_query")
    review_feedback: List[str] = Field(
        default_factory=list,
        description="List of specific issues / successes found by the grader",
    )
    step_number: int = Field(description="Current step within the episode")
    max_steps: int = Field(description="Maximum steps allowed for this task")
    score_so_far: float = Field(description="Best score achieved so far in this episode [0, 1]")
    hints: List[str] = Field(
        default_factory=list,
        description="Optional hints unlocked after repeated failures",
    )
