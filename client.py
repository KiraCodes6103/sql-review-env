"""SQL Review Env Client — wraps openenv.core.EnvClient."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SqlReviewAction, SqlReviewObservation
except ImportError:
    from models import SqlReviewAction, SqlReviewObservation


class SqlReviewEnv(EnvClient[SqlReviewAction, SqlReviewObservation, State]):
    """
    Client for the SQL Review Env environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example (server already running):
        with SqlReviewEnv(base_url="http://localhost:8000") as client:
            result = client.reset()
            obs = result.observation
            while not result.done:
                action = SqlReviewAction(
                    revised_query="SELECT ...",
                    explanation="Fixed the JOIN condition",
                )
                result = client.step(action)

    Example (Docker):
        client = SqlReviewEnv.from_docker_image("sql-review-env:latest")
        try:
            result = client.reset()
            ...
        finally:
            client.close()
    """

    def _step_payload(self, action: SqlReviewAction) -> Dict:
        return {
            "revised_query": action.revised_query,
            "explanation": action.explanation,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SqlReviewObservation]:
        obs_data = payload.get("observation", {})
        observation = SqlReviewObservation(
            task_name=obs_data.get("task_name", ""),
            task_description=obs_data.get("task_description", ""),
            schema_ddl=obs_data.get("schema_ddl", ""),
            original_query=obs_data.get("original_query", ""),
            current_query=obs_data.get("current_query", ""),
            execution_feedback=obs_data.get("execution_feedback", ""),
            review_feedback=obs_data.get("review_feedback", []),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 10),
            score_so_far=obs_data.get("score_so_far", 0.0),
            hints=obs_data.get("hints", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )