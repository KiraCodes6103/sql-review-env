"""
FastAPI application for the SQL Review Env environment.

Uses openenv.core.env_server.http_server.create_app() which automatically
generates all required OpenEnv endpoints:
  POST /reset, POST /step, GET /state, GET /schema, WS /ws, GET /health

Task selection: pass ?task_name=<name> as a query parameter to /reset.
Supported tasks: fix_syntax (easy), optimize_query (medium), security_audit (hard)

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path so sql_review_env_core is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install 'openenv-core[core]'"
    ) from e

try:
    from ..models import SqlReviewAction, SqlReviewObservation
    from .environment import SqlReviewEnvironment
except (ImportError, ModuleNotFoundError):
    from models import SqlReviewAction, SqlReviewObservation
    from server.environment import SqlReviewEnvironment


app = create_app(
    SqlReviewEnvironment,
    SqlReviewAction,
    SqlReviewObservation,
    env_name="sql_review_env",
    max_concurrent_envs=4,
)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
