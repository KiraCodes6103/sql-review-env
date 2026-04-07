"""
Inference Script — SQL Review Environment
==========================================

Runs an LLM agent against all three tasks and prints OpenEnv-compliant
structured logs in the required [START] / [STEP] / [END] format.

Environment variables (injected by hackathon validator):
    API_BASE_URL        LiteLLM proxy endpoint  — REQUIRED, no default
    API_KEY             Proxy API key            — REQUIRED, no default
    MODEL_NAME          Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    LOCAL_IMAGE_NAME    Docker image name (optional — if using from_docker_image)
    HF_TOKEN            Alias for API_KEY (accepted as fallback locally)

Output format (strictly required by hackathon validators):
    [START] task=<task> env=sql-review-env model=<model>
    [STEP]  step=<n> action=<sql_preview> reward=<r> done=<bool> error=<msg|null>
    [END]   success=<bool> steps=<n> score=<s> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
#
# IMPORTANT: API_BASE_URL and API_KEY are read from the environment with NO
# default fallback. The hackathon validator injects these at runtime and
# checks that all LLM calls go through their LiteLLM proxy. Any hardcoded
# URL or key will cause the "No API calls made through our proxy" failure.
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ["API_BASE_URL"]          # REQUIRED — no default
API_KEY: str = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")  # validator injects API_KEY
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "sql-review-env"
TEMPERATURE = 0.2
MAX_TOKENS = 1024
SUCCESS_THRESHOLD = 0.7

ALL_TASKS = ["fix_syntax", "optimize_query", "security_audit"]

# ---------------------------------------------------------------------------
# Path setup — must happen before local imports
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in [_HERE, _PARENT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# OpenEnv client imports
# ---------------------------------------------------------------------------

try:
    from sql_review_env import SqlReviewEnv, SqlReviewAction
except ImportError:
    try:
        from client import SqlReviewEnv
        from models import SqlReviewAction
    except ImportError as _e:
        raise ImportError(
            "Cannot import SqlReviewEnv. Run from inside the project folder "
            "with openenv-core installed:\n"
            "  pip install 'openenv-core[core]>=0.2.2'\n"
            f"Original error: {_e}"
        ) from _e

# ---------------------------------------------------------------------------
# Structured logging — EXACT format required by hackathon validators
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    action_oneline = " ".join(action.split())[:120]
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_oneline!r} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert SQL reviewer. Your job is to fix, optimize, or secure a SQL query.

For each step you will receive:
- A task description and the database schema
- The original broken/slow/insecure query
- Your most recent submission and the feedback on it
- Optional hints if you are struggling

You MUST respond with a JSON object with exactly two fields:
{
  "revised_query": "<your improved SQL here>",
  "explanation": "<brief explanation of what you changed and why>"
}

Rules:
- Output ONLY valid JSON — no markdown, no backticks, no extra text.
- revised_query must be a complete, runnable SQL statement (tasks 1 and 2).
- For task 3 (security_audit), use :param placeholders — no string concatenation.
- Read the review feedback carefully and fix the specific issues mentioned.
""").strip()


def build_user_prompt(obs) -> str:
    hints = getattr(obs, "hints", []) or []
    hint_block = ""
    if hints:
        hint_block = "\nHINTS:\n" + "\n".join(f"  - {h}" for h in hints)
    review_feedback = getattr(obs, "review_feedback", []) or []
    review = "\n".join(f"  * {f}" for f in review_feedback)

    return textwrap.dedent(f"""
TASK: {obs.task_name} (step {obs.step_number}/{obs.max_steps})
OBJECTIVE: {obs.task_description}

SCHEMA:
{obs.schema_ddl}

ORIGINAL QUERY:
{obs.original_query}

YOUR LAST SUBMISSION:
{obs.current_query}

EXECUTION RESULT:
{obs.execution_feedback}

REVIEW FEEDBACK:
{review}

CURRENT SCORE: {obs.score_so_far:.2f}
{hint_block}

Respond with JSON only.
""").strip()


# ---------------------------------------------------------------------------
# LLM call — ALL calls go through the OpenAI client initialised with
# API_BASE_URL and API_KEY from environment. No other provider is used.
# ---------------------------------------------------------------------------

def get_model_action(client: OpenAI, obs) -> SqlReviewAction:
    """Call the LLM via the proxy; return a SqlReviewAction."""
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return SqlReviewAction(
            revised_query=data.get("revised_query", "SELECT 1;"),
            explanation=data.get("explanation", ""),
        )
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return SqlReviewAction(revised_query="SELECT 1;", explanation="")


# ---------------------------------------------------------------------------
# Async episode runner
# ---------------------------------------------------------------------------

async def run_episode(client: OpenAI, task_name: str,
                      base_url: str = "http://localhost:8000") -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    if LOCAL_IMAGE_NAME:
        env = await SqlReviewEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = SqlReviewEnv(base_url=base_url)

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, obs.max_steps + 1):
            if result.done:
                break

            action = get_model_action(client, obs)

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action.revised_query,
                     reward=reward, done=done, error=error)

            if done:
                break

        score = getattr(obs, "score_so_far", max(rewards) if rewards else 0.0)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score = 0.0
        success = False

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    # OpenAI client MUST be initialised with the injected API_BASE_URL and
    # API_KEY so all LLM traffic goes through the hackathon's LiteLLM proxy.
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    single_task = os.getenv("SQL_REVIEW_TASK")
    tasks_to_run = [single_task] if single_task else ALL_TASKS

    server_url = os.getenv("ENV_SERVER_URL", "http://localhost:8000")

    all_scores: List[float] = []
    for task_name in tasks_to_run:
        score = await run_episode(client, task_name, base_url=server_url)
        all_scores.append(score)
        print(f"\n{'=' * 60}\n", flush=True)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(
        f"[SUMMARY] tasks={len(all_scores)} avg_score={avg:.3f} "
        f"scores={','.join(f'{s:.3f}' for s in all_scores)}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
