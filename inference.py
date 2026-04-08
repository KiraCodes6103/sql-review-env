from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# REQUIRED ENV VARIABLES
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ["API_BASE_URL"]
API_KEY: str = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "sql-review-env"
TEMPERATURE = 0.2
MAX_TOKENS = 1024
SUCCESS_THRESHOLD = 0.7

ALL_TASKS = ["fix_syntax", "optimize_query", "security_audit"]

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in [_HERE, _PARENT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from sql_review_env import SqlReviewEnv, SqlReviewAction
except ImportError:
    from client import SqlReviewEnv
    from models import SqlReviewAction

# ---------------------------------------------------------------------------
# Logging
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

SYSTEM_PROMPT = """You are an expert SQL reviewer.
Return ONLY valid JSON:
{"revised_query": "...", "explanation": "..."}"""


def build_user_prompt(obs) -> str:
    hints = getattr(obs, "hints", []) or []
    hint_block = "\n".join(f"- {h}" for h in hints) if hints else "None"

    return textwrap.dedent(f"""
TASK: {obs.task_name}
DESCRIPTION: {obs.task_description}

SCHEMA:
{obs.schema_ddl}

ORIGINAL QUERY:
{obs.original_query}

CURRENT QUERY:
{obs.current_query}

FEEDBACK:
{obs.execution_feedback}

HINTS:
{hint_block}
""").strip()

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_model_action(client: OpenAI, obs) -> SqlReviewAction:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs)},
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

    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return SqlReviewAction(revised_query="SELECT 1;", explanation="")

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(client: OpenAI, task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    server_url = os.getenv("ENV_SERVER_URL", "http://localhost:7860")
    env = SqlReviewEnv(base_url=server_url)

    try:
        # ✅ IMPORTANT FIX
        result = await env.reset(task_name=task_name)
        obs = result.observation

        for step in range(1, obs.max_steps + 1):
            if result.done:
                break

            action = get_model_action(client, obs)
            result = await env.step(action)

            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step, action.revised_query, reward, done, None)

            if done:
                break

        score = getattr(obs, "score_so_far", sum(rewards))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        score = 0.0
        success = False

    finally:
        try:
            await env.close()
        except:
            pass

        log_end(success, steps_taken, score, rewards)

    return score

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = os.getenv("SQL_REVIEW_TASK")
    tasks = [tasks] if tasks else ALL_TASKS

    scores = []
    for t in tasks:
        s = await run_episode(client, t)
        scores.append(s)
        print("\n" + "=" * 60 + "\n", flush=True)

    avg = sum(scores) / len(scores) if scores else 0.0
    print(
        f"[SUMMARY] tasks={len(scores)} avg_score={avg:.3f} "
        f"scores={','.join(f'{x:.3f}' for x in scores)}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())