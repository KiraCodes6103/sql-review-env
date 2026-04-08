"""
Inference Script — SQL Review Environment
==========================================

Runs an LLM agent against all three tasks and prints structured logs.

The hackathon validator injects:
    API_BASE_URL   — LiteLLM proxy endpoint (REQUIRED)
    API_KEY        — proxy key              (REQUIRED)
    MODEL_NAME     — model to use           (optional, has default)

The OpenAI client is initialised ONLY from these variables so every LLM
call goes through the validator's proxy. No other provider is used.

Output format:
    [START] task=<task> env=sql-review-env model=<model>
    [STEP]  step=<n> action=<sql_preview> reward=<r> done=<bool> error=<msg|null>
    [END]   success=<bool> steps=<n> score=<s> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — injected by the hackathon validator at runtime
# API_BASE_URL and API_KEY have NO defaults so the validator's values are
# always used and LLM calls always go through the LiteLLM proxy.
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ["API_BASE_URL"]   # validator injects this — no default
API_KEY: str      = os.environ["API_KEY"]         # validator injects this — no default
MODEL_NAME: str   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK         = "sql-review-env"
TEMPERATURE       = 0.2
MAX_TOKENS        = 1024
SUCCESS_THRESHOLD = 0.7

ALL_TASKS = ["fix_syntax", "optimize_query", "security_audit"]

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Import environment logic directly — no HTTP server required
# ---------------------------------------------------------------------------

from sql_review_env_core import TASKS, GRADERS, HINTS_MAP, SCHEMA_DDL, make_db, run_query


class _Env:
    """Runs the SQL review graders in-process. No server or Docker needed."""

    def __init__(self, task_name: str):
        self._task_name = task_name
        self._task      = TASKS[task_name]
        self._conn      = None
        self._step_n    = 0
        self._query     = ""
        self._best      = 0.0
        self._hint_budget = 2

    def reset(self) -> dict:
        if self._conn:
            self._conn.close()
        self._conn    = make_db()
        self._step_n  = 0
        self._query   = self._task["original_query"]
        self._best    = 0.0
        self._hint_budget = 2
        return self._obs("No query submitted yet.", [], [], False, 0.0)

    def step(self, revised_query: str, explanation: str = "") -> dict:
        self._step_n += 1
        self._query   = revised_query.strip()

        result = GRADERS[self._task_name](
            revised_query=revised_query,
            explanation=explanation,
            task=self._task,
            conn=self._conn,
            step=self._step_n,
            max_steps=self._task["max_steps"],
        )
        reward      = result["total"]
        self._best  = max(self._best, reward)

        rows, err = run_query(self._conn, self._query)
        exec_fb   = f"ERROR: {err}" if err else f"OK — {len(rows)} rows. Preview: {rows[:2]}"

        review_fb = result["breakdown"].split("; ")

        hints: List[str] = []
        if reward < 0.3 and self._hint_budget > 0 and self._step_n >= 2:
            pool  = HINTS_MAP.get(self._task_name, [])
            hints = pool[: self._task["max_steps"] - self._hint_budget + 1]
            self._hint_budget -= 1

        done = self._step_n >= self._task["max_steps"] or reward >= 0.95
        return self._obs(exec_fb, review_fb, hints, done, reward)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _obs(self, exec_fb, review_fb, hints, done, reward) -> dict:
        return {
            "task_name":          self._task_name,
            "task_description":   self._task["description"],
            "schema_ddl":         SCHEMA_DDL.strip(),
            "original_query":     self._task["original_query"],
            "current_query":      self._query,
            "execution_feedback": exec_fb,
            "review_feedback":    review_fb,
            "step_number":        self._step_n,
            "max_steps":          self._task["max_steps"],
            "score_so_far":       self._best,
            "hints":              hints,
            "done":               done,
            "reward":             reward,
        }


# ---------------------------------------------------------------------------
# Structured logging
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
You are an expert SQL reviewer. Fix, optimize, or secure the given SQL query.

Respond with a JSON object with exactly two fields:
{
  "revised_query": "<your improved SQL>",
  "explanation":   "<brief explanation of what you changed>"
}

Rules:
- Output ONLY valid JSON — no markdown, no backticks, no extra text.
- For tasks 1 and 2: revised_query must be a complete runnable SQL statement.
- For task 3 (security_audit): use :param placeholders, remove the role column, add LIMIT.
""").strip()


def build_prompt(obs: dict) -> str:
    hints = obs.get("hints") or []
    hint_block = ("\nHINTS:\n" + "\n".join(f"  - {h}" for h in hints)) if hints else ""
    review = "\n".join(f"  * {f}" for f in (obs.get("review_feedback") or []))
    return textwrap.dedent(f"""
TASK: {obs['task_name']} (step {obs['step_number']}/{obs['max_steps']})
OBJECTIVE: {obs['task_description']}

SCHEMA:
{obs['schema_ddl']}

ORIGINAL QUERY:
{obs['original_query']}

YOUR LAST SUBMISSION:
{obs['current_query']}

EXECUTION RESULT:
{obs['execution_feedback']}

REVIEW FEEDBACK:
{review}

SCORE SO FAR: {obs['score_so_far']:.2f}
{hint_block}

Respond with JSON only.
""").strip()


# ---------------------------------------------------------------------------
# LLM call — uses the OpenAI client pointed at the validator's proxy
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, obs: dict) -> tuple[str, str]:
    """Returns (revised_query, explanation). Falls back to SELECT 1 on error."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw  = (completion.choices[0].message.content or "").strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return data.get("revised_query", "SELECT 1;"), data.get("explanation", "")
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "SELECT 1;", ""


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = _Env(task_name)
    try:
        obs = env.reset()

        while not obs["done"]:
            revised_query, explanation = call_llm(client, obs)
            obs = env.step(revised_query, explanation)

            reward      = obs["reward"]
            done        = obs["done"]
            steps_taken = obs["step_number"]
            rewards.append(reward)

            log_step(step=steps_taken, action=revised_query,
                     reward=reward, done=done, error=None)

            if done:
                break

        score   = min(max(obs["score_so_far"], 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Initialise the OpenAI client exclusively from the injected env vars.
    # This guarantees every chat.completions.create() call hits the proxy.
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    tasks_to_run = [os.getenv("SQL_REVIEW_TASK")] if os.getenv("SQL_REVIEW_TASK") else ALL_TASKS

    all_scores: List[float] = []
    for task_name in tasks_to_run:
        score = run_episode(client, task_name)
        all_scores.append(score)
        print(f"\n{'=' * 60}\n", flush=True)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(
        f"[SUMMARY] tasks={len(all_scores)} avg_score={avg:.3f} "
        f"scores={','.join(f'{s:.3f}' for s in all_scores)}",
        flush=True,
    )


if __name__ == "__main__":
    main()