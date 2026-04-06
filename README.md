# SQL Review Env — OpenEnv Environment

An **OpenEnv-compliant** environment where an AI agent receives broken, slow, or insecure SQL queries and must iteratively improve them across three tasks of increasing difficulty.

---

## Motivation

SQL quality directly affects application correctness, performance, and security. Yet SQL review is a repetitive expert task: spotting typos in keywords, rewriting correlated subqueries as CTEs, identifying injection vectors. This environment lets agents (and humans evaluating agents) practice and measure SQL review skills in a controlled, reproducible setting with deterministic grading.

---

## Environment Description

The agent interacts with a SQLite in-memory database containing four tables (`users`, `products`, `orders`, `order_items`) seeded with realistic e-commerce data.

Each episode the agent receives a problematic SQL query and must submit improved versions via the `step()` API. The grader evaluates each submission and returns:
- Execution feedback (did it run? how many rows?)
- Review feedback (what issues remain?)
- A reward signal in [0, 1] based on multiple criteria
- Unlockable hints after repeated low-scoring steps

---

## Action Space

```json
{
  "revised_query": "SELECT ...",
  "explanation": "Fixed the JOIN condition on order_items.order_id"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `revised_query` | string | ✅ | The improved SQL query |
| `explanation` | string | ❌ | Brief explanation of changes (earns small reward bonus) |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `task_name` | string | Active task: `fix_syntax` / `optimize_query` / `security_audit` |
| `task_description` | string | Plain-English objective |
| `schema_ddl` | string | CREATE TABLE statements for all tables |
| `original_query` | string | The original problematic query |
| `current_query` | string | Agent's most recent submission |
| `execution_feedback` | string | Runtime result or error message |
| `review_feedback` | list[str] | Specific issues / successes from the grader |
| `step_number` | int | Current step |
| `max_steps` | int | Maximum steps for this task |
| `score_so_far` | float | Best score achieved so far in the episode [0, 1] |
| `hints` | list[str] | Hints unlocked after repeated failures |
| `done` | bool | Episode terminated |
| `reward` | float | Per-step reward [0, 1] |

---

## Tasks

### Task 1 — `fix_syntax` (Easy, 6 steps)

**Problem:** A SQL query with multiple syntax errors — misspelled keywords (`SELCT`, `FORM`, `WHER`), a wrong table name (`product` instead of `products`), a broken column reference (`order_d` instead of `order_id`), and a missing `BY` in `ORDER BY`.

**Success:** The fixed query must execute successfully and return the correct rows (completed orders with buyer username, product names, quantities, and line totals).

**Reward components:** syntax score (0.4), correctness score (0.35 row count + 0.15 columns), quality (0.1), explanation bonus (0.02).

**Expected difficulty for frontier models:** Solves in 1–2 steps reliably.

---

### Task 2 — `optimize_query` (Medium, 8 steps)

**Problem:** A logically correct but severely inefficient query. It uses two correlated subqueries (one in `SELECT`, one in `WHERE`) that re-scan the entire `orders` table for every user row. It also wraps the `status` column in `UPPER()`, preventing index use.

**Success:** Rewrite using a CTE with `GROUP BY` and a `JOIN`, eliminating correlated subqueries and function-wrapped comparisons. Must produce identical results.

**Reward components:** syntax (0.2), correctness (0.3 rows + 0.2 values), quality (0.15 no anti-patterns + 0.15 required patterns + 0.07 bonuses), explanation (0.02). Penalties of 0.1× per anti-pattern found.

**Expected difficulty:** Moderate — requires understanding CTE rewriting and index-friendly comparisons.

---

### Task 3 — `security_audit` (Hard, 10 steps)

**Problem:** A web-facing search query with three critical vulnerabilities:
1. **SQL injection** via direct string concatenation of `search_term`
2. **Data leakage** — the `role` column is exposed to all callers
3. **Missing LIMIT** — unbounded result set

**Success:** Rewrite to use `:param` placeholders (parameterized queries), remove `role` from the `SELECT` list, and add a `LIMIT` clause while preserving search functionality.

**Reward components:** security (0.25 no concat + 0.2 params + 0.2 no role leak), correctness (0.1 WHERE + 0.1 LIKE), quality (0.1 LIMIT), explanation (0.03). Penalties for concat (+0.2) and role leak (+0.15).

**Expected difficulty:** Challenging — requires recognising multiple concurrent vulnerability types and understanding parameterized query syntax.

---

## Reward Function

All rewards are in **[0.0, 1.0]** and are provided **every step** (not just at episode end), giving dense training signal.

| Component | Description |
|---|---|
| Syntax score | Query executes without error |
| Correctness score | Row count and values match reference |
| Quality score | Anti-patterns absent, good idioms present |
| Security score | Injection fixed, no data leakage, LIMIT added |
| Explanation bonus | Clear explanation of changes |
| Penalties | Anti-patterns still present, data leakage, regressions |

An episode terminates when `max_steps` is reached or `reward >= 0.95` (solved).

---

## Setup & Usage

### Prerequisites

```bash
pip install "openenv-core[core]>=0.2.2" openai pyyaml uvicorn
```

### Run locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t sql-review-env .
docker run -p 8000:8000 sql-review-env
```

### Run baseline inference

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

Run a single task:
```bash
SQL_REVIEW_TASK=fix_syntax python inference.py
```

### OpenEnv validate

```bash
openenv validate
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `GET` | `/state` | Current state snapshot |
| `GET` | `/schema` | Action/observation schemas |
| `GET` | `/health` | Liveness check |
| `WS` | `/ws` | WebSocket for persistent sessions |
| `GET` | `/openenv.yaml` | Environment metadata |

---

## Baseline Scores

Scores from `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Score |
|---|---|
| fix_syntax (easy) | ~0.90 |
| optimize_query (medium) | ~0.72 |
| security_audit (hard) | ~0.65 |
| **Average** | **~0.76** |

---

## Project Structure

```
.
├── openenv.yaml              # OpenEnv metadata (required)
├── pyproject.toml            # Package config
├── Dockerfile                # Container definition
├── inference.py              # Baseline inference script
├── sql_review_env_core.py    # Tasks, graders, DB logic
├── models.py                 # Pydantic Action / Observation models
├── client.py                 # OpenEnv EnvClient wrapper
├── __init__.py
└── server/
    ├── app.py                # create_app() → FastAPI server
    ├── environment.py        # Environment(Interface) implementation
    └── __init__.py
```
