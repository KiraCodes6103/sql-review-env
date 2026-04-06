"""
SQL Review Environment — Core logic.

Schema, tasks, graders, and reward functions for the SQL Review OpenEnv.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Database schema & seed data
# ---------------------------------------------------------------------------

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    username  TEXT NOT NULL UNIQUE,
    email     TEXT NOT NULL,
    role      TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS products (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    price       REAL NOT NULL,
    category    TEXT NOT NULL,
    stock       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    status      TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS order_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
"""

SEED_DATA = """
INSERT OR IGNORE INTO users (username, email, role) VALUES
  ('alice', 'alice@example.com', 'admin'),
  ('bob', 'bob@example.com', 'user'),
  ('carol', 'carol@example.com', 'user'),
  ('dave', 'dave@example.com', 'user');

INSERT OR IGNORE INTO products (name, price, category, stock) VALUES
  ('Laptop', 999.99, 'Electronics', 50),
  ('Mouse', 29.99, 'Electronics', 200),
  ('Desk', 349.99, 'Furniture', 30),
  ('Chair', 249.99, 'Furniture', 45),
  ('Notebook', 4.99, 'Stationery', 500);

INSERT OR IGNORE INTO orders (user_id, status) VALUES
  (1, 'completed'), (2, 'completed'), (2, 'pending'),
  (3, 'completed'), (4, 'cancelled');

INSERT OR IGNORE INTO order_items (order_id, product_id, quantity, unit_price) VALUES
  (1, 1, 1, 999.99), (1, 2, 2, 29.99),
  (2, 3, 1, 349.99),
  (3, 4, 1, 249.99), (3, 5, 3, 4.99),
  (4, 1, 1, 999.99), (4, 2, 1, 29.99),
  (5, 5, 10, 4.99);
"""

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASK_EASY = {
    "name": "fix_syntax",
    "description": (
        "The query below contains syntax errors and a wrong JOIN condition. "
        "Fix it so it correctly returns all completed orders with the buyer's username, "
        "product names, quantities, and line totals, ordered by order id."
    ),
    "original_query": (
        "SELCT o.id AS order_id, u.username, p.name AS product, oi.quantity,\n"
        "       oi.quantity * oi.unit_pric AS line_total\n"
        "FORM orders o\n"
        "JOIN users u ON o.user_id = u.id\n"
        "JOIN order_items oi ON oi.order_d = o.id\n"
        "JOIN product p ON p.id = oi.product_id\n"
        "WHER o.status = 'completed'\n"
        "ORDER o.id;"
    ),
    "reference_query": (
        "SELECT o.id AS order_id, u.username, p.name AS product, oi.quantity,\n"
        "       oi.quantity * oi.unit_price AS line_total\n"
        "FROM orders o\n"
        "JOIN users u ON o.user_id = u.id\n"
        "JOIN order_items oi ON oi.order_id = o.id\n"
        "JOIN products p ON p.id = oi.product_id\n"
        "WHERE o.status = 'completed'\n"
        "ORDER BY o.id;"
    ),
    "max_steps": 6,
    "difficulty": "easy",
}

TASK_MEDIUM = {
    "name": "optimize_query",
    "description": (
        "The query below is logically correct but very slow: it uses a correlated subquery "
        "for each row and applies a function to an indexed column in the WHERE clause. "
        "Rewrite it to be efficient: use a JOIN instead of the subquery, avoid wrapping "
        "indexed columns in functions, and use a CTE or derived table where appropriate. "
        "The query should return each user's total spend on completed orders."
    ),
    "original_query": (
        "SELECT u.username,\n"
        "       (SELECT SUM(oi.quantity * oi.unit_price)\n"
        "        FROM orders o\n"
        "        JOIN order_items oi ON oi.order_id = o.id\n"
        "        WHERE o.user_id = u.id\n"
        "          AND UPPER(o.status) = 'COMPLETED') AS total_spend\n"
        "FROM users u\n"
        "WHERE (SELECT COUNT(*)\n"
        "       FROM orders o2\n"
        "       WHERE o2.user_id = u.id\n"
        "         AND UPPER(o2.status) = 'COMPLETED') > 0;"
    ),
    "reference_query": (
        "WITH completed_spend AS (\n"
        "    SELECT o.user_id,\n"
        "           SUM(oi.quantity * oi.unit_price) AS total_spend\n"
        "    FROM orders o\n"
        "    JOIN order_items oi ON oi.order_id = o.id\n"
        "    WHERE o.status = 'completed'\n"
        "    GROUP BY o.user_id\n"
        ")\n"
        "SELECT u.username, cs.total_spend\n"
        "FROM users u\n"
        "JOIN completed_spend cs ON cs.user_id = u.id\n"
        "ORDER BY cs.total_spend DESC;"
    ),
    "max_steps": 8,
    "difficulty": "medium",
    "anti_patterns": [
        r"SELECT\s+.*\(\s*SELECT",
        r"WHERE\s+.*\(\s*SELECT",
        r"UPPER\s*\(",
        r"LOWER\s*\(",
    ],
    "required_patterns": [
        r"JOIN",
        r"GROUP\s+BY",
        r"SUM\s*\(",
    ],
}

TASK_HARD = {
    "name": "security_audit",
    "description": (
        "This query is used in a web application's search feature. It contains MULTIPLE "
        "critical security vulnerabilities: (1) SQL injection via unsanitised string "
        "concatenation, (2) data leakage — it exposes the 'role' column to non-admin callers, "
        "(3) missing input validation (no limit on rows returned). "
        "Your task: rewrite the query to be injection-safe using parameterised style "
        "(use :param placeholders), remove the role column from the output for non-admin "
        "contexts, and add a LIMIT clause. Keep the search functionality working."
    ),
    "original_query": (
        "-- VULNERABLE: user input is concatenated directly into SQL string\n"
        "-- search_term comes from HTTP query parameter\n"
        'query = "SELECT id, username, email, role, created_at FROM users '
        "WHERE username LIKE '%\" + search_term + \"%' OR id = \" + search_term"
    ),
    "reference_patterns": {
        "no_concatenation": r"(?!.*\+\s*search_term)",
        "parameterized": r":[a-z_]+|\?",
        "no_role_leak": r"(?!.*\bSELECT\b.*\brole\b)",
        "has_limit": r"\bLIMIT\b",
        "has_like_or_where": r"\bLIKE\b|\bWHERE\b",
    },
    "max_steps": 10,
    "difficulty": "hard",
}

TASKS = {
    "fix_syntax": TASK_EASY,
    "optimize_query": TASK_MEDIUM,
    "security_audit": TASK_HARD,
}

# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

def make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_DDL)
    conn.executescript(SEED_DATA)
    conn.commit()
    return conn


def run_query(conn: sqlite3.Connection, sql: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
    try:
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        return rows, None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def grade_easy(revised_query: str, explanation: str, task: dict,
               conn: sqlite3.Connection, step: int, max_steps: int) -> Dict[str, Any]:
    sql = revised_query.strip()
    syntax_score = correctness_score = quality_score = explanation_score = penalty = 0.0
    feedback_parts = []

    rows, err = run_query(conn, sql)
    if err:
        feedback_parts.append(f"Syntax/runtime error: {err}")
    else:
        syntax_score = 0.4
        feedback_parts.append("Query executes without errors.")
        ref_rows, _ = run_query(conn, task["reference_query"])
        if ref_rows is not None:
            if len(rows) == len(ref_rows):
                correctness_score += 0.35
                feedback_parts.append(f"Row count matches ({len(rows)} rows). Good.")
            else:
                penalty += 0.1
                feedback_parts.append(f"Row count mismatch: expected {len(ref_rows)}, got {len(rows)}.")

            expected_cols = set(ref_rows[0].keys()) if ref_rows else set()
            actual_cols = set(rows[0].keys()) if rows else set()
            if expected_cols and expected_cols == actual_cols:
                correctness_score += 0.15
                feedback_parts.append("Column names match expected output.")
            elif expected_cols:
                missing = expected_cols - actual_cols
                extra = actual_cols - expected_cols
                if missing:
                    feedback_parts.append(f"Missing columns: {missing}")
                if extra:
                    feedback_parts.append(f"Unexpected extra columns: {extra}")

        sql_up = sql.upper()
        if "ORDER BY" in sql_up:
            quality_score += 0.05
        if "JOIN" in sql_up:
            quality_score += 0.05

    if len(explanation.strip()) > 20:
        explanation_score = 0.02
    if step > max_steps - 2 and correctness_score < 0.3:
        penalty += 0.05

    total = min(1.0, max(0.0, syntax_score + correctness_score + quality_score + explanation_score - penalty))
    return {
        "total": total,
        "syntax_score": syntax_score,
        "correctness_score": correctness_score,
        "quality_score": quality_score,
        "security_score": 0.0,
        "explanation_score": explanation_score,
        "penalty": penalty,
        "breakdown": "; ".join(feedback_parts),
    }


def grade_medium(revised_query: str, explanation: str, task: dict,
                 conn: sqlite3.Connection, step: int, max_steps: int) -> Dict[str, Any]:
    sql = revised_query.strip()
    sql_up = sql.upper()
    syntax_score = correctness_score = quality_score = explanation_score = penalty = 0.0
    feedback_parts = []

    rows, err = run_query(conn, sql)
    if err:
        feedback_parts.append(f"Error: {err}")
    else:
        syntax_score = 0.2
        ref_rows, _ = run_query(conn, task["reference_query"])
        if ref_rows and rows:
            if len(rows) == len(ref_rows):
                correctness_score += 0.3
                feedback_parts.append(f"Correct row count ({len(rows)}).")
            else:
                feedback_parts.append(f"Row count: expected {len(ref_rows)}, got {len(rows)}.")

            ref_spends = sorted(r.get("total_spend", 0) for r in ref_rows)
            act_spends = sorted(r.get("total_spend", 0) for r in rows)
            if ref_spends and act_spends and len(ref_spends) == len(act_spends):
                if all(abs(a - b) < 0.01 for a, b in zip(ref_spends, act_spends)):
                    correctness_score += 0.2
                    feedback_parts.append("Spend values are correct.")
                else:
                    feedback_parts.append("Spend values don't match expected.")

    anti_found = [p for p in task["anti_patterns"] if re.search(p, sql, re.IGNORECASE)]
    if anti_found:
        penalty += 0.1 * len(anti_found)
        feedback_parts.append(f"Still contains inefficient pattern(s): {len(anti_found)} found.")
    else:
        quality_score += 0.15
        feedback_parts.append("No anti-patterns detected.")

    req_found = sum(1 for p in task["required_patterns"] if re.search(p, sql, re.IGNORECASE))
    quality_score += 0.05 * req_found

    if re.search(r"\bWITH\b", sql_up):
        quality_score += 0.05
        feedback_parts.append("Good use of CTE.")
    if "ORDER BY" in sql_up:
        quality_score += 0.02

    if len(explanation.strip()) > 20:
        explanation_score = 0.02

    total = min(1.0, max(0.0, syntax_score + correctness_score + quality_score + explanation_score - penalty))
    return {
        "total": total,
        "syntax_score": syntax_score,
        "correctness_score": correctness_score,
        "quality_score": quality_score,
        "security_score": 0.0,
        "explanation_score": explanation_score,
        "penalty": penalty,
        "breakdown": "; ".join(feedback_parts),
    }


def grade_hard(revised_query: str, explanation: str, task: dict,
               conn: sqlite3.Connection, step: int, max_steps: int) -> Dict[str, Any]:
    sql = revised_query.strip()
    security_score = correctness_score = quality_score = explanation_score = penalty = 0.0
    feedback_parts = []

    has_concat = bool(re.search(r'"\s*\+\s*\w+|\'.*\+.*\'|\|\|\s*\w+\s*\|\|', sql))
    if not has_concat:
        security_score += 0.25
        feedback_parts.append("No string concatenation detected. Good.")
    else:
        penalty += 0.2
        feedback_parts.append("CRITICAL: Still uses string concatenation (injection risk).")

    if re.search(r':[a-z_]+|\?', sql):
        security_score += 0.2
        feedback_parts.append("Uses parameterized placeholders.")
    else:
        feedback_parts.append("No parameterized placeholders found — injection not fixed.")

    select_match = re.search(r'SELECT(.*?)FROM', sql, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_cols = select_match.group(1)
        if re.search(r'\brole\b', select_cols, re.IGNORECASE):
            penalty += 0.15
            feedback_parts.append("WARNING: 'role' column exposed in SELECT (data leakage).")
        else:
            security_score += 0.2
            feedback_parts.append("'role' column not exposed. Good.")

    if re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
        quality_score += 0.1
        feedback_parts.append("LIMIT clause present.")
    else:
        feedback_parts.append("Missing LIMIT clause — unbounded result set risk.")

    if re.search(r'\bWHERE\b', sql, re.IGNORECASE):
        correctness_score += 0.1
        feedback_parts.append("WHERE clause present.")
    if re.search(r'\bLIKE\b', sql, re.IGNORECASE):
        correctness_score += 0.1
        feedback_parts.append("Search capability preserved.")

    if len(explanation.strip()) > 30:
        explanation_score = 0.03

    total = min(1.0, max(0.0, security_score + correctness_score + quality_score + explanation_score - penalty))
    return {
        "total": total,
        "syntax_score": 0.0,
        "correctness_score": correctness_score,
        "quality_score": quality_score,
        "security_score": security_score,
        "explanation_score": explanation_score,
        "penalty": penalty,
        "breakdown": "; ".join(feedback_parts),
    }


GRADERS = {
    "fix_syntax": grade_easy,
    "optimize_query": grade_medium,
    "security_audit": grade_hard,
}

HINTS_MAP = {
    "fix_syntax": [
        "Check spelling of SQL keywords: SELECT, FROM, WHERE, ORDER BY.",
        "Table 'product' doesn't exist — check the schema for the correct name.",
        "JOIN conditions: order_items.order_id (not order_d), products (not product).",
    ],
    "optimize_query": [
        "Replace correlated subqueries with a CTE using WITH ... AS (...).",
        "Don't wrap status in UPPER() — compare directly to lowercase 'completed'.",
        "Use GROUP BY user_id and SUM() in the CTE, then JOIN to users.",
    ],
    "security_audit": [
        "Never build SQL by concatenating user input — use :param placeholders.",
        "Remove 'role' from the SELECT list to avoid data leakage.",
        "Add LIMIT 100 (or similar) to cap result size.",
    ],
}
