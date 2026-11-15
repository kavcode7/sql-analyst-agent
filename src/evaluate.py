"""
Phase 4 — Evaluation harness.

For every benchmark question and every system:
  1. ask the system the question
  2. pull out the SQL it used
  3. run that SQL and the gold SQL on the database
  4. compare the two results  ->  correct / incorrect

Run from the project root:
    python src/evaluate.py --show-gold                     # print gold answers so you can verify them
    python src/evaluate.py                                 # all systems, all questions, 1 run
    python src/evaluate.py --systems agent_loop --runs 3   # one system, 3 repeats
    python src/evaluate.py --ids e2 h2 h5                  # only some questions
    python src/evaluate.py --obfuscated                    # contamination test: disguised database
"""
import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

import schema
from schema import get_connection, get_schema
from tools import execute_with_timeout

load_dotenv()

RESULTS_DIR = Path("results")
MAX_RESULT_ROWS = 1000
SQL_BLOCK = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)


# ---------- running SQL ----------

def execute(sql: str) -> list[tuple]:
    with closing(get_connection()) as conn:  # read-only, one connection per call (thread-safe)
        _, rows = execute_with_timeout(conn, sql, MAX_RESULT_ROWS, timeout_s=30)
        return rows


# ---------- comparing results ----------
# Rule: the prediction is correct if every column in the gold result also appears as a
# column in the predicted result. Extra predicted columns (e.g. a count next to a name)
# and row order are ignored. Numbers match within 1% (so 12.49 == 12.5), text ignores case.
# Gold SQL is written to return ONLY what the question asks for, which keeps this fair.

def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _cells_equal(a, b) -> bool:
    if _is_number(a) and _is_number(b):
        return math.isclose(float(a), float(b), rel_tol=0.01, abs_tol=0.01)
    return str(a).strip().lower() == str(b).strip().lower()


def _sort_key(value):
    if _is_number(value):
        return (0, float(value), "")
    return (1, 0.0, str(value).strip().lower())


def _column(rows: list[tuple], index: int) -> list:
    return sorted((row[index] for row in rows), key=_sort_key)


def results_match(gold_rows: list[tuple], pred_rows: list[tuple]) -> bool:
    if len(gold_rows) != len(pred_rows):
        return False
    if not gold_rows:
        return True
    pred_columns = [_column(pred_rows, i) for i in range(len(pred_rows[0]))]
    for gold_index in range(len(gold_rows[0])):
        gold_column = _column(gold_rows, gold_index)
        if not any(
            all(_cells_equal(g, p) for g, p in zip(gold_column, pred_column))
            for pred_column in pred_columns
        ):
            return False
    return True


# ---------- the systems under test ----------
# Each system takes a question and returns (sql_it_used, its_full_answer_text).

def extract_sql(text: str | None) -> str | None:
    """The agents end their answer with a ```sql block; take the last one."""
    blocks = SQL_BLOCK.findall(text or "")
    return blocks[-1].strip() if blocks else None


def make_systems(client: Anthropic) -> dict:
    import agent
    import ask
    import graph_agent

    with closing(get_connection()) as conn:
        schema = get_schema(conn)  # build once, reuse for every one-shot call

    def one_shot(question):
        result = ask.generate_sql(client, schema, question)
        return (result.sql if result.answerable else None), result.reasoning

    def agent_loop(question):
        with closing(get_connection()) as conn:
            text = agent.run_agent(question, client, conn, verbose=False)
        return extract_sql(text), text

    def langgraph(question):
        text = graph_agent.run_graph_agent(question)["answer"]
        return extract_sql(text), text

    return {"one_shot": one_shot, "agent_loop": agent_loop, "langgraph": langgraph}


# ---------- grading ----------

def grade(item: dict, gold_rows, predicted_sql, answer_text) -> tuple[bool, str | None]:
    if item["gold_sql"] is None:
        # Unanswerable: correct only if the system declined (gave no SQL) and didn't just time out.
        stopped = (answer_text or "").startswith("Stopped:")
        if predicted_sql is None and not stopped:
            return True, None
        return False, "answered an unanswerable question" if predicted_sql else "hit step limit"
    if predicted_sql is None:
        return False, "no SQL in answer"
    try:
        pred_rows = execute(predicted_sql)
    except Exception as error:
        return False, f"SQL error: {error}"
    if results_match(gold_rows, pred_rows):
        return True, None
    return False, f"wrong result: {pred_rows[:3]}"


def run_one(system_name, system_fn, item, gold_rows, run_index) -> dict:
    start = time.perf_counter()
    try:
        predicted_sql, answer_text = system_fn(item["question"])
        correct, error = grade(item, gold_rows, predicted_sql, answer_text)
    except Exception as crash:
        predicted_sql, answer_text = None, None
        correct, error = False, f"system crashed: {crash}"
    return {
        "system": system_name,
        "run": run_index,
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "correct": correct,
        "error": error,
        "predicted_sql": predicted_sql,
        "answer": answer_text,
        "latency_s": round(time.perf_counter() - start, 2),
    }


# ---------- reporting ----------

def print_summary(results: list[dict], systems: list[str]) -> None:
    categories = sorted({r["category"] for r in results})
    print("\n" + "=" * 70)
    header = f"{'category':<16}" + "".join(f"{s:>16}" for s in systems)
    print(header)
    print("-" * len(header))

    def accuracy(rows):
        return f"{100 * sum(r['correct'] for r in rows) / len(rows):.0f}%" if rows else "-"

    for category in categories:
        line = f"{category:<16}"
        for system in systems:
            rows = [r for r in results if r["system"] == system and r["category"] == category]
            line += f"{accuracy(rows):>16}"
        print(line)

    print("-" * len(header))
    for label, key in [("OVERALL", None), ("avg latency (s)", "latency")]:
        line = f"{label:<16}"
        for system in systems:
            rows = [r for r in results if r["system"] == system]
            if key == "latency":
                line += f"{sum(r['latency_s'] for r in rows) / len(rows):>16.1f}"
            else:
                line += f"{accuracy(rows):>16}"
        print(line)
    print("=" * 70)

    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n{len(failures)} failures:")
        for r in sorted(failures, key=lambda r: (r["system"], r["id"])):
            print(f"  [{r['system']}] {r['id']} (run {r['run']}): {r['error']}")


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=["one_shot", "agent_loop", "langgraph"])
    parser.add_argument("--runs", type=int, default=1, help="repeat each question N times")
    parser.add_argument("--ids", nargs="+", help="only run these question ids")
    parser.add_argument("--workers", type=int, default=5, help="questions run in parallel")
    parser.add_argument("--show-gold", action="store_true", help="print gold answers and exit")
    parser.add_argument("--obfuscated", action="store_true",
                        help="run on the disguised database (contamination test)")
    args = parser.parse_args()

    # Every get_connection() call reads schema.DB_PATH, so changing it here switches the
    # database for the gold queries AND for every system under test.
    if args.obfuscated:
        schema.DB_PATH = Path("data/olist_obf.db")
        benchmark_path = Path("benchmark/questions_obf.json")
    else:
        benchmark_path = Path("benchmark/questions.json")
    print(f"Database: {schema.DB_PATH}   Benchmark: {benchmark_path}")

    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if args.ids:
        benchmark = [item for item in benchmark if item["id"] in args.ids]

    # Expected answers come from running the gold SQL, never from typed-in numbers.
    gold = {item["id"]: execute(item["gold_sql"]) if item["gold_sql"] else None for item in benchmark}

    if args.show_gold:
        for item in benchmark:
            print(f"\n[{item['id']}] {item['question']}")
            print(f"  SQL:    {item['gold_sql']}")
            print(f"  ANSWER: {gold[item['id']][:5] if gold[item['id']] is not None else 'UNANSWERABLE'}")
        return

    # timeout: give up on a single API call after 60s; max_retries: retry throttled calls.
    all_systems = make_systems(Anthropic(timeout=60.0, max_retries=3))
    systems = {name: all_systems[name] for name in args.systems}

    jobs = [
        (name, fn, item, gold[item["id"]], run)
        for run in range(1, args.runs + 1)
        for item in benchmark
        for name, fn in systems.items()
    ]
    print(f"Running {len(jobs)} jobs ({len(benchmark)} questions x {len(systems)} systems x {args.runs} runs)...")

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = "obf" if args.obfuscated else "real"
    out_path = RESULTS_DIR / f"eval_{tag}_{datetime.now():%Y%m%d_%H%M%S}.json"

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, *job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            mark = "PASS" if result["correct"] else "FAIL"
            print(f"  {done:>3}/{len(jobs)} {mark} [{result['system']}] {result['id']} ({result['latency_s']}s)")
            # Save after every job, so stopping early (Ctrl+C) never loses finished results.
            out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print_summary(results, list(systems))
    print(f"\nFull results saved to {out_path}")

    if os.getenv("LANGFUSE_PUBLIC_KEY") and "langgraph" in systems:
        from langfuse import get_client

        get_client().flush()


if __name__ == "__main__":
    main()
