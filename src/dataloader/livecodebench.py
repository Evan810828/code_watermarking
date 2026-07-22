"""LiveCodeBench code_generation test split (release v1, 400 tasks).

Prompts follow the official LiveCodeBench generic prompt format
(lcb_runner/prompts/code_generation.py): a fixed system message plus a
question/format/answer user message; stdin problems ask for a self-contained
program, LeetCode-style problems ask to complete the starter code.

pass@1 runs against the prepped per-task test files
(SCRATCH/tests/{question_id}.json: public first then private, capped at 20,
inputs >10MB skipped at prep time):
- stdin problems: run the code as a program, feed test input on stdin,
  compare stdout line-by-line (exact after strip, float fallback at 1e-6).
- functional (LeetCode) problems: prepend an import header, call
  Solution().{func_name}(*json-decoded args), compare the JSON-serialized
  return value (float tolerance 1e-6, recursive).

Data prep (one-time; streams the 9.4GB test.jsonl with inline private tests):
    python -m dataloader.livecodebench   # from src/
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from . import DATA_DIR, NO_COMMENTS_RULE, slice_tasks

NAME = "livecodebench"
MAX_NEW_TOKENS = 2000
EVAL_TIMEOUT = 6.0

# Per-task test files are too large for the repo (~4GB). Default location is
# data/lcb/ (gitignored); point LCB_SCRATCH elsewhere (e.g. a scratch
# filesystem) to relocate. Populate once via `python -m dataloader.livecodebench`.
SCRATCH = Path(os.environ.get("LCB_SCRATCH", DATA_DIR / "lcb"))
TESTS_DIR = SCRATCH / "tests"

# Official LiveCodeBench system prompt + the universal no-comments rule.
SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests. " + NO_COMMENTS_RULE
)

FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program runs, "
    "it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def load_tasks(limit=None, shard=None, num_shards=None) -> list[dict]:
    with open(DATA_DIR / "livecodebench_test.json") as f:
        return slice_tasks(json.load(f), limit, shard, num_shards)


def task_id(task: dict) -> str:
    return task["task_id"]


def build_prompt(task: dict) -> str:
    """User-message content, LiveCodeBench generic format."""
    prompt = f"### Question:\n{task['question_content']}\n\n"
    if task["starter_code"].strip():
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{task['starter_code']}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def result_extras(task: dict) -> dict:
    # the full task (with question_content) is large; store labels only
    return {"platform": task["platform"], "difficulty": task["difficulty"]}


# ------------------------------------------------------------------ pass@1

IMPORT_HEADER = """\
import sys, os, collections, itertools, functools, math, heapq, bisect, re, json, string, random
from typing import *
from collections import *
from itertools import *
from functools import *
from heapq import *
from bisect import *
sys.setrecursionlimit(100000)
"""

FUNCTIONAL_DRIVER = """

if __name__ == "__main__":
    import json as _json, sys as _sys
    _args = [_json.loads(l) for l in _sys.stdin.read().strip().split("\\n")]
    _res = Solution().{func_name}(*_args)
    print(_json.dumps(_res))
"""


def values_close(a, b, tol=1e-6):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
        except (TypeError, ValueError):
            return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(values_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_close(a[k], b[k], tol) for k in a)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return a == b


def stdout_matches(got: str, expected: str) -> bool:
    gl = [l.rstrip() for l in got.strip().splitlines()]
    el = [l.rstrip() for l in expected.strip().splitlines()]
    if gl == el:
        return True
    if len(gl) != len(el):
        return False
    for g, e in zip(gl, el):
        if g == e:
            continue
        gt, et = g.split(), e.split()
        if len(gt) != len(et):
            return False
        for x, y in zip(gt, et):
            if x == y:
                continue
            try:
                if abs(float(x) - float(y)) > 1e-6 * max(1.0, abs(float(y))):
                    return False
            except ValueError:
                return False
    return True


def evaluate(task: dict, code: str, timeout: float = EVAL_TIMEOUT) -> dict:
    """task only needs 'task_id'; the test spec is loaded lazily per call
    (all 400 spec files together are ~4GB and would OOM a small cgroup if
    preloaded)."""
    spec_path = TESTS_DIR / f"{task['task_id']}.json"
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"LCB test spec missing: {spec_path}. Run the one-time prep "
            "(`python -m dataloader.livecodebench` from src/) or point the "
            "LCB_SCRATCH environment variable at a prepped directory."
        )
    with open(spec_path) as tf:
        spec = json.load(tf)
    if spec["testtype"] == "functional":
        if spec["func_name"] is None:
            return {"passed": False, "reason": "no_func_name"}
        program = IMPORT_HEADER + code + FUNCTIONAL_DRIVER.format(func_name=spec["func_name"])
    else:
        program = IMPORT_HEADER + code
    for i, t in enumerate(spec["tests"]):
        try:
            proc = subprocess.run(
                [sys.executable, "-c", program],
                input=t["input"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "reason": f"timeout@test{i}"}
        if proc.returncode != 0:
            return {"passed": False, "reason": f"error@test{i}", "stderr": proc.stderr[-300:]}
        if spec["testtype"] == "functional":
            try:
                ok = values_close(json.loads(proc.stdout.strip()), json.loads(t["output"]))
            except (json.JSONDecodeError, ValueError):
                ok = proc.stdout.strip() == t["output"].strip()
        else:
            ok = stdout_matches(proc.stdout, t["output"])
        if not ok:
            return {"passed": False, "reason": f"wrong@test{i}"}
    return {"passed": True, "reason": "pass", "n_tests": len(spec["tests"])}


# ------------------------------------------------------------------ data prep


def prep(jsonl_path: str, max_tests: int = 20, max_input_bytes: int = 10_000_000):
    """Stream the raw test.jsonl and write data/livecodebench_test.json (slim
    task list, no test cases) plus SCRATCH/tests/{question_id}.json."""
    tasks = []
    n_tests_written = 0
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            qid = d["question_id"]
            meta = json.loads(d["metadata"]) if d["metadata"] else {}
            pub = json.loads(d["public_test_cases"])
            priv = json.loads(d["private_test_cases"])

            kept, skipped_large = [], 0
            for t in pub + priv:
                if len(kept) >= max_tests:
                    break
                if len(t["input"]) > max_input_bytes:
                    skipped_large += 1
                    continue
                kept.append({"input": t["input"], "output": t["output"]})
            testtype = (pub + priv)[0]["testtype"] if (pub + priv) else "stdin"

            with open(TESTS_DIR / f"{qid}.json", "w") as tf:
                json.dump(
                    {
                        "task_id": qid,
                        "testtype": testtype,
                        "func_name": meta.get("func_name"),
                        "n_public": len(pub),
                        "n_private": len(priv),
                        "n_kept": len(kept),
                        "skipped_large": skipped_large,
                        "tests": kept,
                    },
                    tf,
                )
            n_tests_written += 1

            tasks.append(
                {
                    "task_id": qid,
                    "title": d["question_title"],
                    "platform": d["platform"],
                    "contest_date": d["contest_date"],
                    "difficulty": d["difficulty"],
                    "question_content": d["question_content"],
                    "starter_code": d["starter_code"],
                    "func_name": meta.get("func_name"),
                    "testtype": testtype,
                    "n_public": len(pub),
                    "n_private": len(priv),
                }
            )
            if len(tasks) % 50 == 0:
                print(f"{len(tasks)} processed", flush=True)

    out = DATA_DIR / "livecodebench_test.json"
    with open(out, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Wrote {len(tasks)} tasks -> {out}")
    print(f"Wrote {n_tests_written} test files -> {TESTS_DIR}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prep LiveCodeBench data")
    ap.add_argument("--jsonl", default=str(SCRATCH / "test.jsonl"))
    ap.add_argument("--max-tests", type=int, default=20)
    ap.add_argument("--max-input-bytes", type=int, default=10_000_000)
    args = ap.parse_args()
    prep(args.jsonl, args.max_tests, args.max_input_bytes)
