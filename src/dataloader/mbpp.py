"""MBPP test split (500 tasks, data/mbpp_test.json)."""

import json

from . import DATA_DIR, NO_COMMENTS_RULE, run_python, slice_tasks

NAME = "mbpp"
# Simple system prompt: the task prompt (text + test cases) already
# specifies the problem fully.
SYSTEM_PROMPT = "You are an expert Python programmer. " + NO_COMMENTS_RULE
MAX_NEW_TOKENS = 2000
EVAL_TIMEOUT = 10.0


def load_tasks(limit=None, shard=None, num_shards=None) -> list[dict]:
    with open(DATA_DIR / "mbpp_test.json") as f:
        return slice_tasks(json.load(f), limit, shard, num_shards)


def task_id(task: dict) -> int:
    return int(task["task_id"])


def build_prompt(task: dict) -> str:
    tests = "\n".join(task["test_list"])
    return (
        f"{task['text']}\n\n"
        "Your code should pass these tests:\n\n"
        f"```python\n{tests}\n```\n\n"
        "Output only your solution in a single ```python code block."
    )


def human_reference(task: dict) -> str:
    """The benchmark's own solution — the negative class the CodeWM paper's ROC
    metric scores against."""
    return task["code"]


def result_extras(task: dict) -> dict:
    return {"task": task}


def evaluate(task: dict, code: str, timeout: float = EVAL_TIMEOUT) -> dict:
    setup = task.get("test_setup_code") or ""
    tests = "\n".join(task["test_list"])
    program = f"{setup}\n{code}\n\n{tests}\n"
    return run_python(program, timeout)
