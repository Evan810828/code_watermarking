"""HumanEval test split (164 tasks, data/humaneval_test.json)."""

import json

from . import DATA_DIR, NO_COMMENTS_RULE, run_python, slice_tasks

NAME = "humaneval"
# Simple system prompt: the task prompt (docstring + examples) already
# specifies the problem fully.
SYSTEM_PROMPT = "You are an expert Python programmer. " + NO_COMMENTS_RULE
MAX_NEW_TOKENS = 2000
EVAL_TIMEOUT = 10.0


def load_tasks(limit=None, shard=None, num_shards=None) -> list[dict]:
    with open(DATA_DIR / "humaneval_test.json") as f:
        return slice_tasks(json.load(f), limit, shard, num_shards)


def task_id(task: dict) -> str:
    return task["task_id"]


def build_prompt(task: dict) -> str:
    return (
        "Complete the following Python function. "
        "Output only the full function implementation (including the signature) "
        "in a single ```python code block.\n\n"
        f"```python\n{task['prompt']}```"
    )


def human_reference(task: dict) -> str:
    """Stub + canonical solution — the negative class the CodeWM paper's ROC
    metric scores against."""
    return task["prompt"] + task["canonical_solution"]


def result_extras(task: dict) -> dict:
    return {"task": task}


def evaluate(task: dict, code: str, timeout: float = EVAL_TIMEOUT) -> dict:
    program = f"{code}\n\n{task['test']}\n\ncheck({task['entry_point']})\n"
    return run_python(program, timeout)
