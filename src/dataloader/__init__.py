"""Benchmark integrations. One module per dataset, all exposing the same
interface so runners never need dataset-specific code:

    NAME            dataset name (str)
    SYSTEM_PROMPT   system message for the chat template, or None
    MAX_NEW_TOKENS  generation budget used by the pilot runs
    EVAL_TIMEOUT    per-program timeout (seconds) for pass@1
    load_tasks(limit=None, shard=None, num_shards=None) -> list[dict]
    task_id(task) -> str | int
    build_prompt(task) -> str          # user-message content
    human_reference(task) -> str|None  # benchmark's own solution, or None if
                                       # the dataset ships no reference code
    result_extras(task) -> dict        # extra per-result keys to store
    evaluate(task, code, timeout) -> {"passed": bool, "reason": str, ...}

IMPORTANT: modules in this package must NOT import the top-level src/utils.py
(the MarkLLM repro puts the MarkLLM repo — which has its own `utils` package —
ahead of src/ on sys.path). Keep them self-contained.
"""

import subprocess
import sys
from importlib import import_module
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PILOT_DIR / "data"

DATASETS = ("humaneval", "mbpp", "livecodebench")

# Universal system-prompt rule, appended to every dataset's SYSTEM_PROMPT:
# generated code must not contain comments (watermark segmentation and
# detection should see code tokens, not free-form commentary).
NO_COMMENTS_RULE = (
    "Do not include any comments in your code: no '#' comments, no "
    "docstrings, and no explanatory text inside the code block."
)


def system_prompt(ds, allow_comments: bool = False) -> str | None:
    """A dataset's system message, with NO_COMMENTS_RULE optionally dropped.

    Every dataloader composes SYSTEM_PROMPT as "<base> " + NO_COMMENTS_RULE, so
    the comments-allowed control cell is that same string minus the rule — the
    only difference between the two cells is the rule itself. The invariant is
    asserted rather than assumed: a dataloader that ever stops ending with the
    rule verbatim would otherwise leave the prohibition in the control cell, and
    the experiment would silently compare a cell against itself.
    """
    if not allow_comments:
        return ds.SYSTEM_PROMPT
    prompt = ds.SYSTEM_PROMPT
    if not prompt:
        return prompt
    if prompt.count(NO_COMMENTS_RULE) != 1 or not prompt.endswith(NO_COMMENTS_RULE):
        raise ValueError(
            f"{ds.NAME}.SYSTEM_PROMPT must end with exactly one verbatim "
            "NO_COMMENTS_RULE for the comments-allowed control to be well defined"
        )
    return prompt.removesuffix(NO_COMMENTS_RULE).strip() or None


def get(name: str):
    """Return the dataloader module for a dataset name."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name} (choose from {DATASETS})")
    return import_module(f"dataloader.{name}")


def normalize_name(meta_dataset: str) -> str:
    """Map a meta['dataset'] value (possibly an old long form such as
    'livecodebench_code_generation_test') to a dataloader name."""
    for n in DATASETS:
        if meta_dataset.startswith(n):
            return n
    raise ValueError(f"cannot map dataset name: {meta_dataset}")


def slice_tasks(tasks: list, limit=None, shard=None, num_shards=None) -> list:
    if limit:
        tasks = tasks[:limit]
    if shard is not None and num_shards:
        tasks = tasks[shard::num_shards]
    return tasks


def run_python(program: str, timeout: float, stdin: str | None = None) -> dict:
    """Run a python program in a fresh subprocess; pass iff exit code 0."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "reason": "timeout"}
    if proc.returncode != 0:
        return {"passed": False, "reason": "fail"}
    return {"passed": True, "reason": "pass"}
