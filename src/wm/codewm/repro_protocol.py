"""Upstream evaluation protocol for the CodeWM reproduction runs.

The completion-prompt protocol is used by run_codewm_repro.py and retained
separately from the chat-prompt benchmark protocol.

The original repo (../code-watermark) evaluates on bigcode-harness style
*completion* prompts — no chat template, no system prompt: HumanEval is the
raw function stub, MBPP is the InCoder-style 3-shot docstring prompt. Per
task it samples n=20 completions at T=0.2/top_p=0.95/top_k=0 (max_length
2048 total, 60 s generation budget, batch stop when every sample contains a
task stop word), truncates each sample at the stop words, executes
prompt+completion against the benchmark tests (code_eval pass@k), and runs
z-detection on the re-tokenized postprocessed completion of the FIRST valid
sample — with the human reference solution (prompt + canonical solution) as
the detection negative class.

Faithful re-implementation of lm_eval/tasks/{humaneval,mbpp}.py plus the
src/main.py generate/evaluate plumbing. Not imported from the sibling repo:
its lm_eval package pulls in the whole task zoo and the `evaluate` hub
metric (absent from this env); the required logic is small and copied
verbatim instead. Documented deltas (none change behavior):
  - pass@k runs each candidate + "\n" + reference in a fresh subprocess
    (dataloader.run_python semantics; pass iff exit 0 within 3 s) instead of
    HF `evaluate`'s in-process code_eval; the unbiased pass@k estimator is
    the same.
  - MBPP postprocessing is kept bug-for-bug: upstream's first_block does
    re.split("|".join(stop_words)), and both the "\n<|/" stop word and the
    tokenizer eos string appended at generation time contain regex
    metacharacters, so MBPP completions are also split at bare "<", "/",
    ">" — reproduced as-is because the original numbers come from this code.
  - MBPP loads google-research-datasets/mbpp (current hub name of the same
    "mbpp" dataset upstream loads; identical content and splits).
  - models whose config vocab is padded beyond len(tokenizer) (e.g.
    deepseek-coder's 32256 vs 32022) are used as-is; upstream resizes the
    embeddings down to len(tokenizer). Every component here (CodeModel,
    greenlist pairing, detection) consistently uses the config vocab, and
    the CodeModel is retrained per model either way, so only the (already
    stochastic) pair realization differs — never the mechanism.

Import constraint (IMPLEMENTATION.md): wm/ must not import top-level src/utils.py.
"""

from __future__ import annotations

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import StoppingCriteria

# upstream src/main.py run recipe (run.sh): the sampling/decoding settings
# every repro stage (corpus gen and inference) must use
GEN_KWARGS = dict(do_sample=True, temperature=0.2, top_p=0.95, top_k=0)
MAX_LENGTH = 2048  # prompt + completion budget (tokens)
MAX_TIME = 60.0  # per-generate-call wall clock budget (s)
N_SAMPLES = 20
EVAL_TIMEOUT = 3.0  # code_eval default; HumanEval task also hardcodes 3.0


class EndOfFunctionCriteria(StoppingCriteria):
    """Stops once ALL sequences in the batch contain a stop string
    (upstream src/utils.py, verbatim)."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        return all(
            any(stop in d for stop in self.eof_strings) for d in decoded_generations
        )


class ReproHumanEval:
    """lm_eval/tasks/humaneval.py (strip_prompt=True, the upstream default)."""

    NAME = "humaneval"

    def __init__(self):
        from datasets import load_dataset

        # current hub name of upstream's "openai_humaneval" (same dataset)
        self.docs = list(load_dataset("openai/openai_humaneval")["test"])
        self.stop_words = [
            "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```",
            "<file_sep>",
        ]

    def task_id(self, doc) -> str:
        return doc["task_id"]

    def get_prompt(self, doc) -> str:
        return doc["prompt"].strip()

    def get_reference(self, doc) -> str:
        return "\n" + doc["test"] + "\n" + f"check({doc['entry_point']})"

    def get_full_data(self, doc) -> str:
        return self.get_prompt(doc) + "\n" + doc["canonical_solution"]

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def postprocess_generation(self, generation: str, doc) -> str:
        prompt = self.get_prompt(doc)
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)


class ReproMBPP:
    """lm_eval/tasks/mbpp.py (InCoder-style 3-shot prompt, task_id-seeded)."""

    NAME = "mbpp"

    def __init__(self):
        from datasets import load_dataset

        ds = load_dataset("google-research-datasets/mbpp")
        self.train = list(ds["train"])
        self.docs = list(ds["test"])
        assert len(self.docs) == 500, "wrong MBPP test split"
        self.rng = torch.Generator()
        self.few_shot = 3
        self.stop_words = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/"]

    def task_id(self, doc) -> int:
        return doc["task_id"]

    def get_prompt(self, doc) -> str:
        prompt = ""
        self.rng.manual_seed(doc["task_id"])
        train_idx_permutation = torch.randperm(
            len(self.train), generator=self.rng
        )[: self.few_shot].tolist()
        for train_idx in train_idx_permutation:
            train_data = self.train[train_idx]
            prompt += f'"""\n{train_data["text"]} Your code should satisfy these tests:\n'
            prompt += "\n".join(train_data["test_list"]) + '\n"""\n'
            prompt += train_data["code"] + "\n"
        prompt += f'"""\n{doc["text"]} Your code should satisfy these tests:\n'
        prompt += (
            "\n".join(doc["test_list"]) + '\n"""\n'
            if doc["task_id"] != 493
            else "\n".join(doc["test_list"][2:3]) + '\n"""\n'
        )
        return prompt

    def get_reference(self, doc) -> str:
        return "\n".join(doc["test_list"])

    def get_full_data(self, doc) -> str:
        return self.get_prompt(doc) + doc["code"]

    @staticmethod
    def first_block(string, stop_words):
        """Upstream verbatim, regex quirks included (see module docstring)."""
        return re.split("|".join(stop_words), string)[0].rstrip()

    def postprocess_generation(self, generation: str, doc) -> str:
        prompt = self.get_prompt(doc)
        output = generation[len(prompt) :]
        return prompt + self.first_block(output, self.stop_words)


TASKS = {"humaneval": ReproHumanEval, "mbpp": ReproMBPP}


# --------------------------------------------------------------- detection


def full_and_prefix_len(tokenizer, prompt: str, text: str):
    """Upstream evaluate_code slicing: re-tokenize the postprocessed text
    and locate the prompt-prefix boundary. Returns (full_ids, prefix_len),
    or None when nothing lies beyond the prompt (invalid sample). Mirrors
    the prefix_len slice + the find_code_start fallback as a
    longest-common-prefix slice."""
    full = tokenizer(text, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    prefix = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    plen = len(prefix)
    if len(full) <= plen:
        return None
    if full[: plen - 1] == prefix[:-1]:
        return full, plen
    common = 0
    while common < min(len(full), plen) and full[common] == prefix[common]:
        common += 1
    if common == len(full):
        return None
    return full, common


def completion_ids_for_detection(tokenizer, prompt: str, text: str):
    """The completion ids alone (WPN detection scores just these)."""
    sliced = full_and_prefix_len(tokenizer, prompt, text)
    if sliced is None:
        return None
    full, plen = sliced
    return full[plen:]


# ----------------------------------------------------------------- pass@k


def _run_program(program: str, timeout: float) -> bool:
    """Fresh-subprocess execution, pass iff exit code 0 (code_eval
    semantics; same criterion as dataloader.run_python)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def _estimator(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021; code_eval's estimator)."""
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def pass_at_k(task, doc, candidates: list[str], workers: int = 8) -> dict:
    """code_eval over postprocessed candidates: run candidate + "\\n" +
    reference, then the unbiased pass@{1,10} estimator."""
    reference = task.get_reference(doc)
    programs = [c + "\n" + reference for c in candidates]
    with ThreadPoolExecutor(max_workers=min(workers, len(programs))) as ex:
        results = list(ex.map(lambda p: _run_program(p, EVAL_TIMEOUT), programs))
    n, c = len(results), sum(results)
    return {
        "pass@1": _estimator(n, c, 1),
        "pass@10": _estimator(n, c, 10) if n >= 10 else None,
        "pass_results": results,
    }
