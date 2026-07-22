"""Official SemStamp from bohanhou14/SemStamp — not MarkLLM and not the pilot
re-implementation.

Official components used verbatim:
  - sampling_lsh_utils.lsh_reject_completion  (generation-time rejection sampling)
  - sbert_lsh_model.SBERTLSHModel             (SBERT + nearpy LSH, seed 1234)
  - detection_utils.detect_lsh                (chain z-test detection)

Adapter shims (documented, none touch the algorithm):
  - model.config._name_or_path gets an "-opt" suffix appended (prepare_model)
    because the official lsh_reject_completion raises NotImplementedError
    unless "opt" is a substring of the model name; the gated branch
    (gen_sent) is model-agnostic.
  - detection_utils instantiates a BERTScorer (deberta-xlarge) at import time,
    used only for paraphrase quality metrics we don't need -> stubbed.
  - model.generate is wrapped with a pass-through counter (prepare_model) to
    record the number of rejection-sampling trials and generated tokens.
  - The released lsh_reject_completion's N_max fallback is unreachable dead
    code: a candidate outside the accept mask hits an unconditional `continue`
    before the fallback line, so near-deterministic (code) segments resample
    forever (the same dead code was inherited by MarkLLM's copy).
    lsh_reject_completion_fallback_enforced is a verbatim copy with the
    single condition fixed to `not in accept_mask AND current_trials <
    MAX_TRIALS`, which makes the documented fallback branch reachable exactly
    as described in the paper.
  - SentenceEndCriteria.update() counts sentences on the raw text it is given,
    but the stop check counts on tokenizer.decode(..., skip_special_tokens=True).
    With a chat-template prompt (special-token markers stripped by decode) the
    two counts can disagree, making the next gen_sent() return 0 tokens and the
    official loop break mid-completion. DecodeConsistentSentenceEndCriteria
    (patched into sampling_lsh_utils) makes update() count in the same decoded
    space; when the counts already agree (the original plain-text setting)
    behavior is identical.
  - segmenter="line" (the pilot's code adaptation, not in the released code):
    a segment is one non-blank code line instead of one nltk sentence.
    LineEndCriteria mirrors SentenceEndCriteria in decoded space, and gen_line
    replaces gen_sent's discard-last-token heuristic with an explicit
    token-boundary cut after the first completed non-blank line (leading blank
    lines stay attached to the segment). Everything else — rejection loop,
    margin test, mask seeding, chain z-test — is the official code path.

Detection scores sents = [chat prompt] + segments(completion) (nltk
sent_tokenize, or non-blank lines in line mode), mirroring the official
pipeline where the stored text is prompt+continuation and the first segment
seeds the chain (at generation time the mask is seeded by
LSH(full prompt string)).
"""

import sys
import types

import torch
from nltk.tokenize import sent_tokenize

from . import add_repo_paths

add_repo_paths()
# detection_utils builds a BERTScorer at import; only needed for paraphrase
# metrics -> stub the module before the import.
sys.modules.setdefault("bert_score", types.SimpleNamespace(BERTScorer=lambda **kw: None))
import sampling_lsh_utils  # noqa: E402
import sampling_utils  # noqa: E402
from detection_utils import detect_lsh  # noqa: E402
from sampling_utils import SentenceEndCriteria  # noqa: E402
from sbert_lsh_model import SBERTLSHModel  # noqa: E402


class DecodeConsistentSentenceEndCriteria(SentenceEndCriteria):
    """update() must count sentences in the same string space the stop check
    sees (decode with skip_special_tokens=True); see module docstring."""

    def update(self, current_text):
        ids = self.tokenizer.encode(current_text)
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
        self.current_num_sentences = len(sent_tokenize(decoded))


# lsh_reject_completion resolves SentenceEndCriteria from its own module namespace
sampling_lsh_utils.SentenceEndCriteria = DecodeConsistentSentenceEndCriteria


def segment_lines(text: str) -> list[str]:
    """line-mode segmentation: one segment per non-blank line (matches the
    pilot's segment_text)."""
    return [l.strip() for l in text.split("\n") if l.strip()]


class LineEndCriteria(SentenceEndCriteria):
    """segmenter='line' analogue of SentenceEndCriteria: a 'sentence' is a
    non-blank code line. Counts lines that have started (blank lines don't
    count) in the same decoded space as the stop check; like the official
    criteria, stops once generation is one line PLUS the start of the next."""

    @staticmethod
    def _n_started_lines(text: str) -> int:
        return sum(1 for piece in text.split("\n") if piece.strip())

    def update(self, current_text):
        ids = self.tokenizer.encode(current_text)
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
        self.current_num_sentences = self._n_started_lines(decoded)

    def __call__(self, input_ids, scores, **kwargs):
        assert input_ids.size(0) == 1
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return self._n_started_lines(text) > self.current_num_sentences + 1


def gen_line(model, tokenizer, text_ids, gen_config, stopping_criteria):
    """gen_sent for segmenter='line'. The official gen_sent discards the last
    generated token (the one that started the next sentence); for lines that
    heuristic can eat the newline when a token spans it, so instead cut at the
    first token boundary where a non-blank line has been completed (leading
    blank lines stay attached). No cut point (EOS / budget) keeps everything."""
    outputs = model.generate(text_ids, gen_config, stopping_criteria=stopping_criteria)
    gen_ids = outputs[0, text_ids.size(1):]
    cut = None
    for j in range(1, len(gen_ids) + 1):
        decoded = tokenizer.decode(gen_ids[:j], skip_special_tokens=True)
        pieces = decoded.split("\n")
        if len(pieces) > 1 and any(p.strip() for p in pieces[:-1]):
            cut = j
            break
    if cut is not None:
        gen_ids = gen_ids[:cut]
    new_text_ids = torch.cat([text_ids, gen_ids.unsqueeze(0)], dim=1)
    new_text = tokenizer.decode(new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
    return new_text, new_text_ids


def lsh_reject_completion_fallback_enforced(
    prompt, model, tokenizer, gen_config, lsh_model, lsh_dim,
    lmbd=1.0, device="cuda", margin=0.002, segmenter="sent", **kwargs
):
    """Verbatim copy of sampling_lsh_utils.lsh_reject_completion with three
    deltas: (1) the mask-rejection `continue` is gated on
    current_trials < MAX_TRIALS so the released N_max fallback branch is
    reachable (see module docstring), (2) MAX_TRIALS is read from
    sampling_lsh_utils at call time so set_max_trials() applies, and
    (3) segmenter='line' swaps the sentence stop-criteria/gen_sent pair for
    their line-mode analogues (LineEndCriteria/gen_line)."""
    from transformers import StoppingCriteriaList
    from sampling_lsh_utils import get_mask_from_seed, reject_close_generation
    from sampling_utils import gen_sent

    stats = {}
    if segmenter == "line":
        sent_end_criteria = LineEndCriteria(tokenizer)
        gen_segment = gen_line
    else:
        sent_end_criteria = DecodeConsistentSentenceEndCriteria(tokenizer)
        gen_segment = gen_sent
    lsh_seed = lsh_model.get_hash([prompt])[0]
    accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed)

    text = prompt
    new_text = prompt
    text_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = len(text_ids[0])
    sent_end_criteria.update(new_text)

    total_trials = 0
    success_trials = 0
    current_trials = 0
    maxedout_trials = 0
    debug_text_segments = [(prompt, text_ids.size(1), lsh_seed)]
    while True:
        MAX_TRIALS = sampling_lsh_utils.MAX_TRIALS
        stopping_criteria = StoppingCriteriaList([sent_end_criteria])
        if "opt" in model.config._name_or_path:
            new_text, new_text_ids = gen_segment(
                model=model,
                tokenizer=tokenizer,
                text_ids=text_ids,
                gen_config=gen_config,
                stopping_criteria=stopping_criteria,
            )
        else:
            raise NotImplementedError("model type not supported")
        if new_text == "":
            print(
                "WARNING: stopped generation because generated nothing (after discarding last generated token)",
                flush=True,
            )
            break
        total_trials += 1
        current_trials += 1
        accepted_text, _ = reject_close_generation(lsh_model, [new_text], margin=margin, cutoff=None)
        if len(accepted_text) == 0 and current_trials < MAX_TRIALS:
            continue
        lsh_candidate = lsh_model.get_hash([new_text])[0]
        if lsh_candidate not in accept_mask and current_trials < MAX_TRIALS:
            continue
        if (lsh_candidate in accept_mask) or current_trials >= MAX_TRIALS:
            if current_trials >= MAX_TRIALS and lsh_candidate not in accept_mask:
                print(
                    f"WARNING: desired semantic signature can't be sampled after max_trials {MAX_TRIALS}",
                    flush=True,
                )
                maxedout_trials += 1
            debug_text_segments.append((new_text, new_text_ids.size(1) - text_ids.size(1), lsh_candidate))
            current_trials = 0
            success_trials += 1
            lsh_seed = lsh_candidate
            accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed)
            text += new_text
            text_ids = new_text_ids
            sent_end_criteria.update(text)
            if (len(text_ids[0]) - prompt_length) >= gen_config.max_new_tokens - 1:
                break
    return text


def set_max_trials(n: int):
    """MAX_TRIALS is a module-level constant snapshot; patch both copies."""
    sampling_utils.MAX_TRIALS = n
    sampling_lsh_utils.MAX_TRIALS = n


def prepare_model(model) -> dict:
    """Apply the "-opt" name shim and wrap model.generate with a pass-through
    counter; returns the counter dict ({"calls", "tokens"})."""
    model.config._name_or_path = model.config._name_or_path + "-opt"
    counter = {"calls": 0, "tokens": 0}
    orig_generate = model.generate

    def counting_generate(input_ids, *a, **kw):
        out = orig_generate(input_ids, *a, **kw)
        counter["calls"] += 1
        counter["tokens"] += int(out.shape[1] - input_ids.shape[1])
        return out

    model.generate = counting_generate
    return counter


class OfficialSemStamp:
    """Official SemStamp generation + detection behind the unified runner
    interface (generate(prompt) -> wm dict, detect(prompt, completion))."""

    def __init__(self, model, tokenizer, gen_config, device, embedder,
                 sp_dim, lmbd, margin, max_trials, z_threshold,
                 segmenter="line"):
        set_max_trials(max_trials)
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config
        self.device = device
        self.sp_dim = sp_dim
        self.lmbd = lmbd
        self.margin = margin
        self.z_threshold = z_threshold
        self.segmenter = segmenter
        self.counter = prepare_model(model) if model is not None else None
        self.lsh_model = SBERTLSHModel(
            lsh_model_path=embedder,
            device=device,
            batch_size=32,
            lsh_dim=sp_dim,
            sbert_type="base",
        )
        self.meta = {
            "algorithm": "SemStamp-LSH",
            "implementation": "official bohanhou14/SemStamp (sampling_lsh_utils, detection_utils)",
            "embedder": embedder,
            "sp_dim": sp_dim,
            "lmbd": lmbd,
            "margin": margin,
            "max_trials": max_trials,
            "z_threshold": z_threshold,
            "segmenter": (
                "line (pilot code adaptation)" if segmenter == "line"
                else "nltk sent_tokenize (official)"
            ),
        }

    def generate(self, prompt: str) -> dict:
        self.counter["calls"] = self.counter["tokens"] = 0
        text = lsh_reject_completion_fallback_enforced(
            prompt,
            self.model,
            self.tokenizer,
            self.gen_config,
            self.lsh_model,
            self.sp_dim,
            lmbd=self.lmbd,
            device=self.device,
            margin=self.margin,
            segmenter=self.segmenter,
        )
        completion = text[len(prompt):]
        det = self.detect(prompt, completion)
        return {
            "completion": completion,
            "detection": det,
            "sampling_stats": {
                "total_trials": self.counter["calls"],
                "total_gen_tokens": self.counter["tokens"],
                "n_test_sents": det["n_test_sents"],
            },
        }

    def detect(self, prompt: str, completion: str) -> dict:
        segments = (
            segment_lines(completion) if self.segmenter == "line"
            else sent_tokenize(completion)
        )
        sents = [prompt] + segments
        n_test = len(sents) - 1
        if n_test < 1:
            return {"z_score": None, "n_test_sents": 0, "is_watermarked": False}
        z = float(detect_lsh(sents=sents, lsh_model=self.lsh_model, lmbd=self.lmbd, lsh_dim=self.sp_dim))
        return {"z_score": z, "n_test_sents": n_test, "is_watermarked": z > self.z_threshold}
