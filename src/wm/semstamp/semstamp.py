"""SemStamp watermark (Hou et al., 2024, "SemStamp: A Semantic Watermark with
Paraphrastic Robustness for Text Generation").

Implementation follows MarkLLM's watermark/semstamp
(https://github.com/THU-BPM/MarkLLM/blob/main/watermark/semstamp/semstamp.py):

- Embedding space is partitioned by `dimension_d` random hyperplanes (LSH,
  nearpy RandomBinaryProjections with rand_seed=1234 — replicated here with
  numpy.random.RandomState so the normals are bit-for-bit identical without
  the nearpy dependency) into 2^d regions.
- A pseudo-random subset of gamma * 2^d regions is "valid", seeded by
  prime_P * (LSH signature of the previous segment).
- Generation: sample candidate next segments until one (a) is at least
  margin_m away from every hyperplane (|cos| >= margin_m) and (b) falls in a
  valid region; give up after N_max trials and keep the last sample
  (MarkLLM behaviour). The accepted segment's signature seeds the next step.
- Detection: re-derive the valid set per segment (seeded by the previous
  segment) and z-test the fraction of valid segments against gamma.

Default hyperparameters mirror MarkLLM's config/SEMSTAMP.json:
gamma=0.25, margin_m=0.01, dimension_d=3, prime_P=15485863, N_max=100,
z_threshold=3.0, embedder=AbeHou/SemStamp-c4-sbert.

Deviations from MarkLLM (documented, motivated by code generation):
- Segmenter is pluggable: "nltk" (MarkLLM's sent_tokenize) or "line" (each
  non-blank line is a segment). nltk sentence segmentation is degenerate on
  code (a whole code block is 1-3 "sentences"), so the pilot uses "line".
- Candidate trials are batched (num_return_sequences) for throughput; the
  first accepted candidate in sample order is kept, which preserves
  rejection-sampling semantics (candidates are i.i.d.).
- The valid-region permutation is drawn on a CPU torch.Generator regardless
  of the compute device (torch RNG streams differ between CPU and CUDA;
  MarkLLM uses the compute device, which would make detection
  device-dependent).
- Sampling parameters follow this pilot's Qwen settings (temperature 0.7,
  top_p 0.8, top_k 20), not MarkLLM's OPT-specific settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList


@dataclass
class SemStampConfig:
    gamma: float = 0.25          # fraction of valid LSH regions
    margin_m: float = 0.01       # min |cos| to every hyperplane
    dimension_d: int = 3         # number of LSH hyperplanes -> 2^d regions
    prime_P: int = 15485863
    N_max: int = 100             # max trials per segment
    z_threshold: float = 3.0
    embedder_path: str = "AbeHou/SemStamp-c4-sbert"
    lsh_rand_seed: int = 1234    # nearpy RandomBinaryProjections rand_seed
    segmenter: str = "line"      # "line" | "nltk"

    def to_dict(self) -> dict:
        return asdict(self)


class SBERTLSH:
    """SBERT embedder + random-hyperplane LSH.

    Replicates MarkLLM's SBERTLSHModel: nearpy RandomBinaryProjections
    draws its hyperplane normals as numpy.random.RandomState(seed).randn(d, dim)
    and hashes sign(normals @ v) to a bit string (bit i = 1 iff dot > 0).
    """

    def __init__(self, config: SemStampConfig, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer, models

        self.config = config
        word_embedding_model = models.Transformer(config.embedder_path)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
        self.embedder = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device=device
        )
        self.embedder.eval()
        self.dim = self.embedder.get_sentence_embedding_dimension()
        rand = np.random.RandomState(config.lsh_rand_seed)
        self.normals = rand.randn(config.dimension_d, self.dim)  # (d, dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.embedder.encode(texts, batch_size=32, show_progress_bar=False)
        )

    def signatures(self, embeds: np.ndarray) -> list[int]:
        """LSH region index per row (bit i of the binary string = sign of
        the projection on normal i, MSB first — matches nearpy int(s, 2))."""
        proj = embeds @ self.normals.T  # (n, d)
        sigs = []
        for row in proj:
            bits = "".join("1" if x > 0.0 else "0" for x in row)
            sigs.append(int(bits, 2))
        return sigs

    def min_abs_cos(self, embeds: np.ndarray) -> np.ndarray:
        """Per row: min over hyperplanes of |cosine(embed, normal)|."""
        e = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
        n = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)
        return np.abs(e @ n.T).min(axis=1)

    def hash_texts(self, texts: list[str]) -> list[int]:
        return self.signatures(self.embed(texts))


class SemStampUtils:
    """Valid-region mask derivation and z scoring."""

    def __init__(self, config: SemStampConfig):
        self.config = config
        self.rng = torch.Generator(device="cpu")
        self.n_bins = 2 ** config.dimension_d
        self.n_accept = int(self.n_bins * config.gamma)

    def accept_mask(self, lsh_seed: int) -> set[int]:
        """Valid region indices given the previous segment's signature."""
        self.rng.manual_seed(self.config.prime_P * lsh_seed)
        perm = torch.randperm(self.n_bins, generator=self.rng)
        return set(perm[: self.n_accept].tolist())

    def compute_z_score(self, n_valid: int, n_test: int) -> float:
        g = self.config.gamma
        return (n_valid - g * n_test) / math.sqrt(n_test * g * (1 - g))


def segment_text(text: str, mode: str) -> list[str]:
    """Split text into watermark segments (whitespace-stripped, non-empty)."""
    if mode == "line":
        return [l.strip() for l in text.split("\n") if l.strip()]
    if mode == "nltk":
        from nltk.tokenize import sent_tokenize

        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    raise ValueError(f"unknown segmenter: {mode}")


class StopAfterNewline(StoppingCriteria):
    """Per-sequence stop once the text generated after `prompt_len` contains
    a newline (segment boundary) — the line-mode analogue of MarkLLM's
    SentenceEndCriteria. Checks decoded text, so multi-char tokens that
    embed a newline (e.g. "\\n    ") are caught."""

    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        done = []
        for row in input_ids:
            text = self.tokenizer.decode(
                row[self.prompt_len :], skip_special_tokens=True
            )
            done.append("\n" in text)
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


class StopAfterSentence(StoppingCriteria):
    """MarkLLM's SentenceEndCriteria generalised to a batch: stop a sequence
    once its decoded continuation is more than one nltk sentence."""

    def __init__(self, tokenizer, prompt_len: int):
        from nltk.tokenize import sent_tokenize

        self._sent_tokenize = sent_tokenize
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        done = []
        for row in input_ids:
            text = self.tokenizer.decode(
                row[self.prompt_len :], skip_special_tokens=True
            )
            done.append(len(self._sent_tokenize(text)) > 1)
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


class SemStampGenerator:
    """Segment-level rejection sampling on top of a HF causal LM."""

    def __init__(
        self,
        model,
        tokenizer,
        lsh: SBERTLSH,
        config: SemStampConfig,
        gen_kwargs: dict,
        num_candidates: int = 8,
        segment_max_tokens: int = 64,
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lsh = lsh
        self.config = config
        self.utils = SemStampUtils(config)
        self.gen_kwargs = gen_kwargs  # sampling params only
        self.num_candidates = num_candidates
        self.segment_max_tokens = segment_max_tokens
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

    def _sample_candidates(self, context_ids: torch.LongTensor, seg_budget: int):
        """Sample num_candidates continuations of one segment each.

        Returns a list of dicts: {ids, text, hit_eos, has_newline}, where
        `ids` are the candidate's tokens cut at the first newline / EOS.
        """
        prompt_len = context_ids.shape[1]
        if self.config.segmenter == "line":
            stopper = StopAfterNewline(self.tokenizer, prompt_len)
        else:
            stopper = StopAfterSentence(self.tokenizer, prompt_len)
        with torch.no_grad():
            seqs = self.model.generate(
                context_ids,
                attention_mask=torch.ones_like(context_ids),
                max_new_tokens=seg_budget,
                num_return_sequences=self.num_candidates,
                stopping_criteria=StoppingCriteriaList([stopper]),
                pad_token_id=self.tokenizer.pad_token_id,
                **self.gen_kwargs,
            )
        cands = []
        for row in seqs:
            gen_ids = row[prompt_len:]
            # Sequences stopped early by the criterion are padded with
            # pad_token (== EOS for Qwen), so truncate at the FIRST eos-id and
            # decide `hit_eos` from the text, not from the presence of eos ids.
            eos_pos = (gen_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            maybe_eos = len(eos_pos) > 0
            if maybe_eos:
                gen_ids = gen_ids[: eos_pos[0]]
            if self.config.segmenter == "line":
                # cut at the first token whose cumulative decode contains "\n"
                cut = None
                for j in range(1, len(gen_ids) + 1):
                    if "\n" in self.tokenizer.decode(gen_ids[:j], skip_special_tokens=True):
                        cut = j
                        break
                if cut is not None:
                    gen_ids = gen_ids[:cut]
                    hit_eos = False  # stopped at a segment boundary
                else:
                    hit_eos = maybe_eos
            else:
                # nltk mode, MarkLLM behaviour: if the continuation ran past
                # the sentence end, drop the last token (it started the next
                # sentence); otherwise the model finished on its own.
                from nltk.tokenize import sent_tokenize

                text_full = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                if len(sent_tokenize(text_full)) > 1:
                    gen_ids = gen_ids[:-1]
                    hit_eos = False
                else:
                    hit_eos = maybe_eos
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            cands.append(
                {
                    "ids": gen_ids,
                    "text": text,
                    "hit_eos": hit_eos,
                    "has_newline": "\n" in text,
                }
            )
        return cands

    def generate(self, chat_text: str) -> dict:
        """Generate one watermarked completion for a chat-templated prompt."""
        enc = self.tokenizer(chat_text, return_tensors="pt").to(self.device)
        context_ids = enc.input_ids
        prompt_len = context_ids.shape[1]

        lsh_seed = self.lsh.hash_texts([chat_text])[0]
        accept = self.utils.accept_mask(lsh_seed)

        total_trials = 0
        maxedout_segments = 0
        n_segments = 0
        n_blank_segments = 0
        finished = False
        segment_log = []  # per accepted non-blank segment: diagnostics

        while not finished:
            gen_so_far = context_ids.shape[1] - prompt_len
            seg_budget = min(self.segment_max_tokens, self.max_new_tokens - gen_so_far)
            if seg_budget <= 0:
                break
            current_trials = 0
            chosen = None
            chosen_sig = None
            status = "accepted"
            distinct_sigs = set()  # candidate diversity seen for this segment
            while chosen is None:
                cands = self._sample_candidates(context_ids, seg_budget)
                nonblank = [c for c in cands if c["text"].strip()]
                if nonblank:
                    embeds = self.lsh.embed([c["text"].strip() for c in nonblank])
                    sigs = self.lsh.signatures(embeds)
                    margins = self.lsh.min_abs_cos(embeds)
                    distinct_sigs.update(sigs)
                k = 0
                for c in cands:
                    if not c["text"].strip():
                        # blank line / empty continuation: no semantic content —
                        # accept as-is, do not test or advance the seed chain
                        # (detection skips blank lines symmetrically).
                        if len(c["ids"]) == 0:
                            finished = True  # generated nothing: stop (MarkLLM)
                        chosen = c
                        status = "blank"
                        break
                    current_trials += 1
                    total_trials += 1
                    if margins[k] >= self.config.margin_m and sigs[k] in accept:
                        chosen = c
                        chosen_sig = sigs[k]
                        break
                    k += 1
                if chosen is None and current_trials >= self.config.N_max:
                    # give up: keep the last candidate (MarkLLM fallback)
                    chosen = nonblank[-1] if nonblank else cands[-1]
                    if chosen["text"].strip():
                        chosen_sig = sigs[len(nonblank) - 1]
                    maxedout_segments += 1
                    status = "maxedout"
            if finished and len(chosen["ids"]) == 0:
                break
            context_ids = torch.cat(
                [context_ids, chosen["ids"].unsqueeze(0).to(self.device)], dim=1
            )
            if chosen["text"].strip():
                n_segments += 1
                segment_log.append(
                    {
                        "text": chosen["text"].strip(),
                        "sig": chosen_sig,
                        "seed_before": lsh_seed,
                        "in_valid_region": chosen_sig in accept
                        if chosen_sig is not None
                        else None,
                        "status": status,
                        "trials": current_trials,
                        "distinct_candidate_sigs": len(distinct_sigs),
                    }
                )
            else:
                n_blank_segments += 1
            if chosen_sig is not None:
                lsh_seed = chosen_sig
                accept = self.utils.accept_mask(lsh_seed)
            if chosen["hit_eos"]:
                finished = True
            if context_ids.shape[1] - prompt_len >= self.max_new_tokens:
                finished = True

        gen_ids = context_ids[0, prompt_len:]
        completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return {
            "completion": completion,
            "num_gen_tokens": int(gen_ids.shape[0]),
            "sampling_stats": {
                "total_trials": total_trials,
                "n_segments": n_segments,
                "n_blank_segments": n_blank_segments,
                "maxedout_segments": maxedout_segments,
                "avg_trials_per_segment": total_trials / n_segments if n_segments else None,
            },
            "segment_log": segment_log,
        }


class SemStampDetector:
    """Segment-chain z-test detector (mirrors MarkLLM's detect_watermark):
    the first segment seeds the chain; segments 2..n are tested."""

    def __init__(self, lsh: SBERTLSH, config: SemStampConfig):
        self.lsh = lsh
        self.config = config
        self.utils = SemStampUtils(config)

    def detect(self, text: str) -> dict:
        segments = segment_text(text, self.config.segmenter)
        if len(segments) < 2:
            return {
                "z_score": None,
                "n_segments": len(segments),
                "n_tested": 0,
                "valid_fraction": None,
                "is_watermarked": False,
            }
        sigs = self.lsh.hash_texts(segments)
        n_valid = 0
        lsh_seed = sigs[0]
        for sig in sigs[1:]:
            if sig in self.utils.accept_mask(lsh_seed):
                n_valid += 1
            lsh_seed = sig
        n_test = len(segments) - 1
        z = self.utils.compute_z_score(n_valid, n_test)
        return {
            "z_score": z,
            "n_segments": len(segments),
            "n_tested": n_test,
            "n_valid": n_valid,
            "valid_fraction": n_valid / n_test,
            "is_watermarked": bool(z > self.config.z_threshold),
        }
