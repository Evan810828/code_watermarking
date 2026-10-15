"""SeqMark mean-centred rejection sampling for complete programs.

Reference: https://github.com/nle18/seqmark, revision
3c1ae8f9988e025e3d564839817a15a85bdcf0ff, watermark/correct_watermark.py;
https://arxiv.org/abs/2601.11629, Algorithms 1 and 2.

The released demonstration truncates to one natural-language sentence. This
adapter treats the extracted program as one sequence. It retains the released
Gaussian projections, prompt-seeded region permutation, mean subtraction,
first-accepted candidate, last-candidate fallback, and Bernoulli z-score.
Detection reuses the generation centre, matching CorrectWatermarkLM's
save_mean_embs=True default. Centre estimation uses 50 samples, as specified
in paper Section 7.3 and Table 7. Each sequence is encoded separately, matching
the reference's numerical operations. No execution verdict is used to select
or centre a generation. Dataset-specific generation parameters remain the
runner's responsibility; these defaults do not establish the paper's selected
MBPP configuration.
"""

from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np
import torch


UPSTREAM_REVISION = "3c1ae8f9988e025e3d564839817a15a85bdcf0ff"


@dataclass(frozen=True)
class SeqMarkConfig:
    embedder: str = "google/embeddinggemma-300m"
    gamma: float = 0.25
    lsh_dim: int = 3
    hash_key: int = 12345
    center_samples: int = 50
    center_temperature: float = 1.2
    max_trials: int = 50
    candidate_batch_size: int = 1
    detector_center: str = "generation_cache"
    z_threshold: float = 3.0
    mask_device: str = "cuda"

    def __post_init__(self):
        if not 1 <= self.lsh_dim <= 16:
            raise ValueError("lsh_dim must be between 1 and 16")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must be strictly between zero and one")
        if self.detector_center not in ("generation_cache", "independent_resample"):
            raise ValueError("unknown detector centre policy")
        if min(self.center_samples, self.max_trials, self.candidate_batch_size) < 1:
            raise ValueError("sample counts must be positive")
        if self.center_temperature <= 0:
            raise ValueError("center_temperature must be positive")

    def to_dict(self):
        return {
            "algorithm": "SeqMark",
            "implementation": "port (complete-program sequence)",
            "upstream_revision": UPSTREAM_REVISION,
            "segmenter": "program",
            "center_extraction": "extracted_program_else_empty_string; no filtering",
            "encoder_batch_size": 1,
            "region_count_rule": "int(gamma * 2**lsh_dim), as in reference",
            "classification_flag": "green_region_membership",
            **asdict(self),
        }


def stream_seed(seed, dataset, task_id, stage):
    """Stable, separate random streams, independent of sharding and resume."""
    key = f"{seed}:{dataset}:{task_id}:{stage}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


class SeqMark:
    def __init__(self, encoder, config):
        self.encoder = encoder
        self.config = config
        self.normals = np.random.RandomState(config.hash_key).randn(
            config.lsh_dim, encoder.get_sentence_embedding_dimension()
        )
        self.rng = torch.Generator(device=config.mask_device)

    def embed(self, texts):
        # The reference calls encoder.encode([sequence]) for every centre
        # sample and candidate. Batch-dependent rounding matters when mean
        # subtraction leaves a vector very close to zero.
        embeddings = np.concatenate([
            np.asarray(self.encoder.encode(
                [text], batch_size=1, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=False,
            ))
            for text in texts
        ], axis=0)
        if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
            raise ValueError("encoder returned invalid embeddings")
        return embeddings

    def signatures(self, embeddings, center=None):
        embeddings = np.asarray(embeddings)
        if center is not None:
            embeddings = embeddings - np.asarray(center)
        # One dot product per vector matches NearPy's operation and bit order.
        return [int("".join("1" if p > 0 else "0"
                            for p in self.normals.dot(e)), 2)
                for e in embeddings]

    def accept_mask(self, prompt):
        signature = self.signatures(self.embed([prompt]))[0]
        self.rng.manual_seed(self.config.hash_key * signature)
        perm = torch.randperm(2 ** self.config.lsh_dim,
                              device=self.config.mask_device, generator=self.rng)
        return set(perm[:int(self.config.gamma * len(perm))].tolist())

    def estimate_center(self, samples, extract_program):
        # Every sampled completion contributes exactly once. Failed extraction
        # is represented by an empty string, rather than unbounded resampling,
        # prose contamination, or selecting candidates using test outcomes.
        programs = [extract_program(s["completion"]) for s in samples]
        embeddings = self.embed([p if p is not None else "" for p in programs])
        # Preserve the reference's (samples, 1, embedding_dim) reduction.
        return np.mean(embeddings[:, None, :], axis=0)[0], sum(p is None for p in programs)

    def cached_center(self, center):
        # CorrectWatermarkLM serializes c_embedding.tolist() and restores it
        # with np.array(...), which produces a float64 centre during detection.
        return np.array(np.asarray(center).tolist())

    def detect(self, program, center, accept_mask):
        if not program or not program.strip():
            return None
        signature = self.signatures(self.embed([program]), center)[0]
        green = signature in accept_mask
        gamma = self.config.gamma
        z = (int(green) - gamma) / math.sqrt(gamma * (1 - gamma))
        return {
            "z_score": z,
            "is_watermarked": bool(green),
            "passes_z_threshold": z > self.config.z_threshold,
            "num_segments": 1,
            "num_green_segments": int(green),
            "lsh_signature": signature,
            "accepted_regions": sorted(accept_mask),
        }

    def generate(self, sample, center, accept_mask, extract_program):
        """sample(count, trial_offset) supplies independent, ordered draws."""
        audit = []
        physical_samples = 0
        for offset in range(0, self.config.max_trials, self.config.candidate_batch_size):
            count = min(self.config.candidate_batch_size, self.config.max_trials - offset)
            candidates = sample(count, offset)
            if len(candidates) != count:
                raise ValueError("generation returned the wrong number of candidates")
            physical_samples += count
            programs = [extract_program(c["completion"]) for c in candidates]
            signatures = self.signatures(
                self.embed([p if p is not None else "" for p in programs]), center
            )
            for i, (candidate, program, signature) in enumerate(zip(candidates, programs, signatures)):
                accepted = program is not None and signature in accept_mask
                audit.append({"trial": offset + i + 1,
                              "completion_sha256": hashlib.sha256(candidate["completion"].encode()).hexdigest(),
                              "extractable": program is not None,
                              "signature": signature, "accepted": accepted})
                if accepted or offset + i + 1 == self.config.max_trials:
                    return candidate, {
                        "trials": offset + i + 1,
                        "generated_candidates": physical_samples,
                        "accepted": accepted,
                        "fallback": not accepted,
                        "candidate_audit": audit,
                    }
        raise AssertionError("rejection sampler did not return a candidate")
