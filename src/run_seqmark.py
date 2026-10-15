"""SeqMark on the existing code benchmarks, with resumable per-task evidence.

Run from src/: python run_seqmark.py --dataset humaneval mbpp --backend vllm
Use --smoke --limit 2 for operational checks, excluded from result summaries.
Final JSONs are compatible with eval_pass1.py and utils.py summarize-methods.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
import uuid

import dataloader
import utils


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def dataset_names(args):
    return dataloader.DATASETS if args.dataset == ["all"] else args.dataset


def encoder_pipeline(embedder):
    return ("transformer_mean_pooling" if embedder == "AbeHou/SemStamp-c4-sbert"
            else "native_sentence_transformer")


def load_encoder(args):
    from sentence_transformers import SentenceTransformer, models
    path = args.embedder_path or args.embedder
    if encoder_pipeline(args.embedder) == "transformer_mean_pooling":
        # Match both existing SemStamp implementations, including truncation.
        transformer = models.Transformer(path)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(),
                                 pooling_mode_mean_tokens=True)
        encoder = SentenceTransformer(modules=[transformer, pooling], device=args.embedder_device)
    else:
        encoder = SentenceTransformer(path, device=args.embedder_device)
    encoder.eval()
    return encoder


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


class Sampler:
    def __init__(self, args):
        self.args = args
        if args.backend == "vllm":
            from vllm import LLM
            self.model = LLM(
                model=args.model_path or args.model,
                dtype="bfloat16", max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
                enable_prefix_caching=True,
                limit_mm_per_prompt={"image": 0, "video": 0},
                seed=args.seed,
            )
            self.tokenizer = self.model.get_tokenizer()
        else:
            self.tokenizer, self.model = utils.load_hf_model(
                args.model_path or args.model, "cuda", padding_side="left"
            )

    def sample(self, prompt, count, seed, temperature, max_tokens):
        if self.args.backend == "vllm":
            from vllm import SamplingParams
            params = SamplingParams(
                n=count, seed=seed, temperature=temperature,
                top_p=self.args.top_p, top_k=self.args.top_k,
                repetition_penalty=self.args.repetition_penalty,
                max_tokens=max_tokens,
            )
            outputs = self.model.generate([prompt], params, use_tqdm=False)[0].outputs
            return [{"completion": o.text, "num_gen_tokens": len(o.token_ids)}
                    for o in sorted(outputs, key=lambda o: o.index)]
        import torch
        torch.manual_seed(seed)
        encoded = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded, num_return_sequences=count, max_new_tokens=max_tokens,
                do_sample=True, temperature=temperature,
                top_p=self.args.top_p, top_k=max(0, self.args.top_k),
                repetition_penalty=self.args.repetition_penalty,
            )
        prompt_length = encoded.input_ids.shape[1]
        result = []
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        for row in outputs[:, prompt_length:].tolist():
            stop = next((i + 1 for i, token in enumerate(row) if token in eos), len(row))
            result.append({"completion": self.tokenizer.decode(row[:stop], skip_special_tokens=True),
                           "num_gen_tokens": stop})
        return result

    def sample_seeds(self, prompt, seeds, temperature, max_tokens):
        """Independent, ordered n=1 requests; batching only schedules the pool."""
        if self.args.backend != "vllm":
            return [self.sample(prompt, 1, s, temperature, max_tokens)[0] for s in seeds]
        from vllm import SamplingParams
        params = [SamplingParams(n=1, seed=s, temperature=temperature,
                                 top_p=self.args.top_p, top_k=self.args.top_k,
                                 repetition_penalty=self.args.repetition_penalty,
                                 max_tokens=max_tokens) for s in seeds]
        outputs = self.model.generate([prompt] * len(seeds), params, use_tqdm=False)
        return [{"completion": o.outputs[0].text,
                 "num_gen_tokens": len(o.outputs[0].token_ids)} for o in outputs]


def source_identity():
    paths = [Path(__file__), Path(utils.__file__)]
    paths += sorted((utils.REPO_DIR / "src/wm/seqmark").glob("*.py"))
    paths += sorted((utils.REPO_DIR / "src/dataloader").glob("*.py"))
    hashes = {str(p.relative_to(utils.REPO_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=utils.REPO_DIR, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=utils.REPO_DIR, text=True)
    identity = {"git_revision": revision, "git_status": dirty,
                "source_sha256": hashes, "source_fingerprint": digest(hashes)}
    snapshot = utils.OUT_DIR / "seqmark/source_snapshots" / identity["source_fingerprint"]
    for relative, expected in hashes.items():
        target = snapshot / relative
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError(f"source snapshot changed: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((utils.REPO_DIR / relative).read_bytes())
    if not (snapshot / "source_identity.json").exists():
        atomic_json(snapshot / "source_identity.json", identity)
    return identity


def evaluate_output(output, interpreter):
    """Use the existing benchmark evaluator, with immutable input identity."""
    data = json.loads(output.read_text())
    name = data["meta"]["dataset"]
    watermark_meta = data["meta"]["watermark"]
    impl_tag = utils.seqmark_impl_tag(watermark_meta["embedder"], watermark_meta.get("detector_center"))
    tag = "seqmark_" + impl_tag + "_" + utils.model_tag(data["meta"]["model"])
    verdict_path = output.parent / f"pass1_{tag}_{name}.json"
    provenance = verdict_path.with_suffix(".provenance.json")
    input_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    if provenance.exists():
        prior = json.loads(provenance.read_text())
        if (prior["input_sha256"] != input_hash or not verdict_path.exists()
                or prior["verdict_sha256"] != hashlib.sha256(verdict_path.read_bytes()).hexdigest()):
            raise ValueError(f"incompatible existing evaluation: {provenance}")
        return
    command = [interpreter, str(utils.REPO_DIR / "src/eval_pass1.py"),
               "--input", str(output), "--tag", tag, "--workers", "4"]
    subprocess.run(command, check=True)
    atomic_json(provenance, {
        "input": str(output), "input_sha256": input_hash,
        "verdict_sha256": hashlib.sha256(verdict_path.read_bytes()).hexdigest(),
        "command": command, "finished_at": now(),
    })


def merge_outputs(args):
    """Merge complete deterministic shards, evaluate, then update Results."""
    count = args.merge_shards
    names = dataset_names(args)
    tag = utils.model_tag(args.model)
    impl_tag = utils.seqmark_impl_tag(args.embedder, args.detector_center)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    ledger = utils.OUT_DIR / "seqmark/runs" / run_id
    ledger.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": run_id, "study": "baseline_watermarking_seqmark",
                "stage": "merge_and_evaluate", "status": "running",
                "started_at": now(), "ended_at": None, "code": source_identity(),
                "command": sys.argv, "resolved_arguments": vars(args),
                "environment": {"host": socket.gethostname(), "python": platform.python_version()},
                "source_runs": [], "datasets": []}
    atomic_json(ledger / "manifest.json", manifest)
    metrics = {}
    try:
        for name in names:
            ds = dataloader.get(name)
            tasks = ds.load_tasks()
            by_id = {}
            semantics = None
            root = utils.OUT_DIR / "seqmark" / name
            paths = [root / f"seqmark_{name}_{impl_tag}_{tag}.shard{i}of{count}.json" for i in range(count)]
            for shard, path in enumerate(paths):
                data = json.loads(path.read_text())
                meta = data["meta"]
                if meta.get("status") != "completed" or meta.get("operational_smoke"):
                    raise ValueError(f"incomplete/non-scientific shard: {path}")
                if meta["watermark"]["embedder"] != args.embedder:
                    raise ValueError(f"wrong encoder in shard: {path}")
                expected = [ds.task_id(t) for t in tasks[shard::count]]
                if [r["task_id"] for r in data["results"]] != expected:
                    raise ValueError(f"wrong task set in shard: {path}")
                current = {k: meta[k] for k in (
                    "model", "model_path", "dataset", "system_prompt", "enable_thinking",
                    "watermark", "embedder_path", "embedder_max_seq_length", "generation",
                    "detection_scope", "negative_class", "seed_policy", "source_fingerprint")}
                current["embedder_pipeline"] = meta.get("embedder_pipeline", "native_sentence_transformer")
                current["reference_draws"] = meta.get("reference_draws")
                current["parameter_selection"] = meta.get("parameter_selection")
                if semantics is not None and semantics != current:
                    raise ValueError(f"incompatible shard configurations: {path}")
                semantics = current
                manifest["source_runs"].append(meta["manifest"])
                for result in data["results"]:
                    if result["task_id"] in by_id:
                        raise ValueError("duplicate task across shards")
                    by_id[result["task_id"]] = result
            results = [by_id[ds.task_id(task)] for task in tasks]
            output = root / f"seqmark_{name}_{impl_tag}_{tag}.json"
            meta = {**semantics, "num_tasks": len(tasks), "status": "completed",
                    "run_id": run_id, "manifest": str(ledger / "manifest.json"),
                    "source_shards": [str(p) for p in paths], "finished_at": now(),
                    "data_fingerprint": digest(tasks), "task_ids": [ds.task_id(t) for t in tasks],
                    "seed": args.seed, "impl": "port", "summary": utils.detection_summary(results)}
            if output.exists():
                if json.loads(output.read_text())["results"] != results:
                    raise ValueError(f"refusing to replace completed results: {output}")
            else:
                atomic_json(output, {"meta": meta, "results": results})
            manifest["datasets"].append({"dataset": name, "data_fingerprint": meta["data_fingerprint"],
                                         "output": str(output), "config": semantics})
            atomic_json(ledger / "manifest.json", manifest)
            s = meta["summary"]
            metrics[name] = {k: s[k] for k in (
                "auroc", "tpr@1pct", "tpr@5pct", "n_pos", "n_unscoreable_watermarked",
                "mean_z_watermarked", "mean_z_unwatermarked")}
            if args.evaluate_python:
                evaluate_output(output, args.evaluate_python)
                verdicts = json.loads((root / f"pass1_seqmark_{impl_tag}_{tag}_{name}.json").read_text())
                for branch in ("watermarked", "unwatermarked"):
                    metrics[name][f"pass@1_{branch}"] = sum(v[branch]["passed"] for v in verdicts) / len(verdicts)
        atomic_json(ledger / "metrics.json", metrics)
        if args.write_readme:
            utils.summarize_seqmark(tag, write_readme=True, detector_center=args.detector_center)
        else:
            utils.summarize_seqmark(tag, detector_center=args.detector_center)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "aborted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["ended_at"] = now()
        atomic_json(ledger / "manifest.json", manifest)


class ReferenceDraws:
    """Share LLM calibration draws and controls across the encoder comparison."""

    def __init__(self, args, ds, contract):
        tag = utils.model_tag(args.model)
        reference_tag = utils.seqmark_impl_tag("google/embeddinggemma-300m", args.detector_center)
        self.path = utils.OUT_DIR / "seqmark" / ds.NAME / f"seqmark_{ds.NAME}_{reference_tag}_{tag}.json"
        data = json.loads(self.path.read_text())
        meta = data["meta"]
        if meta.get("status") != "completed" or meta.get("operational_smoke"):
            raise ValueError(f"reference must be a completed scientific split: {self.path}")
        for key in ("model", "model_path", "dataset", "system_prompt", "enable_thinking",
                    "generation", "detection_scope", "seed_policy"):
            if meta[key] != contract[key]:
                raise ValueError(f"reference {key} differs: {self.path}")
        expected_watermark = {k: v for k, v in contract["watermark"].items() if k != "embedder"}
        actual_watermark = {k: v for k, v in meta["watermark"].items() if k != "embedder"}
        if actual_watermark != expected_watermark:
            raise ValueError(f"reference watermark parameters differ: {self.path}")
        if meta["watermark"]["embedder"] != "google/embeddinggemma-300m":
            raise ValueError("the reference must use embeddinggemma-300m")
        full_tasks = ds.load_tasks()
        if ([r["task_id"] for r in data["results"]] != [ds.task_id(t) for t in full_tasks]
                or meta["data_fingerprint"] != digest(full_tasks)):
            raise ValueError(f"reference task set differs: {self.path}")
        self.results = {r["task_id"]: r for r in data["results"]}
        self.source_fingerprint = meta["source_fingerprint"]
        self.center_samples = contract["watermark"]["center_samples"]
        self.detector_center = contract["watermark"]["detector_center"]
        self.provenance = {
            "output": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "shared": ["generation_center_samples", "detection_center_samples", "unwatermarked_completion"],
            "candidate_draws": "regenerated with the same sampling parameters and per-batch seeds",
        }

    def task(self, tid, prompt, seeds):
        result = self.results[tid]
        path = utils.REPO_DIR / result["audit_file"]
        raw = path.read_bytes()
        audit = json.loads(raw)
        contract = json.loads((path.parent / "config.json").read_text())
        if (audit["result"] != result or audit["contract_sha256"] != digest(contract)
                or contract["source_fingerprint"] != self.source_fingerprint
                or audit["seeds"] != seeds
                or audit["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest()):
            raise ValueError(f"reference audit does not match the current prompt and seeds: {path}")
        for key in ("generation_center_samples", "detection_center_samples"):
            expected = (0 if key == "detection_center_samples" and self.detector_center == "generation_cache"
                        else self.center_samples)
            if len(audit[key]) != expected:
                raise ValueError(f"wrong reference centre sample count: {path}")
        return {
            "generation_center_samples": audit["generation_center_samples"],
            "detection_center_samples": audit["detection_center_samples"],
            "baseline": {k: result["unwatermarked"][k] for k in ("completion", "num_gen_tokens")},
            "provenance": {"audit_file": str(path), "sha256": hashlib.sha256(raw).hexdigest()},
        }


def run_dataset(args, ds, sampler, watermark, config, invocation, manifest_path):
    from wm.seqmark import stream_seed
    tasks = ds.load_tasks(limit=args.limit, shard=args.shard, num_shards=args.num_shards)
    max_tokens = args.max_new_tokens or ds.MAX_NEW_TOKENS
    tag = utils.model_tag(args.model)
    impl_tag = utils.seqmark_impl_tag(args.embedder, args.detector_center)
    shard_tag = f".shard{args.shard}of{args.num_shards}" if args.shard is not None else ""
    root = utils.OUT_DIR / "seqmark" / ("diagnostics" if args.smoke else ds.NAME)
    stem = f"seqmark_{ds.NAME}_{impl_tag}_{tag}{shard_tag}"
    if args.smoke:
        stem += f".{invocation['run_id']}"
    output = Path(args.output).resolve() if args.output else root / f"{stem}.json"
    state = output.with_suffix(".tasks")
    state.mkdir(parents=True, exist_ok=True)
    data_fingerprint = digest(tasks)
    contract = {
        "model": args.model, "model_path": args.model_path,
        "dataset": ds.NAME, "data_fingerprint": data_fingerprint,
        "task_ids": [ds.task_id(t) for t in tasks],
        "system_prompt": dataloader.system_prompt(ds),
        "enable_thinking": False,
        "watermark": config.to_dict(),
        "embedder_path": args.embedder_path,
        "embedder_max_seq_length": watermark.encoder.max_seq_length,
        "embedder_pipeline": encoder_pipeline(args.embedder),
        "generation": {"backend": args.backend, "dtype": "bfloat16",
                       "max_new_tokens": max_tokens, "do_sample": True,
                       "temperature": args.temperature, "top_p": args.top_p,
                       "top_k": args.top_k, "repetition_penalty": args.repetition_penalty,
                       "seed": args.seed},
        "detection_scope": utils.DETECTION_SCOPE,
        "negative_class": ("shared_reference_unwatermarked_same_model_prompt_and_backend"
                           if args.reuse_reference_draws
                           else "fresh_unwatermarked_same_model_prompt_and_backend"),
        "seed_policy": "sha256(seed:dataset:task_id:stage), first 32 bits",
        "source_fingerprint": invocation["code"]["source_fingerprint"],
        "operational_smoke": args.smoke,
        "parameter_selection": validate_selection(args),
    }
    reference = ReferenceDraws(args, ds, contract) if args.reuse_reference_draws else None
    contract["reference_draws"] = reference.provenance if reference else None
    contract_hash = digest(contract)
    contract_path = state / "config.json"
    if contract_path.exists():
        prior = json.loads(contract_path.read_text())
        if prior != contract:
            raise ValueError(f"refusing incompatible resume: {contract_path}")
    else:
        atomic_json(contract_path, contract)
    invocation["studies"].append({"dataset": ds.NAME, "config": contract,
                                  "output": str(output), "state": str(state)})
    atomic_json(manifest_path, invocation)
    results = []
    started = time.monotonic()
    for index, task in enumerate(tasks):
        tid = ds.task_id(task)
        task_path = state / (digest(tid)[:20] + ".json")
        if task_path.exists():
            saved = json.loads(task_path.read_text())
            if saved["contract_sha256"] != contract_hash or saved["result"]["task_id"] != tid:
                raise ValueError(f"incompatible task checkpoint: {task_path}")
            results.append(saved["result"])
            continue
        begin = time.monotonic()
        prompt = utils.task_chat_text(sampler.tokenizer, ds, task)
        seed_for = lambda stage: stream_seed(args.seed, ds.NAME, tid, stage)
        seeds = {stage: seed_for(stage) for stage in ("generation_center", "detection_center", "baseline")}
        assert len(set(seeds.values())) == len(seeds)
        shared = reference.task(tid, prompt, seeds) if reference else None
        center_draws = (shared["generation_center_samples"] if shared else
                        sampler.sample(prompt, config.center_samples,
                                       seeds["generation_center"], config.center_temperature, max_tokens))
        center, empty_center = watermark.estimate_center(center_draws, utils.extract_program)
        mask = watermark.accept_mask(prompt)
        candidate, sampling_audit = watermark.generate(
            lambda count, offset: sampler.sample(prompt, count, seed_for(f"rejection:{offset}"),
                                                   args.temperature, max_tokens),
            center, mask, utils.extract_program,
        )
        baseline = (shared["baseline"] if shared else
                    sampler.sample(prompt, 1, seeds["baseline"], args.temperature, max_tokens)[0])
        if config.detector_center == "generation_cache":
            detector_draws = []
            detector_center = watermark.cached_center(center)
            empty_detector_center = empty_center
        else:
            detector_draws = (shared["detection_center_samples"] if shared else
                              sampler.sample(prompt, config.center_samples,
                                             seeds["detection_center"], config.center_temperature, max_tokens))
            detector_center, empty_detector_center = watermark.estimate_center(detector_draws, utils.extract_program)
        detect = lambda program, prompt=None: watermark.detect(program, detector_center, mask)
        result = {
            "task_id": tid, "dataset": ds.NAME,
            "prompt_text": ds.build_prompt(task), **ds.result_extras(task),
            "watermarked": utils.branch_record(candidate["completion"], candidate["num_gen_tokens"], detect, prompt),
            "unwatermarked": utils.branch_record(baseline["completion"], baseline["num_gen_tokens"], detect, prompt),
            "audit_file": str(task_path),
        }
        audit = {
            "contract_sha256": contract_hash, "run_id": invocation["run_id"],
            "seeds": seeds, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "generation_center": center.tolist(), "detection_center": detector_center.tolist(),
            "generation_center_samples": center_draws, "detection_center_samples": detector_draws,
            "unextractable_generation_center_samples": empty_center,
            "unextractable_detection_center_samples": empty_detector_center,
            "sampling": sampling_audit, "seconds": time.monotonic() - begin,
            "reference_draws": shared["provenance"] if shared else None,
            "result": result,
        }
        atomic_json(task_path, audit)
        results.append(result)
        print(f"{ds.NAME} [{index + 1}/{len(tasks)}] task={tid} "
              f"seconds={audit['seconds']:.1f} trials={sampling_audit['trials']} "
              f"elapsed={time.monotonic() - started:.0f}s", flush=True)
    meta = {**contract, "impl": "port", "seed": args.seed, "num_tasks": len(tasks),
            "status": "completed", "finished_at": now(), "run_id": invocation["run_id"],
            "manifest": str(manifest_path), "summary": utils.detection_summary(results)}
    if output.exists():
        prior = json.loads(output.read_text())
        if prior["results"] != results:
            raise ValueError(f"refusing to overwrite completed results: {output}")
    else:
        atomic_json(output, {"meta": meta, "results": results})
    print(f"Completed {ds.NAME}: {len(results)} tasks -> {output}", flush=True)
    return output


SWEEP_DIMS = (2, 3, 4, 5)
SWEEP_GAMMAS = (0.1, 0.25, 0.5, 0.75)
SWEEP_RULE = ("maximize mean F1 across both encoders, using the released max-TPR/FPR "
              "threshold rule; ties: mean AUROC, fewer mean trials, smaller LSH dimension, "
              "smaller gamma; no test-split tuning")


def paper_classification(pos, neg, threshold=None):
    """Released classification rule, retaining finite, scoreable examples."""
    import numpy as np
    from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score
    pos, neg = [x for x in pos if x is not None], [x for x in neg if x is not None]
    if not pos or not neg:
        raise ValueError("classification requires both scoreable classes")
    labels = np.array([1] * len(pos) + [0] * len(neg))
    scores = np.array(pos + neg)
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        index = max((i for i in range(len(fpr)) if fpr[i] != 0),
                    key=lambda i: tpr[i] / fpr[i])
        threshold = float(thresholds[index])
    predictions = scores >= threshold
    return {"threshold": threshold, "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "fpr": float(predictions[len(pos):].mean()), "auroc": float(roc_auc_score(labels, scores)),
            "n_pos": len(pos), "n_neg": len(neg)}


def replay_sweep(watermark, prompt, center, candidates, baseline):
    """Replay all regions against the same ordered candidate pool and centre."""
    import math
    from wm.seqmark import SeqMark
    programs = [utils.extract_program(c["completion"]) for c in candidates]
    embeddings = watermark.embed([p if p is not None else "" for p in programs])
    base_program = utils.extract_program(baseline["completion"])
    base_embedding = watermark.embed([base_program or ""])
    detection_center = watermark.cached_center(center)
    settings = []
    for dim in SWEEP_DIMS:
        for gamma in SWEEP_GAMMAS:
            wm = SeqMark(watermark.encoder, replace(watermark.config, lsh_dim=dim, gamma=gamma))
            mask = wm.accept_mask(prompt)
            signatures = wm.signatures(embeddings, center)
            accepted = [p is not None and s in mask for p, s in zip(programs, signatures)]
            chosen = next((i for i, hit in enumerate(accepted) if hit), len(candidates) - 1)
            detected_signatures = wm.signatures(embeddings[[chosen]], detection_center)
            base_signature = wm.signatures(base_embedding, detection_center)[0]
            score = lambda green: (int(green) - gamma) / math.sqrt(gamma * (1 - gamma))
            settings.append({"lsh_dim": dim, "gamma": gamma,
                             "accepted_regions": sorted(mask), "candidate_signatures": signatures,
                             "selected_index": chosen, "trials": chosen + 1,
                             "accepted": accepted[chosen], "fallback": not accepted[chosen],
                             "watermarked_z": (score(detected_signatures[0] in mask)
                                               if programs[chosen] and programs[chosen].strip() else None),
                             "unwatermarked_z": (score(base_signature in mask)
                                                 if base_program and base_program.strip() else None)})
    return settings


def sweep_tasks(args):
    tasks = json.loads(Path(args.sweep_data).read_text())
    ids = [int(t["task_id"]) for t in tasks]
    test = dataloader.get("mbpp").load_tasks()
    if (len(tasks) != 100 or len(set(ids)) != 100 or not all(601 <= i <= 974 for i in ids)
            or set(ids).intersection(int(t["task_id"]) for t in test)
            or {t["text"].strip() for t in tasks}.intersection(t["text"].strip() for t in test)):
        raise ValueError("sweep must use 100 distinct MBPP training tasks, disjoint from test")
    return tasks


def run_sweep(args, sampler, watermark, config, invocation, manifest_path):
    from wm.seqmark import SeqMark, stream_seed
    ds = dataloader.get("mbpp")
    tasks = sweep_tasks(args)
    max_tokens = args.max_new_tokens or ds.MAX_NEW_TOKENS
    second_args = argparse.Namespace(**vars(args))
    second_args.embedder = utils.SEQMARK_EMBEDDERS[1]
    second_args.embedder_path = args.sweep_sbert_path
    print("Loading paired SBERT encoder ...", flush=True)
    second = SeqMark(load_encoder(second_args), replace(config, embedder=second_args.embedder))
    encoders = {args.embedder: watermark, second_args.embedder: second}
    root = utils.OUT_DIR / "seqmark/sweep"
    output = root / f"sweep.shard{args.shard}of{args.num_shards}.json"
    state = output.with_suffix(".tasks")
    contract = {"dataset": "mbpp_train", "data_fingerprint": digest(tasks),
                "task_ids": [ds.task_id(t) for t in tasks], "watermark": config.to_dict(),
                "source_fingerprint": invocation["code"]["source_fingerprint"],
                "model": args.model, "model_path": args.model_path,
                "embedder_paths": {args.embedder: args.embedder_path, second_args.embedder: args.sweep_sbert_path},
                "generation": {"backend": args.backend, "max_new_tokens": max_tokens,
                               "temperature": args.temperature, "top_p": args.top_p,
                               "top_k": args.top_k, "repetition_penalty": args.repetition_penalty,
                               "seed": args.seed},
                "grid": {"lsh_dim": SWEEP_DIMS, "gamma": SWEEP_GAMMAS},
                "selection_rule": SWEEP_RULE,
                "candidate_pool": "50 ordered independent n=1 requests, batched by vLLM; first accepted or last fallback",
                "shard": args.shard, "num_shards": args.num_shards}
    # Round-trip tuples before comparing resumable JSON contracts.
    contract = json.loads(json.dumps(contract))
    if (state / "config.json").exists():
        if json.loads((state / "config.json").read_text()) != contract:
            raise ValueError("incompatible sweep resume")
    else:
        atomic_json(state / "config.json", contract)
    invocation["studies"].append({"dataset": "mbpp_train", "config": contract, "output": str(output)})
    atomic_json(manifest_path, invocation)
    rows = []
    subset = tasks[args.shard::args.num_shards]
    for index, task in enumerate(subset):
        tid = ds.task_id(task)
        path = state / f"{tid}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["contract_sha256"] != digest(contract) or saved["result"]["task_id"] != tid:
                raise ValueError("incompatible sweep task checkpoint")
            rows.append(saved["result"])
            continue
        begin = time.monotonic()
        prompt = utils.task_chat_text(sampler.tokenizer, ds, task)
        seed_for = lambda stage: stream_seed(args.seed, "mbpp_train", tid, stage)
        center_draws = sampler.sample(prompt, config.center_samples, seed_for("generation_center"),
                                      config.center_temperature, max_tokens)
        seeds = [seed_for(f"rejection:{i}") for i in range(config.max_trials)]
        candidates = sampler.sample_seeds(prompt, seeds, args.temperature, max_tokens)
        baseline = sampler.sample(prompt, 1, seed_for("baseline"), args.temperature, max_tokens)[0]
        centers, results = {}, {}
        for name, wm in encoders.items():
            center, empty = wm.estimate_center(center_draws, utils.extract_program)
            centers[name] = {"generation_center": center.tolist(), "detection_center": wm.cached_center(center).tolist(),
                             "unextractable_center_samples": empty}
            results[name] = replay_sweep(wm, prompt, center, candidates, baseline)
        result = {"task_id": tid, "encoders": results, "audit_file": str(path)}
        atomic_json(path, {"contract_sha256": digest(contract), "run_id": invocation["run_id"],
                           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                           "generation_center_samples": center_draws, "detection_center_samples": [],
                           "candidate_pool": candidates, "candidate_seeds": seeds,
                           "center_seed": seed_for("generation_center"), "baseline_seed": seed_for("baseline"),
                           "baseline": baseline, "centers": centers, "result": result,
                           "seconds": time.monotonic() - begin})
        rows.append(result)
        print(f"sweep [{index + 1}/{len(subset)}] task={tid} seconds={time.monotonic()-begin:.1f}", flush=True)
    if output.exists():
        if json.loads(output.read_text())["results"] != rows:
            raise ValueError("refusing to replace completed sweep")
    else:
        atomic_json(output, {"meta": {**contract, "manifest": str(manifest_path), "status": "completed"}, "results": rows})
    return output


def select_sweep_result(args):
    tasks = sweep_tasks(args)
    root = utils.OUT_DIR / "seqmark/sweep"
    rows, contracts, sources = [], [], []
    for shard in range(args.merge_shards):
        path = root / f"sweep.shard{shard}of{args.merge_shards}.json"
        data = json.loads(path.read_text())
        if (data["meta"]["status"] != "completed"
                or data["meta"]["data_fingerprint"] != digest(tasks)
                or [r["task_id"] for r in data["results"]] != [int(t["task_id"]) for t in tasks[shard::args.merge_shards]]):
            raise ValueError(f"invalid sweep shard: {path}")
        contracts.append({k: v for k, v in data["meta"].items() if k not in ("shard", "manifest")})
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "manifest": data["meta"]["manifest"]})
        rows.extend(data["results"])
    if any(c != contracts[0] for c in contracts[1:]):
        raise ValueError("sweep shards have different contracts")
    metrics = []
    for index, (dim, gamma) in enumerate((d, g) for d in SWEEP_DIMS for g in SWEEP_GAMMAS):
        encoders = {}
        for name in utils.SEQMARK_EMBEDDERS:
            settings = [r["encoders"][name][index] for r in rows]
            assert all(s["lsh_dim"] == dim and s["gamma"] == gamma for s in settings)
            encoders[name] = {**paper_classification([s["watermarked_z"] for s in settings],
                                                    [s["unwatermarked_z"] for s in settings]),
                              "mean_trials": sum(s["trials"] for s in settings) / len(settings),
                              "fallback_count": sum(s["fallback"] for s in settings)}
        means = {"mean_" + key: sum(e[key] for e in encoders.values()) / 2 for key in ("f1", "auroc")}
        metrics.append({"lsh_dim": dim, "gamma": gamma, "encoders": encoders, **means,
                        "mean_trials": sum(e["mean_trials"] for e in encoders.values()) / 2})
    best = max(metrics, key=lambda m: (m["mean_f1"], m["mean_auroc"], -m["mean_trials"], -m["lsh_dim"], -m["gamma"]))
    selected = {"selection_rule": SWEEP_RULE, "contract": contracts[0], "sources": sources,
                "selected": best, "all_settings": metrics}
    path = root / "selected_config.json"
    if path.exists():
        if json.loads(path.read_text()) != selected:
            raise ValueError("refusing to replace frozen sweep selection")
    else:
        atomic_json(path, selected)
    print(json.dumps(best, indent=2))
    print(f"Frozen configuration: {path}", flush=True)
    return path


def select_sweep(args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    ledger = utils.OUT_DIR / "seqmark/runs" / run_id
    manifest = {"run_id": run_id, "study": "baseline_watermarking_seqmark",
                "stage": "sweep_selection", "status": "running", "started_at": now(),
                "ended_at": None, "command": sys.argv, "resolved_arguments": vars(args),
                "code": source_identity(), "environment": {"host": socket.gethostname(),
                                                            "python": platform.python_version()}}
    atomic_json(ledger / "manifest.json", manifest)
    try:
        path = select_sweep_result(args)
        result = json.loads(path.read_text())
        manifest.update(status="completed", output=str(path), source_runs=result["sources"])
        atomic_json(ledger / "metrics.json", {"selected": result["selected"], "all_settings": result["all_settings"]})
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["ended_at"] = now()
        atomic_json(ledger / "manifest.json", manifest)


def validate_selection(args):
    if not args.selected_config:
        return None
    path = Path(args.selected_config).resolve()
    selected = json.loads(path.read_text())
    contract = selected["contract"]
    for source in selected["sources"]:
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("sweep evidence changed after selection")
    for name in ("lsh_dim", "gamma"):
        if getattr(args, name) != selected["selected"][name]:
            raise ValueError(f"{name} differs from frozen sweep selection")
    for name in ("center_samples", "center_temperature", "max_trials", "hash_key", "detector_center", "candidate_batch_size"):
        if getattr(args, name) != contract["watermark"][name]:
            raise ValueError(f"{name} differs from sweep")
    for name in ("temperature", "top_p", "top_k", "repetition_penalty", "seed", "backend"):
        if getattr(args, name) != contract["generation"][name]:
            raise ValueError(f"{name} differs from sweep")
    if args.model != contract["model"] or args.model_path != contract["model_path"]:
        raise ValueError("generator differs from sweep")
    if args.embedder_path != contract["embedder_paths"][args.embedder]:
        raise ValueError("encoder snapshot differs from sweep")
    if (args.max_new_tokens or 2000) != contract["generation"]["max_new_tokens"]:
        raise ValueError("generation budget differs from sweep")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selection_rule": selected["selection_rule"],
            "threshold": selected["selected"]["encoders"][args.embedder]["threshold"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["all", *dataloader.DATASETS], nargs="+", required=True)
    utils.add_common_args(ap)
    ap.set_defaults(temperature=1.2, top_p=1.0, top_k=-1)
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--model-path", help="local snapshot of --model")
    ap.add_argument("--embedder", default="google/embeddinggemma-300m")
    ap.add_argument("--embedder-path", help="local snapshot of --embedder")
    ap.add_argument("--embedder-device", default="cpu")
    ap.add_argument("--reuse-reference-draws", action="store_true",
                    help="share Gemma's saved centre-sampling texts and controls for the encoder comparison")
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--lsh-dim", type=int, default=3)
    ap.add_argument("--hash-key", type=int, default=12345)
    ap.add_argument("--center-samples", type=int, default=50)
    ap.add_argument("--center-temperature", type=float, default=1.2)
    ap.add_argument("--repetition-penalty", type=float, default=1.05,
                    help="released class uses 1.05 for centre, candidate and control sampling")
    ap.add_argument("--max-trials", type=int, default=50)
    ap.add_argument("--candidate-batch-size", type=int, default=1)
    ap.add_argument("--detector-center", choices=("generation_cache", "independent_resample"),
                    default="generation_cache", help="reuse the generation centre, as in the released class")
    ap.add_argument("--z-threshold", type=float, default=3.0)
    ap.add_argument("--shard", type=int)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge-shards", type=int, help="merge this many completed shards, without loading models")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="operational diagnostic; excluded from Results")
    ap.add_argument("--evaluate-python", help="run the existing pass@1 evaluator with this interpreter")
    ap.add_argument("--write-readme", action="store_true", help="write both encoders after HumanEval/MBPP evaluation")
    ap.add_argument("--sweep-data", help="100 MBPP training tasks for the paired LSH/gamma sweep")
    ap.add_argument("--sweep-sbert-path", help="local SBERT snapshot, loaded alongside Gemma during the sweep")
    ap.add_argument("--select-sweep", action="store_true", help="merge sweep shards and freeze the best shared configuration")
    ap.add_argument("--selected-config", help="frozen sweep selection; assert every result-affecting parameter matches")
    args = ap.parse_args()
    if len(set(args.dataset)) != len(args.dataset) or ("all" in args.dataset and len(args.dataset) != 1):
        ap.error("--dataset must contain distinct dataset names or only all")
    if args.output and len(dataset_names(args)) != 1:
        ap.error("--output requires a single dataset")
    if args.reuse_reference_draws and args.embedder == "google/embeddinggemma-300m":
        ap.error("--reuse-reference-draws requires a different encoder from the Gemma reference")
    if args.num_shards < 1 or (args.shard is not None and not 0 <= args.shard < args.num_shards):
        ap.error("invalid shard range")
    if args.smoke and (args.limit is None or args.limit > 3):
        ap.error("--smoke requires --limit between 1 and 3")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if args.write_readme and (not args.evaluate_python or args.smoke or args.limit or args.shard is not None):
        ap.error("--write-readme requires evaluation and complete, unsharded splits")
    if args.select_sweep:
        if not args.sweep_data or not args.merge_shards:
            ap.error("--select-sweep requires --sweep-data and --merge-shards")
        select_sweep(args)
        return
    if args.sweep_data:
        if (args.dataset != ["mbpp"] or args.embedder != utils.SEQMARK_EMBEDDERS[0]
                or not args.sweep_sbert_path or args.shard is None or args.limit or args.smoke
                or args.reuse_reference_draws or args.evaluate_python or args.write_readme):
            ap.error("sweep requires both encoders, MBPP, complete training shards, and no test evaluation")
        if args.center_samples != 50 or args.max_trials != 50 or args.detector_center != "generation_cache":
            ap.error("the authorized sweep uses 50 centre samples, 50 trials, and cached centres")
    validate_selection(args)
    if args.merge_shards:
        if args.merge_shards < 1 or args.limit or args.shard is not None or args.smoke:
            ap.error("merging requires complete scientific shards")
        merge_outputs(args)
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("SeqMark benchmark generation requires a GPU allocation")
    from wm.seqmark import SeqMark, SeqMarkConfig
    config = SeqMarkConfig(**{name: getattr(args, name)
                              for name in asdict(SeqMarkConfig()) if hasattr(args, name)})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    ledger = utils.OUT_DIR / "seqmark" / ("diagnostics" if args.smoke else "runs") / run_id
    ledger.mkdir(parents=True, exist_ok=False)
    manifest_path = ledger / "manifest.json"
    invocation = {
        "run_id": run_id, "study": "baseline_watermarking_seqmark", "status": "running",
        "operational_smoke": args.smoke, "started_at": now(), "ended_at": None,
        "command": sys.argv, "resolved_arguments": vars(args),
        "code": source_identity(), "studies": [],
        "environment": {"host": socket.gethostname(), "python": platform.python_version(),
                        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "versions": {p: importlib.metadata.version(p)
                                     for p in ("torch", "transformers", "sentence-transformers", "numpy")}},
    }
    if args.backend == "vllm":
        invocation["environment"]["versions"]["vllm"] = importlib.metadata.version("vllm")
    atomic_json(manifest_path, invocation)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        print("Loading sequence encoder ...", flush=True)
        encoder = load_encoder(args)
        watermark = SeqMark(encoder, config)
        print("Loading language model ...", flush=True)
        sampler = Sampler(args)
        if args.sweep_data:
            output = run_sweep(args, sampler, watermark, config, invocation, manifest_path)
            invocation["status"] = "completed"
            return
        datasets = dataset_names(args)
        outputs = []
        for name in datasets:
            output = run_dataset(args, dataloader.get(name), sampler, watermark, config,
                                 invocation, manifest_path)
            outputs.append(output)
            if args.evaluate_python and not args.smoke:
                evaluate_output(output, args.evaluate_python)
        if not args.smoke:
            metrics = {}
            for output in outputs:
                data = json.loads(output.read_text())
                s = data["meta"]["summary"]
                metrics[data["meta"]["dataset"]] = {
                    k: s[k] for k in ("auroc", "tpr@1pct", "tpr@5pct", "n_pos",
                                     "n_unscoreable_watermarked", "mean_z_watermarked",
                                     "mean_z_unwatermarked")}
                if args.evaluate_python:
                    tag = "seqmark_" + utils.seqmark_impl_tag(args.embedder, args.detector_center) + "_" + utils.model_tag(args.model)
                    verdicts = json.loads((output.parent / f"pass1_{tag}_{data['meta']['dataset']}.json").read_text())
                    for branch in ("watermarked", "unwatermarked"):
                        metrics[data["meta"]["dataset"]][f"pass@1_{branch}"] = sum(v[branch]["passed"] for v in verdicts) / len(verdicts)
            atomic_json(ledger / "metrics.json", metrics)
        if args.write_readme:
            utils.summarize_seqmark(utils.model_tag(args.model), write_readme=True,
                                   detector_center=args.detector_center)
        invocation["status"] = "completed"
    except BaseException as exc:
        invocation["status"] = "aborted" if isinstance(exc, KeyboardInterrupt) else "failed"
        invocation["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        invocation["ended_at"] = now()
        atomic_json(manifest_path, invocation)


if __name__ == "__main__":
    main()
