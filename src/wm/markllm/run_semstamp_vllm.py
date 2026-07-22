"""SemStamp reproduction with the MarkLLM package + vLLM inference.

All watermark logic comes from the MarkLLM package (see adapter.py):
SBERT embedder + nearpy LSH via `SemStampUtils.SBERTLSHModel`, valid-region
masks via `SemStampUtils.rng` with MarkLLM's exact seeding, margin test and
detection chain per `SemStamp`. Only the LM sampling is swapped out: MarkLLM's
`generate_watermarked_text` supports OPT models only (it raises
NotImplementedError otherwise) and generates one candidate at a time with HF;
here candidates come from vLLM (n per call), and the rejection-sampling loop
is pooled across tasks so every vLLM step is a large batch.

Deviations from the package, all documented:
- MarkLLM's N_max fallback branch is unreachable in the released code (the
  `continue` on a rejected signature precedes it), i.e. it would resample
  forever on low-entropy text. This script enforces the fallback at
  N_max=100 trials (the SEMSTAMP.json value) as the SemStamp paper describes.
- Segmenter is pluggable (--segmenter sent|line). `sent` reproduces the
  package (nltk sent_tokenize, candidate cut after its first sentence like
  SentenceEndCriteria); `line` is the pilot's code-adapted variant.
- Whitespace-only segments are appended without a trial or seed update
  (they carry no semantic content; the detector skips them symmetrically).

Usage (from this directory):
    python run_semstamp_vllm.py --dataset humaneval --segmenter line
"""

import argparse
import json
import time

from adapter import MarkLLMSemStamp, load_pilot_utils

import dataloader

pilot_utils = load_pilot_utils()


def cut_candidate(text: str, stop_reason, finish_reason, segmenter: str):
    """Return (segment_text, hit_eos). Emulates MarkLLM's SentenceEndCriteria
    for sent mode; line mode relies on vLLM stop strings."""
    if segmenter == "line":
        hit_eos = finish_reason == "stop" and stop_reason is None
        return text, hit_eos
    from nltk.tokenize import sent_tokenize

    sents = sent_tokenize(text)
    if len(sents) >= 2:
        pos = text.find(sents[0])
        pos = pos + len(sents[0]) if pos >= 0 else len(text)
        return text[:pos], False
    hit_eos = finish_reason == "stop" and stop_reason is None
    return text, hit_eos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=dataloader.DATASETS, required=True)
    ap.add_argument("--segmenter", choices=["sent", "line"], default="line",
                    help="line is the default for all experiments")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default: the dataset's MAX_NEW_TOKENS")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--num-candidates", type=int, default=32)
    ap.add_argument("--resume", action="store_true", help="resume from an existing shard file")
    ap.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="override SEMSTAMP.json N_max (pilot verified 32 == 100 on "
        "near-deterministic code segments; used for the sent-mode runs "
        "whose candidates always fill the whole token budget)",
    )
    ap.add_argument("--segment-max-tokens", type=int, default=None)  # 64 line / 128 sent
    args = ap.parse_args()
    seg_cap = args.segment_max_tokens or (64 if args.segmenter == "line" else 128)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = dataloader.get(args.dataset)
    max_new_tokens = args.max_new_tokens or ds.MAX_NEW_TOKENS
    tasks = ds.load_tasks(limit=args.limit, shard=args.shard,
                          num_shards=args.num_shards if args.shard is not None else None)
    shard_tag = f".shard{args.shard}of{args.num_shards}" if args.shard is not None else ""

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 8192: one MBPP task's chat prompt alone is 3734 tokens (+512 generation)
    max_model_len = 8192
    llm = LLM(
        model=args.model,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.80,
        seed=args.seed,
        enable_prefix_caching=True,
    )
    overrides = {"N_max": args.n_max} if args.n_max else {}
    ss = MarkLLMSemStamp(device="cpu", **overrides)  # masks on CPU rng (as in the pilot)
    N_max = ss.config.N_max
    margin_m = ss.config.margin_m

    # FP control: unwatermarked completions from the pilot KGW run
    tag = pilot_utils.model_tag(args.model)
    unwm_by_id = {}
    kgw_path = pilot_utils.find_kgw_control(args.dataset, tag)
    if kgw_path.exists():
        with open(kgw_path) as f:
            for r in json.load(f)["results"]:
                unwm_by_id[r["task_id"]] = r["unwatermarked"]["completion"]

    states = []
    for t in tasks:
        prompt = pilot_utils.chat_text(tokenizer, ds.build_prompt(t), ds.SYSTEM_PROMPT)
        seed0 = ss.hash_texts([prompt])[0]  # MarkLLM: chain seeded by prompt hash
        states.append(
            {
                "task": t,
                "task_id": ds.task_id(t),
                "prompt": prompt,
                "prompt_tokens": len(tokenizer.encode(prompt)),
                "context": prompt,
                "gen_tokens": 0,
                "lsh_seed": seed0,
                "accept": ss.accept_mask(seed0),
                "seg_trials": 0,
                "seg_distinct": set(),
                "total_trials": 0,
                "n_segments": 0,
                "n_blank": 0,
                "maxedout": 0,
                "segment_log": [],
                "done": False,
            }
        )

    out_path = (
        pilot_utils.out_dir("semstamp", args.dataset, "markllm")
        / f"semstamp_vllm_{args.segmenter}_{args.dataset}_{tag}{shard_tag}.json"
    )

    if args.resume and out_path.exists():
        # restore per-task state from the checkpointed shard file (used after
        # preemption). Finished tasks are kept as-is; unfinished tasks resume
        # from their accepted prefix (mid-segment trial counts reset).
        with open(out_path) as f:
            prev = {r["task_id"]: r for r in json.load(f)["results"]}
        n_res = 0
        for s in states:
            r = prev.get(s["task_id"])
            if r is None:
                continue
            wm = r["watermarked"]
            if not wm["completion"]:
                continue
            s["context"] = s["prompt"] + wm["completion"]
            s["gen_tokens"] = wm["num_gen_tokens"]
            s["n_segments"] = wm["sampling_stats"]["n_segments"]
            s["n_blank"] = wm["sampling_stats"]["n_blank_segments"]
            s["maxedout"] = wm["sampling_stats"]["maxedout_segments"]
            s["total_trials"] = wm["sampling_stats"]["total_trials"]
            s["segment_log"] = wm["segment_log"]
            s["done"] = r.get("done", False)
            sigs = [x["sig"] for x in wm["segment_log"] if x["sig"] is not None]
            if sigs:
                s["lsh_seed"] = sigs[-1]
                s["accept"] = ss.accept_mask(sigs[-1])
            n_res += 1
        print(f"resumed {n_res} tasks ({sum(s['done'] for s in states)} already done)")

    def save(final=False):
        results = []
        for s in states:
            completion = s["context"][len(s["prompt"]) :]
            entry = {
                "task_id": s["task_id"],
                "dataset": ds.NAME,
                **ds.result_extras(s["task"]),
                "watermarked": {
                    "completion": completion,
                    "extracted_code": pilot_utils.extract_code(completion),
                    "num_gen_tokens": s["gen_tokens"],
                    "sampling_stats": {
                        "total_trials": s["total_trials"],
                        "n_segments": s["n_segments"],
                        "n_blank_segments": s["n_blank"],
                        "maxedout_segments": s["maxedout"],
                    },
                    "segment_log": s["segment_log"],
                },
                "done": s["done"],
            }
            if final:
                entry["watermarked"]["detection"] = ss.detect(completion, args.segmenter)
                if s["task_id"] in unwm_by_id:
                    entry["unwatermarked_control"] = {
                        "completion": unwm_by_id[s["task_id"]],
                        "detection": ss.detect(unwm_by_id[s["task_id"]], args.segmenter),
                    }
            results.append(entry)
        meta = {
            "model": args.model,
            "dataset": ds.NAME,
            "num_tasks": len(states),
            "engine": "vllm",
            "watermark": {
                "algorithm": "SemStamp (MarkLLM package logic, vLLM sampling)",
                "segmenter": args.segmenter,
                "gamma": ss.config.gamma,
                "margin_m": margin_m,
                "dimension_d": ss.config.dimension_d,
                "prime_P": ss.config.prime_P,
                "N_max": N_max,
                "threshold": ss.config.threshold,
                "embedder": ss.config.path_to_embedder,
            },
            "generation": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": max_new_tokens,
                "num_candidates": args.num_candidates,
                "segment_max_tokens": seg_cap,
                "seed": args.seed,
            },
        }
        if final:
            n_wm = sum(r["watermarked"]["detection"]["is_watermarked"] for r in results)
            ctrl = [r for r in results if "unwatermarked_control" in r]
            n_fp = sum(r["unwatermarked_control"]["detection"]["is_watermarked"] for r in ctrl)
            meta["summary"] = {
                "detect_rate_watermarked": n_wm / len(results),
                "false_positive_rate_unwatermarked": n_fp / len(ctrl) if ctrl else None,
            }
        with open(out_path, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        return meta.get("summary")

    t0 = time.time()
    round_idx = 0
    while True:
        # stop tasks that would overflow the model context on the next round
        for s in states:
            if not s["done"] and s["prompt_tokens"] + s["gen_tokens"] + seg_cap + 8 > max_model_len:
                s["done"] = True
        active = [s for s in states if not s["done"]]
        if not active:
            break
        sp = SamplingParams(
            n=args.num_candidates,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=seg_cap,
            seed=args.seed * 1000 + round_idx,  # new draws each retry round
            stop=["\n"] if args.segmenter == "line" else None,
            include_stop_str_in_output=True,
        )
        outs = llm.generate([s["context"] for s in active], sp, use_tqdm=False)

        # cut candidates, batch-embed all non-blank ones across the pool
        percand = []  # (state, [(seg_text, hit_eos), ...])
        embed_texts, embed_ref = [], []
        for s, out in zip(active, outs):
            cands = []
            for c in out.outputs:
                seg_text, hit_eos = cut_candidate(
                    c.text, c.stop_reason, c.finish_reason, args.segmenter
                )
                cands.append({"text": seg_text, "hit_eos": hit_eos, "sig": None, "margin": None})
            percand.append((s, cands))
            for j, c in enumerate(cands):
                if c["text"].strip():
                    embed_texts.append(c["text"].strip())
                    embed_ref.append((len(percand) - 1, j))
        if embed_texts:
            sigs, margins = ss.sigs_and_margins(ss.embed(embed_texts))
            for (pi, j), sig, marg in zip(embed_ref, sigs, margins):
                percand[pi][1][j]["sig"] = sig
                percand[pi][1][j]["margin"] = marg

        for s, cands in percand:
            chosen, chosen_sig, status = None, None, "accepted"
            last_examined = None
            for c in cands:
                if not c["text"].strip():
                    # blank segment: no semantic content, accept without a
                    # trial or seed update (detector skips blanks too)
                    chosen, status = c, "blank"
                    break
                if s["seg_trials"] >= N_max:
                    break
                s["seg_trials"] += 1
                s["total_trials"] += 1
                s["seg_distinct"].add(c["sig"])
                last_examined = c
                if c["margin"] >= margin_m and c["sig"] in s["accept"]:
                    chosen, chosen_sig = c, c["sig"]
                    break
            if chosen is None and s["seg_trials"] >= N_max:
                # N_max fallback (see module docstring)
                chosen = last_examined if last_examined is not None else cands[-1]
                chosen_sig = chosen["sig"]
                status = "maxedout"
                s["maxedout"] += 1
            if chosen is None:
                continue  # keep resampling this segment next round

            seg_text = chosen["text"]
            if seg_text:
                s["context"] += seg_text
                s["gen_tokens"] += len(tokenizer.encode(seg_text, add_special_tokens=False))
            if seg_text.strip():
                s["n_segments"] += 1
                s["segment_log"].append(
                    {
                        "text": seg_text.strip(),
                        "sig": chosen_sig,
                        "in_valid_region": chosen_sig in s["accept"] if chosen_sig is not None else None,
                        "status": status,
                        "trials": s["seg_trials"],
                        "distinct_candidate_sigs": len(s["seg_distinct"]),
                    }
                )
            else:
                s["n_blank"] += 1
            if chosen_sig is not None:
                s["lsh_seed"] = chosen_sig
                s["accept"] = ss.accept_mask(chosen_sig)
            s["seg_trials"] = 0
            s["seg_distinct"] = set()
            if chosen["hit_eos"] or s["gen_tokens"] >= max_new_tokens or not seg_text:
                s["done"] = True

        round_idx += 1
        if round_idx % 3 == 0:
            save()
        n_done = sum(s["done"] for s in states)
        print(
            f"round {round_idx}: {n_done}/{len(states)} done, "
            f"{len(active)} active, {time.time() - t0:.0f}s",
            flush=True,
        )

    summary = save(final=True)
    print(f"Done in {time.time() - t0:.0f}s. summary={json.dumps(summary)} -> {out_path}")


if __name__ == "__main__":
    main()
