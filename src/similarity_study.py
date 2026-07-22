"""Semantic-diversity study: pairwise embedding similarity of repeated samples.

For N tasks per dataset (deterministic random subset, seed 42), draw M
UNWATERMARKED samples per task from the model (same sampling settings and
prompts as the inference runs, via the dataloader) and compute the pairwise
cosine similarity of the M generations' embeddings, using the same SBERT
embedder SemStamp uses (AbeHou/SemStamp-c4-sbert).

Also reports, per task, how many of the 2^d = 8 SemStamp LSH regions the M
samples cover — the semantic-entropy quantity that SemStamp's rejection
sampling needs to be > 1 region to embed a watermark at all.

Usage:
    python similarity_study.py --datasets humaneval mbpp
    python similarity_study.py --datasets livecodebench
    python similarity_study.py --datasets humaneval --n-tasks 2 --n-samples 4  # smoke
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import dataloader
import utils
from wm.semstamp import SBERTLSH, SemStampConfig


def pairwise_stats(embeds: np.ndarray) -> dict:
    e = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    sims = e @ e.T
    iu = np.triu_indices(len(e), k=1)
    pairs = sims[iu]
    return {
        "mean": float(pairs.mean()),
        "std": float(pairs.std()),
        "min": float(pairs.min()),
        "max": float(pairs.max()),
        "median": float(np.median(pairs)),
        "n_pairs": int(pairs.size),
        "matrix": np.round(sims, 4).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=dataloader.DATASETS,
                    default=["humaneval", "mbpp"])
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--select-seed", type=int, default=42)
    utils.add_common_args(ap)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} ...")
    tokenizer, model = utils.load_hf_model(args.model, device)

    ss_config = SemStampConfig()  # embedder + LSH defaults
    print(f"Loading embedder {ss_config.embedder_path} ...")
    lsh = SBERTLSH(ss_config, device=device)

    out_path = (
        Path(args.output)
        if args.output
        else utils.SIMILARITY_OUT / f"similarity_study_{'-'.join(args.datasets)}.json"
    )
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out = {
        "meta": {
            "model": args.model,
            "embedder": ss_config.embedder_path,
            "lsh_dimension_d": ss_config.dimension_d,
            "n_tasks_per_dataset": args.n_tasks,
            "n_samples": args.n_samples,
            "watermark": "none (unwatermarked sampling)",
            "generation": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens or "per-dataset default",
            },
            "seed": args.seed,
            "task_selection": f"random.Random({args.select_seed}).sample",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "tasks": [],
    }

    for dataset in args.datasets:
        ds = dataloader.get(dataset)
        tasks = ds.load_tasks()
        max_new_tokens = args.max_new_tokens or ds.MAX_NEW_TOKENS
        idx = sorted(random.Random(args.select_seed).sample(range(len(tasks)), args.n_tasks))
        print(f"{dataset}: task indices {idx}")

        for i in idx:
            task = tasks[i]
            tid = ds.task_id(task)
            enc = tokenizer(utils.task_chat_text(tokenizer, ds, task), return_tensors="pt").to(device)
            t0 = time.time()
            with torch.no_grad():
                seqs = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    num_return_sequences=args.n_samples,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = enc.input_ids.shape[1]
            completions = []
            for row in seqs:
                gen_ids = row[prompt_len:]
                eos = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos) > 0:
                    gen_ids = gen_ids[: eos[0]]
                completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

            codes = [utils.extract_code(c) or c for c in completions]
            emb_completion = lsh.embed(completions)
            emb_code = lsh.embed(codes)
            sigs = lsh.signatures(emb_completion)

            entry = {
                "dataset": dataset,
                "task_id": tid,
                "gen_seconds": round(time.time() - t0, 1),
                "completions": completions,
                "similarity_completion": pairwise_stats(emb_completion),
                "similarity_code": pairwise_stats(emb_code),
                "lsh_signatures": sigs,
                "n_distinct_lsh_regions": len(set(sigs)),
                "exact_duplicate_completions": len(completions) - len(set(completions)),
            }
            out["tasks"].append(entry)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(
                f"{dataset} {tid}: sim(completion) mean={entry['similarity_completion']['mean']:.4f} "
                f"min={entry['similarity_completion']['min']:.4f} | "
                f"sim(code) mean={entry['similarity_code']['mean']:.4f} | "
                f"LSH regions {entry['n_distinct_lsh_regions']}/8 | "
                f"exact dups {entry['exact_duplicate_completions']}/{args.n_samples} | {entry['gen_seconds']}s",
                flush=True,
            )

    out["meta"]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    agg = {}
    for dataset in args.datasets:
        ts = [t for t in out["tasks"] if t["dataset"] == dataset]
        agg[dataset] = {
            "mean_pairwise_sim_completion": float(
                np.mean([t["similarity_completion"]["mean"] for t in ts])
            ),
            "mean_pairwise_sim_code": float(
                np.mean([t["similarity_code"]["mean"] for t in ts])
            ),
            "mean_distinct_lsh_regions": float(
                np.mean([t["n_distinct_lsh_regions"] for t in ts])
            ),
        }
    out["meta"]["aggregate"] = agg
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(agg, indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
