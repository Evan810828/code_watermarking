"""KGW watermarked inference on any benchmark, with either implementation.

For each task, generates a watermarked completion (KGW logits processor) and
an unwatermarked baseline, runs z-score detection on both, and saves
everything to out/kgw/<impl>/. Checkpoints after every batch and resumes from
an existing output file.

Three-step pattern (CLAUDE.md): dataloader module -> watermark impl (wm/) ->
global inference helpers (utils).

Implementations:
    pilot     wm/kgw re-implementation (MarkLLM algorithm); detection scores
              the generated token ids with their true left context.
    official  jwkirchenbauer/lm-watermarking extended processor; detection on
              the decoded completion text.

Usage (single GPU):
    python run_kgw.py --dataset humaneval --impl pilot
    python run_kgw.py --dataset livecodebench --impl official
    python run_kgw.py --dataset humaneval --impl pilot --limit 4 --batch-size 2  # smoke
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import LogitsProcessorList

import dataloader
import utils


def build_pilot(args, tokenizer, model, device):
    from wm.kgw import KGWConfig, KGWDetector, KGWLogitsProcessor, KGWUtils

    model_cfg = getattr(model.config, "text_config", model.config)
    config = KGWConfig(
        vocab_size=model_cfg.vocab_size,
        gamma=args.gamma,
        delta=args.delta,
        hash_key=args.hash_key,
        prefix_length=args.prefix_length,
        z_threshold=args.z_threshold,
        device=device,
    )
    kgw_utils = KGWUtils(config)
    processor = KGWLogitsProcessor(config, kgw_utils)
    detector = KGWDetector(config, kgw_utils)

    def detect(completion, full_ids, prompt_len):
        # Detection: score generated tokens with their true left context
        # (prompt + generation). With prefix_length=1 the left pads in the
        # batched prompt cannot influence any scored position.
        return detector.detect(full_ids, scored_from=prompt_len)

    meta = {"algorithm": "KGW", "implementation": "pilot (wm/kgw)", **config.to_dict()}
    return processor, detect, meta


def build_official(args, tokenizer, model, device):
    from wm.official.kgw import OfficialKGW

    impl = OfficialKGW(
        tokenizer,
        device,
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme=args.seeding_scheme,
        z_threshold=args.z_threshold,
        ignore_repeated_ngrams=args.ignore_repeated_ngrams,
    )
    return impl.logits_processor, impl.detect, impl.meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=dataloader.DATASETS, required=True)
    ap.add_argument("--impl", choices=["pilot", "official"], required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    utils.add_common_args(ap)
    # KGW hyperparameters (delta=8 from the pilot delta sweep: at the MarkLLM
    # default 2 the watermark is undetectable on code, see README)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=8.0)
    ap.add_argument("--z-threshold", type=float, default=4.0)
    # pilot-only (MarkLLM defaults)
    ap.add_argument("--hash-key", type=int, default=15485863)
    ap.add_argument("--prefix-length", type=int, default=1)
    # official-only
    ap.add_argument("--seeding-scheme", default="lefthash")
    ap.add_argument("--ignore-repeated-ngrams", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = dataloader.get(args.dataset)
    tasks = ds.load_tasks(limit=args.limit)
    max_new_tokens = args.max_new_tokens or ds.MAX_NEW_TOKENS
    print(f"Loaded {len(tasks)} {ds.NAME} tasks")

    tag = utils.model_tag(args.model)
    out_path = (
        Path(args.output)
        if args.output
        else utils.out_dir("kgw", ds.NAME, args.impl) / f"kgw_{ds.NAME}_{args.impl}_{tag}.json"
    )
    out_path.parent.mkdir(exist_ok=True, parents=True)

    results = []
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)["results"]
        done = {r["task_id"] for r in results}
        tasks = [t for t in tasks if ds.task_id(t) not in done]
        print(f"Resuming: {len(results)} done, {len(tasks)} remaining")

    print(f"Loading {args.model} ...")
    tokenizer, model = utils.load_hf_model(args.model, device, padding_side="left")

    build = build_pilot if args.impl == "pilot" else build_official
    processor, detect, wm_meta = build(args, tokenizer, model, device)
    wm_processor = LogitsProcessorList([processor])

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        pad_token_id=tokenizer.pad_token_id,
    )

    meta = {
        "model": args.model,
        "dataset": ds.NAME,
        "num_tasks": len(results) + len(tasks),
        "watermark": wm_meta,
        "generation": {k: v for k, v in gen_kwargs.items() if k != "pad_token_id"},
        "seed": args.seed,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    def generate_batch(chat_texts: list[str], watermark: bool):
        enc = tokenizer(chat_texts, return_tensors="pt", padding=True).to(device)
        prompt_len = enc.input_ids.shape[1]
        with torch.no_grad():
            seqs = model.generate(
                **enc,
                logits_processor=wm_processor if watermark else None,
                **gen_kwargs,
            )
        outs = []
        for b in range(seqs.shape[0]):
            gen_ids = seqs[b, prompt_len:]
            # strip trailing pads after EOS
            eos_pos = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen_ids = gen_ids[: eos_pos[0] + 1]
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
            full_ids = torch.cat([enc.input_ids[b].to(device), gen_ids.to(device)])
            outs.append(
                {
                    "completion": completion,
                    "extracted_code": utils.extract_code(completion),
                    "num_gen_tokens": int(gen_ids.shape[0]),
                    "detection": detect(completion, full_ids, prompt_len),
                }
            )
        return outs

    def save():
        with open(out_path, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)

    t0 = time.time()
    for start in range(0, len(tasks), args.batch_size):
        batch = tasks[start : start + args.batch_size]
        chat_texts = [utils.task_chat_text(tokenizer, ds, t) for t in batch]
        wm_outs = generate_batch(chat_texts, watermark=True)
        base_outs = generate_batch(chat_texts, watermark=False)
        for task, wm, base in zip(batch, wm_outs, base_outs):
            results.append(
                {
                    "task_id": ds.task_id(task),
                    "dataset": ds.NAME,
                    "prompt_text": ds.build_prompt(task),
                    **ds.result_extras(task),
                    "watermarked": wm,
                    "unwatermarked": base,
                }
            )
        save()  # checkpoint after every batch
        done = min(start + args.batch_size, len(tasks))
        wm_z = [z for z in (r["watermarked"]["detection"].get("z_score") for r in results) if z is not None]
        base_z = [z for z in (r["unwatermarked"]["detection"].get("z_score") for r in results) if z is not None]
        print(
            f"[{done}/{len(tasks)}] {time.time() - t0:.0f}s | "
            f"mean z (wm)={sum(wm_z)/max(len(wm_z),1):.2f} (base)={sum(base_z)/max(len(base_z),1):.2f}",
            flush=True,
        )

    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    n_wm = sum(utils.detect_flag(r["watermarked"]["detection"]) for r in results)
    n_fp = sum(utils.detect_flag(r["unwatermarked"]["detection"]) for r in results)
    meta["summary"] = {
        "detect_rate_watermarked": n_wm / len(results),
        "false_positive_rate_unwatermarked": n_fp / len(results),
    }
    save()
    print(f"Done. TPR={n_wm}/{len(results)}, FP={n_fp}/{len(results)} -> {out_path}")


if __name__ == "__main__":
    main()
