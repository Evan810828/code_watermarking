"""SWEET watermarked inference on any benchmark.

For each task, generates a watermarked completion (entropy-gated KGW logits
processor, wm/sweet port of hongcheki/sweet-watermark) and an unwatermarked
baseline, runs entropy-gated z-score detection on both, and saves everything
to out/sweet/<dataset>/. Checkpoints after every batch and resumes from an
existing output file. Mirrors run_kgw.py; the only structural difference is
that SWEET detection re-runs the model to get per-token entropy, so detection
scores each sample's token ids with their true UNPADDED left context (a
padded prompt would change early-token entropies).

Three-step pattern (IMPLEMENTATION.md): dataloader module -> watermark impl (wm/) ->
global inference helpers (utils).

Defaults: gamma/z-threshold/entropy-threshold follow the upstream repo's
main.py (0.5 / 4.0 / 0.5); delta follows this pilot's KGW delta sweep
(delta=8 — the upstream default 0.5 is far below the detectability floor the
sweep found on code, and SWEET biases only high-entropy positions anyway).

Usage (single GPU):
    python run_sweet.py --dataset humaneval
    python run_sweet.py --dataset livecodebench --entropy-threshold 0.9
    python run_sweet.py --dataset humaneval --limit 4 --batch-size 2  # smoke
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import LogitsProcessorList

import dataloader
import utils


def build_config(args, model):
    """SweetConfig from CLI args + the model's vocab (one definition, so the
    generate and rescore paths cannot drift apart)."""
    from wm.sweet import SweetConfig

    model_cfg = getattr(model.config, "text_config", model.config)
    return SweetConfig(
        vocab_size=model_cfg.vocab_size,
        gamma=args.gamma,
        delta=args.delta,
        hash_key=args.hash_key,
        entropy_threshold=args.entropy_threshold,
        z_threshold=args.z_threshold,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=dataloader.DATASETS, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    utils.add_common_args(ap)
    # SWEET hyperparameters (upstream main.py defaults; delta from the pilot
    # KGW delta sweep, see module docstring)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=8.0)
    ap.add_argument("--entropy-threshold", type=float, default=0.5)
    ap.add_argument("--z-threshold", type=float, default=4.0)
    ap.add_argument("--hash-key", type=int, default=15485863)
    ap.add_argument("--rescore", metavar="RESULT_JSON",
                    help="re-detect an existing run's stored completions under "
                    "the current protocol and write <stem>.rescored.json; no "
                    "generation, and the watermark config is taken from the file")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.rescore:
        # SWEET's detector recomputes per-token entropy from the model, so the
        # LLM is required here — but generation still is not.
        src = Path(args.rescore)
        data = json.loads(src.read_text())
        for line in utils.rescore_args_from_meta(
            args, data["meta"], ("gamma", "delta", "hash_key", "entropy_threshold",
                                 "z_threshold", "prefix_length")):
            print(f"  config from file -> {line}")
        model_name = data["meta"].get("model", args.model)
        print(f"Loading {model_name} ...")
        tokenizer, model = utils.load_hf_model(
            model_name, device, trust_remote_code=args.trust_remote_code)
        from wm.sweet import SweetDetector

        detector = SweetDetector(build_config(args, model), model)

        @torch.no_grad()
        def detect_program(text, prompt=None):
            if not text or not text.strip():
                return None
            ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
            if ids.shape[0] < 2:
                return None
            return detector.detect(ids.to(device), 0)

        results = utils.rescore_results(data["results"], detect_program)
        meta = utils.rescore_meta(data["meta"], src, results)
        dst = utils.rescore_path(src)
        dst.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
        s = meta["summary"]
        print(f"Rescored {len(results)} records -> {dst}")
        print(f"  AUROC={s['auroc']} TPR@1%={utils.fmt_pct(s['tpr@1pct'])} "
              f"TPR@5%={utils.fmt_pct(s['tpr@5pct'])}")
        return

    ds = dataloader.get(args.dataset)
    tasks = ds.load_tasks(limit=args.limit)
    max_new_tokens = args.max_new_tokens or ds.MAX_NEW_TOKENS
    print(f"Loaded {len(tasks)} {ds.NAME} tasks")

    tag = utils.model_tag(args.model)
    out_path = (
        Path(args.output)
        if args.output
        else utils.out_dir("sweet", ds.NAME) / f"sweet_{ds.NAME}_port_{tag}.json"
    )
    out_path.parent.mkdir(exist_ok=True, parents=True)

    results = []
    prior_meta = {}
    if out_path.exists():
        with open(out_path) as f:
            prior = json.load(f)
        results, prior_meta = prior["results"], prior.get("meta", {})
        done = {r["task_id"] for r in results}
        tasks = [t for t in tasks if ds.task_id(t) not in done]
        print(f"Resuming: {len(results)} done, {len(tasks)} remaining")

    print(f"Loading {args.model} ...")
    tokenizer, model = utils.load_hf_model(args.model, device, padding_side="left")

    config = build_config(args, model)
    from wm.sweet import SweetDetector, SweetLogitsProcessor

    wm_processor = LogitsProcessorList([SweetLogitsProcessor(config)])
    detector = SweetDetector(config, model)

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
        "watermark": {
            "algorithm": "SWEET",
            "implementation": "port of hongcheki/sweet-watermark (wm/sweet)",
            **config.to_dict(),
        },
        "generation": {k: v for k, v in gen_kwargs.items() if k != "pad_token_id"},
        "system_prompt": dataloader.system_prompt(ds),
        "detection_scope": utils.DETECTION_SCOPE,
        "seed": args.seed,
        "started_at": prior_meta.get("started_at", time.strftime("%Y-%m-%d %H:%M:%S")),
    }
    utils.assert_resumable(prior_meta, meta, out_path)

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
            outs.append(
                utils.branch_record(completion, int(gen_ids.shape[0]), detect_program)
            )
        return outs

    @torch.no_grad()
    def detect_program(text, prompt=None):
        """Unified protocol: score the extracted program standalone.

        SWEET recomputes per-token entropy from the model, and hashes each
        position on its predecessor, so scoring starts at index 1 (the detector
        clamps prefix_len to >=1 itself). Entropies here come from the program
        read on its own rather than from the prompt-conditioned distributions
        seen at generation time — the same trade the other methods make under
        this protocol, and the price of scoring a comparable span.
        """
        if not text or not text.strip():
            return None
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if ids.shape[0] < 2:
            return None
        return detector.detect(ids.to(device), 0)

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
        wm_z = utils.z_scores(r["watermarked"]["detection"] for r in results)
        base_z = utils.z_scores(r["unwatermarked"]["detection"] for r in results)
        print(
            f"[{done}/{len(tasks)}] {time.time() - t0:.0f}s | "
            f"mean z (wm)={sum(wm_z)/max(len(wm_z),1):.2f} (base)={sum(base_z)/max(len(base_z),1):.2f}",
            flush=True,
        )

    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["summary"] = utils.detection_summary(results)
    save()
    s = meta["summary"]
    print(f"Done. n={s['n']} | z>thr rate={s['detect_rate_watermarked']:.1%} | "
          f"AUROC={s['auroc']} TPR@1%={utils.fmt_pct(s['tpr@1pct'])} "
          f"TPR@5%={utils.fmt_pct(s['tpr@5pct'])} -> {out_path}")


if __name__ == "__main__":
    main()
