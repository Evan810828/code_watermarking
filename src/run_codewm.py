"""Code-watermark (Guo & Cheng, NeurIPS 2025) inference on any benchmark.

For each task, generates a watermarked completion (trained pairwise-rank
greenlist + learned entropy switch, wm/codewm port of ../code-watermark) and
an unwatermarked baseline, runs z-score detection on both, and saves
everything to out/codewm/<dataset>/. Checkpoints after every batch and
resumes from an existing output file. Mirrors run_sweet.py; detection here
needs only the small trained CodeModel (never the teacher LLM), and — as in
the upstream evaluator — scores the completion token ids alone.

Three-step pattern (CLAUDE.md): dataloader module -> watermark impl (wm/) ->
global inference helpers (utils).

Defaults follow the upstream repo (run.sh / README): delta 2.0, gamma 0.5
(pair selection fixes the green fraction), switch_threshold 0.7,
context_width 2 (README gen_mbpp recipe; must match the checkpoint), z 4.
The CodeModel checkpoint comes from train_codewm.py.

Usage (single GPU):
    python run_codewm.py --dataset humaneval
    python run_codewm.py --dataset humaneval --limit 4 --batch-size 2  # smoke
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import LogitsProcessorList

import dataloader
import utils

# Trained CodeModel checkpoint (train_codewm.py). CODEWM_WORK_DIR relocates
# the training work dir (default: <repo>/out/codewm_train, gitignored);
# --code-model / CODEWM_CKPT points directly at a checkpoint.
WORK_DIR_DEFAULT = os.environ.get(
    "CODEWM_WORK_DIR", str(Path(__file__).resolve().parents[1] / "out" / "codewm_train")
)
CKPT_DEFAULT = os.environ.get(
    "CODEWM_CKPT", str(Path(WORK_DIR_DEFAULT) / "codemodel_cw2" / "best.pt")
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=dataloader.DATASETS, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    utils.add_common_args(ap)
    # upstream defaults (run.sh / README run example)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--switch-threshold", type=float, default=0.7)
    ap.add_argument("--z-threshold", type=float, default=4.0)
    ap.add_argument("--hash-key", type=int, default=15485863)
    ap.add_argument("--code-model", default=CKPT_DEFAULT,
                    help="trained CodeModel checkpoint (train_codewm.py)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = dataloader.get(args.dataset)
    tasks = ds.load_tasks(limit=args.limit, shard=args.shard, num_shards=args.num_shards)
    max_new_tokens = args.max_new_tokens or ds.MAX_NEW_TOKENS
    print(f"Loaded {len(tasks)} {ds.NAME} tasks")

    tag = utils.model_tag(args.model)
    out_path = (
        Path(args.output)
        if args.output
        else utils.out_dir("codewm", ds.NAME) / f"codewm_{ds.NAME}_port_{tag}.json"
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

    from wm.codewm import CodeDetector, CodeLogitsProcessor, CodeModel, CodeWmConfig

    ckpt_meta = json.loads((Path(args.code_model).parent / "meta.json").read_text())
    config = CodeWmConfig(
        vocab_size=ckpt_meta["vocab_size"],
        gamma=args.gamma,
        delta=args.delta,
        hash_key=args.hash_key,
        context_width=ckpt_meta["context_width"],
        switch_threshold=args.switch_threshold,
        entropy_threshold=ckpt_meta["entropy_threshold"],
        z_threshold=args.z_threshold,
    )
    model_cfg = getattr(model.config, "text_config", model.config)
    assert config.vocab_size == model_cfg.vocab_size, "checkpoint/model vocab mismatch"

    code_model = CodeModel({"vocab_size": config.vocab_size}).to(device)
    code_model.load_state_dict(torch.load(args.code_model, map_location=device))
    code_model.eval()

    wm_processor = LogitsProcessorList([CodeLogitsProcessor(config, code_model)])
    detector = CodeDetector(config, code_model)

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
            "algorithm": "CodeWM (Guo & Cheng, NeurIPS 2025)",
            "implementation": "port of ../code-watermark (wm/codewm)",
            "code_model": str(args.code_model),
            "code_model_best_epoch": ckpt_meta["best_epoch"],
            **config.to_dict(),
        },
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
            outs.append(
                {
                    "completion": completion,
                    "extracted_code": utils.extract_code(completion),
                    "num_gen_tokens": int(gen_ids.shape[0]),
                    # upstream evaluator detects on the completion ids alone
                    "detection": detector.detect(gen_ids),
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
