"""Code-watermark (Guo & Cheng, NeurIPS 2025) inference on any benchmark.

For each task, generates a watermarked completion (trained pairwise-rank
greenlist + learned entropy switch, wm/codewm port of ../code-watermark) and
an unwatermarked baseline, runs z-score detection on both, and saves
everything to out/codewm/<dataset>/. Checkpoints after every batch and
resumes from an existing output file. Mirrors run_sweet.py; detection here
needs only the small trained CodeModel (never the teacher LLM), and — as in
the upstream evaluator — scores the completion token ids alone.

Three-step pattern (IMPLEMENTATION.md): dataloader module -> watermark impl (wm/) ->
global inference helpers (utils).

Defaults follow the upstream repo (run.sh / main.py): delta 2.0, gamma 0.5
(pair selection fixes the green fraction), switch_threshold **-10** (gate
always open — the released default), context_width 2 (must match the
checkpoint), z 4. The CodeModel checkpoint comes from train_codewm.py and
defaults to **best_rank.pt**.

Three corrections landed 2026-07-24 after the reproduction round; runs made
before then are not comparable to runs made after (their `meta.watermark`
records which settings they used):
  1. switch_threshold defaulted to 0.7, taken from the README's WLLM example
     rather than the released run.sh/main.py value of -10. It gated ~90% of
     tokens away, leaving 3-7 scored tokens per task, so z could not reach 4.
  2. the CodeModel default was best.pt, whose total-val-loss pick is
     switch-dominated (alpha_switch=8) and fires long before the ranking — the
     watermark's actual signal — converges. best_rank.pt is the right pick.
  3. detection scored the raw generated ids, i.e. the markdown fence and any
     surrounding prose as well as the code. With the gate open those non-code
     tokens inflate green_fraction on watermarked and unwatermarked alike.
     Detection now scores the extracted program.
Reporting changed with them: the headline is TPR@FPR from the ROC (the paper's
metric, against the benchmark's own reference solution) rather than the
z>threshold flip rate, which is near-0 by construction on short completions.
`python utils.py summarize-codewm` prints both for every run.

With those in place this one protocol reaches the paper's own numbers on the
paper's own models — no separate reproduction path needed. Measured on
deepseek-coder-1.3b-instruct against the benchmark's reference solutions (the
comparison the paper makes; this repo now scores against the paired
unwatermarked completion instead, see utils "unified evaluation protocol"):
MBPP TPR@5% 42.7% / AUROC 0.797 against the paper protocol's 47.8% / 0.798. The
residue is protocol difference (chat prompt, 1 sample at T=0.7 vs a completion
prompt and 20 at T=0.2), not a missing capability.

Usage (single GPU):
    python run_codewm.py --dataset humaneval
    python run_codewm.py --dataset humaneval --limit 4 --batch-size 2  # smoke
    python run_codewm.py --dataset mbpp \
        --model deepseek-ai/deepseek-coder-1.3b-instruct --trust-remote-code \
        --code-model /scratch.../repro_deepseek-coder-1.3b-instruct/codemodel_cw2/best_rank.pt

--variant acw-s runs the paper's no-WPN variant (upstream --wm sweetcode,
wm/codewm/acws.py): pairing from the LLM's own logits, a true-entropy gate
(--entropy-threshold, default 1.2), and detection that reruns the LLM — no
--code-model needed. Default is `acw`, the full method.
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
    "CODEWM_CKPT", str(Path(WORK_DIR_DEFAULT) / "codemodel_cw2" / "best_rank.pt")
)


def load_code_model(args, device, model):
    """CodeModel checkpoint + config, vocab-checked against the LLM."""
    from wm.codewm import CodeModel, CodeWmConfig

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

    # the epoch the loaded file actually holds (best.pt = val-loss pick,
    # best_rank.pt = ranking-quality pick; see train_codewm.py)
    ckpt_meta["ckpt_epoch"] = (
        ckpt_meta.get("best_rank_epoch")
        if Path(args.code_model).name == "best_rank.pt"
        else ckpt_meta["best_epoch"]
    )

    code_model = CodeModel({"vocab_size": config.vocab_size}).to(device)
    code_model.load_state_dict(torch.load(args.code_model, map_location=device))
    code_model.eval()
    return code_model, config, ckpt_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=dataloader.DATASETS, required=True)
    ap.add_argument("--variant", choices=["acw", "acw-s"], default="acw",
                    help="acw: the full method (trained CodeModel, LLM-free "
                    "detection); acw-s: the paper's no-WPN variant (pairing from "
                    "the LLM's own logits, true-entropy gate, detection reruns "
                    "the LLM — no --code-model needed)")
    ap.add_argument("--entropy-threshold", type=float, default=1.2,
                    help="acw-s gate (upstream run.sh/main.py default)")
    ap.add_argument("--allow-comments", action="store_true",
                    help="drop dataloader.NO_COMMENTS_RULE from the system "
                    "prompt (the comments-allowed control cell)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    utils.add_common_args(ap)
    # upstream defaults (run.sh / README run example)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=2.0)
    # -10 is the RELEASED default (upstream run.sh + main.py + load_processors):
    # sigmoid(x) > -10 always, so the gate is permanently open and every token is
    # embedded and scored. The 0.7 used before came from the README's WLLM
    # example line and gated ~90% of tokens away, leaving 3-7 scored tokens per
    # task -> z capped at ~sqrt(7) < 4 -> the z>4 rate could not exceed ~0.
    ap.add_argument("--switch-threshold", type=float, default=-10.0)
    ap.add_argument("--z-threshold", type=float, default=4.0)
    ap.add_argument("--hash-key", type=int, default=15485863)
    ap.add_argument("--code-model", default=CKPT_DEFAULT,
                    help="trained CodeModel checkpoint (train_codewm.py)")
    ap.add_argument("--rescore", metavar="RESULT_JSON",
                    help="re-detect an existing run's stored completions under "
                    "the current protocol and write <stem>.rescored.json; no "
                    "generation, and the watermark config is taken from the file")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.rescore:
        src = Path(args.rescore)
        data = json.loads(src.read_text())
        wm = data["meta"].get("watermark", {})
        for line in utils.rescore_args_from_meta(
            args, data["meta"], ("gamma", "delta", "hash_key", "switch_threshold",
                                 "entropy_threshold", "z_threshold", "variant")):
            print(f"  config from file -> {line}")
        if wm.get("code_model"):
            args.code_model = wm["code_model"]
        model_name = data["meta"].get("model", args.model)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=args.trust_remote_code)

        if args.variant == "acw-s":
            # ACW-s detection reruns the teacher LLM
            print(f"Loading {model_name} ...")
            _, model = utils.load_hf_model(
                model_name, device, trust_remote_code=args.trust_remote_code)
            from wm.codewm.acws import AcwSConfig, AcwSDetector

            model_cfg = getattr(model.config, "text_config", model.config)
            acws_detector = AcwSDetector(AcwSConfig(
                vocab_size=model_cfg.vocab_size, gamma=args.gamma, delta=args.delta,
                hash_key=args.hash_key, entropy_threshold=args.entropy_threshold,
                z_threshold=args.z_threshold), model)
            detector = None
        else:
            # ACW detection is LLM-free: the small CodeModel is all it needs
            from types import SimpleNamespace

            from transformers import AutoConfig
            from wm.codewm import CodeDetector

            shim = SimpleNamespace(config=AutoConfig.from_pretrained(
                model_name, trust_remote_code=args.trust_remote_code))
            code_model, config, _ = load_code_model(args, device, shim)
            detector = CodeDetector(config, code_model)
            acws_detector = None

        @torch.no_grad()
        def detect_program(text, prompt=None):
            if not text or not text.strip():
                return None
            ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
            if acws_detector is not None:
                return None if ids.shape[0] < 2 else acws_detector.detect(ids.to(device), 1)
            return detector.detect(ids)

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
    tasks = ds.load_tasks(limit=args.limit, shard=args.shard, num_shards=args.num_shards)
    max_new_tokens = args.max_new_tokens or ds.MAX_NEW_TOKENS
    print(f"Loaded {len(tasks)} {ds.NAME} tasks")

    tag = utils.model_tag(args.model)
    # comments-allowed control cell gets its own segment; the default name is
    # unchanged from the existing port runs so they still resume
    cseg = "comments_" if args.allow_comments else ""
    vseg = "acws_" if args.variant == "acw-s" else ""
    out_path = (
        Path(args.output)
        if args.output
        else utils.out_dir("codewm", ds.NAME)
        / f"codewm_{ds.NAME}_port_{vseg}d{args.delta:g}_{cseg}{tag}.json"
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
    tokenizer, model = utils.load_hf_model(args.model, device, padding_side="left", trust_remote_code=args.trust_remote_code)

    if args.variant == "acw-s":
        # no-WPN variant: pairing from the LLM's own logits, true-entropy gate;
        # detection reruns the LLM (wm/codewm/acws.py)
        from wm.codewm.acws import AcwSConfig, AcwSDetector, AcwSLogitsProcessor

        model_cfg = getattr(model.config, "text_config", model.config)
        config = AcwSConfig(
            vocab_size=model_cfg.vocab_size,
            gamma=args.gamma,
            delta=args.delta,
            hash_key=args.hash_key,
            entropy_threshold=args.entropy_threshold,
            z_threshold=args.z_threshold,
        )
        wm_processor = LogitsProcessorList([AcwSLogitsProcessor(config)])
        acws_detector = AcwSDetector(config, model)
        wm_meta = {
            "algorithm": "ACW-s (Guo & Cheng, NeurIPS 2025 — no-WPN variant)",
            "implementation": "wm/codewm/acws.py port of upstream --wm sweetcode",
            "variant": "acw-s",
            **config.to_dict(),
        }
    else:
        from wm.codewm import CodeDetector, CodeLogitsProcessor

        code_model, config, ckpt_meta = load_code_model(args, device, model)
        wm_processor = LogitsProcessorList([CodeLogitsProcessor(config, code_model)])
        detector = CodeDetector(config, code_model)
        wm_meta = {
            "algorithm": "ACW / CodeWM (Guo & Cheng, NeurIPS 2025)",
            "implementation": "port of ../code-watermark (wm/codewm)",
            "variant": "acw",
            "code_model": str(args.code_model),
            "code_model_ckpt_epoch": ckpt_meta["ckpt_epoch"],
            **config.to_dict(),
        }

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
        "system_prompt": dataloader.system_prompt(ds, args.allow_comments),
        "allow_comments": args.allow_comments,
        # what the z-test was computed over. Files written before 2026-07-24
        # carry no such key and scored the raw generated ids (fence + prose
        # included); the two are not poolable, so resume and the summary both
        # key off this.
        "detection_scope": "extracted_code",
        "seed": args.seed,
        "started_at": prior_meta.get("started_at", time.strftime("%Y-%m-%d %H:%M:%S")),
    }
    utils.assert_resumable(prior_meta, meta, out_path)

    @torch.no_grad()
    def detect_code(text: str | None, prompt: str | None = None):
        """Score the generated PROGRAM, re-tokenized — not the raw completion.

        The chat template wraps code in a markdown fence and the model often adds
        prose around it. With the released gate (always open) those non-code
        tokens get scored too, which inflates green_fraction on watermarked and
        unwatermarked alike and crushes the separation the z-test lives on
        (measured: 0.754 vs 0.722 on OpenCoder, against 0.607 vs 0.520 when only
        the code is scored). Upstream scores the completion's code only.
        """
        if not text or not text.strip():
            return None
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if args.variant == "acw-s":
            # ACW-s reruns the teacher LLM and hashes on the PRECEDING token, so
            # position 0 has no context: score from index 1 (there is no prompt
            # prefix to strip here — the program is scored standalone).
            if ids.shape[0] < 2:
                return None
            return acws_detector.detect(ids, 1)
        return detector.detect(ids)

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
            code = utils.extract_program(completion)
            outs.append(
                {
                    "completion": completion,
                    "extracted_code": code,
                    "num_gen_tokens": int(gen_ids.shape[0]),
                    # no fenced block -> no program -> nothing to score. Falling
                    # back to the raw completion here would reintroduce exactly
                    # the fence/prose contamination this scoping removes, and
                    # would do so at a rate that differs between runs, silently
                    # making two runs measure different mixtures.
                    "detection": detect_code(code),
                }
            )
        return outs

    def save():
        with open(out_path, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)

    t0 = time.time()
    for start in range(0, len(tasks), args.batch_size):
        batch = tasks[start : start + args.batch_size]
        chat_texts = [
            utils.task_chat_text(tokenizer, ds, t, args.allow_comments) for t in batch
        ]
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
    print(f"Done. n={s['n']} | z>{args.z_threshold} rate={s['detect_rate_watermarked']:.1%} "
          f"| AUROC={s['auroc']} TPR@1%={utils.fmt_pct(s['tpr@1pct'])} "
          f"TPR@5%={utils.fmt_pct(s['tpr@5pct'])} -> {out_path}")


if __name__ == "__main__":
    main()
