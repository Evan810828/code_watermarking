"""SemStamp watermarked inference on any benchmark, with either implementation.

For each task, generates a watermarked completion via segment-level rejection
sampling and runs chain z-test detection on it. As a negative control, the
detector can also score *unwatermarked* completions of the same tasks from an
earlier KGW run (inline when --control-file exists, or standalone via
--mode detect-control). Checkpoints after every task and resumes.

Three-step pattern (CLAUDE.md): dataloader module -> watermark impl (wm/) ->
global inference helpers (utils).

Implementations (both default to the line segmenter; --segmenter nltk gives
the original sentence segmentation):
    pilot     wm/semstamp re-implementation (MarkLLM algorithm; batched
              candidate trials).
    official  authors' bohanhou14/SemStamp code via wm/MarkLLM_official/semstamp
              (documented shims incl. the line-mode criteria, see that module).

Usage (single GPU):
    python run_semstamp.py --dataset humaneval --impl pilot
    python run_semstamp.py --dataset livecodebench --impl official --max-trials 32 \
        --shard 0 --num-shards 6
    python run_semstamp.py --dataset livecodebench --impl official --mode detect-control \
        --control-file ../out/kgw/livecodebench/kgw_lcb_qwen3.5-4b.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

import dataloader
import utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=dataloader.DATASETS, required=True)
    ap.add_argument("--impl", choices=["pilot", "official"], required=True)
    ap.add_argument("--mode", choices=["generate", "detect-control"], default="generate")
    ap.add_argument("--rescore", metavar="RESULT_JSON",
                    help="re-detect an existing run's stored completions under "
                    "the current protocol and write <stem>.rescored.json; no "
                    "generation (the SBERT embedder is loaded, the LLM is not)")
    ap.add_argument("--control-file", default=None,
                    help="KGW run whose unwatermarked completions serve as the "
                    "FP control (default: the pilot KGW file for this dataset)")
    utils.add_common_args(ap)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    # optional split of one shard across GPUs; each sub resumes from its own
    # file and skips tasks already in the parent shard file
    ap.add_argument("--sub", type=int, default=None)
    ap.add_argument("--num-subs", type=int, default=None)
    # SemStamp hyperparameters (MarkLLM SEMSTAMP.json / official README settings)
    ap.add_argument("--gamma", type=float, default=0.25,
                    help="valid-region fraction (the official code calls it lmbd)")
    ap.add_argument("--dimension-d", type=int, default=3,
                    help="LSH hyperplanes (the official code calls it sp_dim)")
    ap.add_argument("--margin", type=float, default=None,
                    help="hyperplane margin; default 0.01 (pilot) / 0.02 (official)")
    ap.add_argument("--max-trials", type=int, default=100,
                    help="N_max rejection-sampling fallback")
    ap.add_argument("--z-threshold", type=float, default=3.0)
    ap.add_argument("--embedder", default="AbeHou/SemStamp-c4-sbert")
    ap.add_argument("--segmenter", choices=["line", "nltk"], default="line",
                    help="segment = non-blank code line (line, the default for "
                    "all experiments) or nltk sentence (the original)")
    # pilot-only
    ap.add_argument("--prime-p", type=int, default=15485863)
    ap.add_argument("--num-candidates", type=int, default=8)
    ap.add_argument("--segment-max-tokens", type=int, default=64)
    args = ap.parse_args()
    margin = args.margin if args.margin is not None else (0.01 if args.impl == "pilot" else 0.02)

    torch.manual_seed(args.seed + (args.shard or 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = dataloader.get(args.dataset)
    max_new_tokens = args.max_new_tokens or ds.MAX_NEW_TOKENS
    tag = utils.model_tag(args.model)
    out_root = utils.out_dir("semstamp", ds.NAME, args.impl)
    base_name = f"semstamp_{ds.NAME}_{args.impl}_{tag}"

    # ---------------------------------------------------------------- impls

    def build_pilot(model, tokenizer):
        from wm.semstamp import SBERTLSH, SemStampConfig, SemStampDetector, SemStampGenerator

        config = SemStampConfig(
            gamma=args.gamma,
            margin_m=margin,
            dimension_d=args.dimension_d,
            prime_P=args.prime_p,
            N_max=args.max_trials,
            z_threshold=args.z_threshold,
            embedder_path=args.embedder,
            segmenter=args.segmenter,
        )
        print(f"Loading embedder {args.embedder} ...")
        lsh = SBERTLSH(config, device=device)
        detector = SemStampDetector(lsh, config)
        generator = None
        if model is not None:
            gen_kwargs = dict(
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            generator = SemStampGenerator(
                model,
                tokenizer,
                lsh,
                config,
                gen_kwargs,
                num_candidates=args.num_candidates,
                segment_max_tokens=args.segment_max_tokens,
                max_new_tokens=max_new_tokens,
            )

        def generate(prompt):
            return generator.generate(prompt)

        def detect_program(text, prompt=None):
            """Unified protocol: segment and score the extracted program.

            SemStamp segments the text (line segmenter) and runs a chain z-test
            over segment embeddings, so scoping to the program changes which
            segments exist — prose sentences around the fence stop counting.
            That is the intent: every method scores the same span.
            """
            return detector.detect(text) if text and text.strip() else None

        wm_meta = {"algorithm": "SemStamp", "implementation": "pilot (wm/semstamp)",
                   **config.to_dict()}
        return generate, detect_program, wm_meta

    def build_official(model, tokenizer):
        from transformers import GenerationConfig

        from wm.MarkLLM_official.semstamp import OfficialSemStamp

        gen_config = None
        if model is not None:
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        impl = OfficialSemStamp(
            model,
            tokenizer,
            gen_config,
            device,
            embedder=args.embedder,
            sp_dim=args.dimension_d,
            lmbd=args.gamma,
            margin=margin,
            max_trials=args.max_trials,
            z_threshold=args.z_threshold,
            segmenter="line" if args.segmenter == "line" else "sent",
        )
        def detect_program(text, prompt=None):
            # official SemStamp anchors the chain's first segment on the prompt
            return impl.detect(prompt or "", text) if text and text.strip() else None

        return impl.generate, detect_program, impl.meta

    build = build_pilot if args.impl == "pilot" else build_official

    # ------------------------------------------------------------- rescore

    if args.rescore:
        # SemStamp detection needs the SBERT embedder, never the LLM.
        from transformers import AutoTokenizer

        src = Path(args.rescore)
        data = json.loads(src.read_text())
        tokenizer = AutoTokenizer.from_pretrained(data["meta"].get("model", args.model))
        _, detect_program, _ = build(None, tokenizer)
        tasks_by_id = {ds.task_id(t): t for t in ds.load_tasks()}

        def prompt_of(r):
            task = tasks_by_id.get(r["task_id"])
            return utils.task_chat_text(tokenizer, ds, task) if task else None

        # SemStamp generates no baseline of its own; the negative class comes
        # from a KGW run's unwatermarked completions. Older files stored it only
        # for the task_ids that happened to be in the control at the time (and
        # the HumanEval cells not at all), so rebuild it here — otherwise the
        # rescore has no negative class and every ROC comes out None.
        ctrl_path = (Path(args.control_file) if args.control_file
                     else utils.find_kgw_control(ds.NAME, utils.model_tag(args.model)))
        unwm_by_id = {}
        if ctrl_path.exists():
            with open(ctrl_path) as f:
                for r in json.load(f)["results"]:
                    unwm_by_id[r["task_id"]] = r["unwatermarked"]["completion"]
            print(f"  negative class from {ctrl_path.name} ({len(unwm_by_id)} tasks)")
        else:
            print(f"  WARNING: no control file at {ctrl_path}; ROC will be empty")
        src_results = []
        for r in data["results"]:
            r = dict(r)
            if "unwatermarked_control" not in r and r["task_id"] in unwm_by_id:
                r["unwatermarked_control"] = {"completion": unwm_by_id[r["task_id"]]}
            src_results.append(r)

        results = utils.rescore_results(src_results, detect_program, prompt_of)
        meta = utils.rescore_meta(data["meta"], src, results)
        dst = utils.rescore_path(src)
        dst.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
        s = meta["summary"]
        print(f"Rescored {len(results)} records -> {dst}")
        print(f"  AUROC={s['auroc']} TPR@1%={utils.fmt_pct(s['tpr@1pct'])} "
              f"TPR@5%={utils.fmt_pct(s['tpr@5pct'])}")
        return

    # ------------------------------------------------------- detect-control

    if args.mode == "detect-control":
        assert args.control_file, "--control-file required"
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        _, detect_program, wm_meta = build(None, tokenizer)
        with open(args.control_file) as f:
            control = json.load(f)["results"]
        tasks = {ds.task_id(t): t for t in ds.load_tasks()}
        out = []
        for i, r in enumerate(control):
            prompt = utils.task_chat_text(tokenizer, ds, tasks[r["task_id"]])
            det = detect_program(
                utils.extract_program(r["unwatermarked"]["completion"]), prompt
            )
            out.append({"task_id": r["task_id"], "detection": det})
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(control)}]", flush=True)
        n_fp = sum(bool(d["detection"]) and utils.detect_flag(d["detection"]) for d in out)
        meta = {
            "control_file": str(args.control_file),
            "watermark": wm_meta,
            "detection_scope": utils.DETECTION_SCOPE,
            "summary": {"false_positive_rate": n_fp / len(out), "n": len(out)},
        }
        out_path = out_root / f"semstamp_{ds.NAME}_{args.impl}_control_detection.json"
        with open(out_path, "w") as f:
            json.dump({"meta": meta, "results": out}, f, indent=2)
        print(f"FP={n_fp}/{len(out)} -> {out_path}")
        return

    # ------------------------------------------------------------- generate

    tasks = ds.load_tasks(limit=args.limit, shard=args.shard,
                          num_shards=args.num_shards if args.shard is not None else None)
    shard_tag = f".shard{args.shard}of{args.num_shards}" if args.shard is not None else ""
    parent_path = out_root / f"{base_name}{shard_tag}.json"
    if args.sub is not None and args.num_subs:
        tasks = tasks[args.sub :: args.num_subs]
        shard_tag += f".sub{args.sub}of{args.num_subs}"
    out_path = (
        Path(args.output) if args.output
        else out_root / f"{base_name}{shard_tag}.json"
    )
    out_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"{len(tasks)} {ds.NAME} tasks in this shard -> {out_path}")

    results = []
    prior_meta = {}
    if out_path.exists():
        with open(out_path) as f:
            prior = json.load(f)
        results, prior_meta = prior["results"], prior.get("meta", {})
    done = {r["task_id"] for r in results}
    if args.sub is not None and out_path != parent_path and parent_path.exists():
        with open(parent_path) as f:
            done |= {r["task_id"] for r in json.load(f)["results"]}
    if done:
        tasks = [t for t in tasks if ds.task_id(t) not in done]
        print(f"Resuming: {len(done)} done elsewhere/here, {len(tasks)} remaining")

    # negative control: unwatermarked completions from an earlier KGW run
    control_path = Path(args.control_file) if args.control_file else (
        utils.find_kgw_control(ds.NAME, tag)
    )
    unwm_by_id = {}
    if control_path.exists():
        with open(control_path) as f:
            for r in json.load(f)["results"]:
                unwm_by_id[r["task_id"]] = r["unwatermarked"]["completion"]

    print(f"Loading {args.model} ...")
    tokenizer, model = utils.load_hf_model(args.model, device)
    generate, detect_program, wm_meta = build(model, tokenizer)

    meta = {
        "model": args.model,
        "dataset": ds.NAME,
        "num_tasks": len(results) + len(tasks),
        "shard": shard_tag or None,
        "num_shards": args.num_shards,
        "watermark": wm_meta,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_candidates": args.num_candidates if args.impl == "pilot" else None,
            "segment_max_tokens": args.segment_max_tokens if args.impl == "pilot" else None,
        },
        "system_prompt": dataloader.system_prompt(ds),
        "detection_scope": utils.DETECTION_SCOPE,
        "seed": args.seed,
        "unwatermarked_source": str(control_path) if unwm_by_id else None,
        "started_at": prior_meta.get("started_at", time.strftime("%Y-%m-%d %H:%M:%S")),
    }
    utils.assert_resumable(prior_meta, meta, out_path)

    def save():
        with open(out_path, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)

    t0 = time.time()
    n_total = len(tasks)
    for i, task in enumerate(tasks):
        tid = ds.task_id(task)
        prompt = utils.task_chat_text(tokenizer, ds, task)
        t_task = time.time()
        gen = generate(prompt)
        wm = utils.branch_record(gen["completion"], None, detect_program, prompt)
        # keep the generator's own bookkeeping (sampling_stats, trials, ...)
        wm.update({k: v for k, v in gen.items()
                   if k not in ("completion", "extracted_code", "detection")})
        wm["gen_seconds"] = round(time.time() - t_task, 1)

        entry = {
            "task_id": tid,
            "dataset": ds.NAME,
            "prompt_text": ds.build_prompt(task),
            **ds.result_extras(task),
            "watermarked": wm,
        }
        if tid in unwm_by_id:
            entry["unwatermarked_control"] = utils.branch_record(
                unwm_by_id[tid], None, detect_program, prompt
            )
        results.append(entry)
        save()  # checkpoint after every task

        z = wm["detection"]["z_score"]
        stats = wm.get("sampling_stats", {})
        print(
            f"[{i + 1}/{n_total}] {time.time() - t0:.0f}s | {tid} | "
            f"{wm['gen_seconds']}s, trials={stats.get('total_trials')} | "
            f"z={z if z is None else round(z, 2)}",
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
