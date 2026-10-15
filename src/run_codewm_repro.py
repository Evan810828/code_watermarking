"""CodeWM generation and evaluation under the original completion-prompt protocol.

Run from src/ with --dataset, --model, and --code-model. This protocol retains
20-sample pass@1 estimation and first-valid-sample detection for comparison
with the historical reproduction artifacts. It is distinct from run_codewm.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import dataloader  # noqa: E402
from wm.codewm import repro_protocol as repro  # noqa: E402
import utils  # noqa: E402
from wm.codewm import CodeDetector, CodeLogitsProcessor, CodeWmConfig, CodeModel  # noqa: E402


def load_code_model(args, device, model):
    """CodeModel checkpoint + config, vocab-checked against the LLM."""
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
    ckpt_meta["ckpt_epoch"] = (
        ckpt_meta.get("best_rank_epoch")
        if Path(args.code_model).name == "best_rank.pt"
        else ckpt_meta["best_epoch"]
    )
    code_model = CodeModel({"vocab_size": config.vocab_size}).to(device)
    code_model.load_state_dict(torch.load(args.code_model, map_location=device))
    code_model.eval()
    return code_model, config, ckpt_meta


def run(args) -> None:
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task = repro.TASKS[args.dataset]()
    docs = dataloader.slice_tasks(task.docs, args.limit, args.shard, args.num_shards)
    print(f"Loaded {len(docs)} {task.NAME} tasks (upstream repro protocol)")

    tag = utils.model_tag(args.model)
    shard_sfx = f".shard{args.shard}of{args.num_shards}" if args.shard is not None else ""
    out_path = (
        Path(args.output)
        if args.output
        else utils.out_dir("codewm", task.NAME)
        / f"codewm_{task.NAME}_repro_d{args.delta:g}_{tag}{shard_sfx}.json"
    )
    out_path.parent.mkdir(exist_ok=True, parents=True)

    results, prior_meta = [], {}
    if out_path.exists() and not args.verify_against:
        prior = json.loads(out_path.read_text())
        results, prior_meta = prior["results"], prior.get("meta", {})
        done = {r["task_id"] for r in results}
        docs = [d for d in docs if task.task_id(d) not in done]
        print(f"Resuming: {len(results)} done, {len(docs)} remaining")

    print(f"Loading {args.model} ...")
    tokenizer, model = utils.load_hf_model(
        args.model, device, trust_remote_code=args.trust_remote_code
    )
    code_model, config, ckpt_meta = load_code_model(args, device, model)
    wm_processor = LogitsProcessorList([CodeLogitsProcessor(config, code_model)])
    detector = CodeDetector(config, code_model)

    # upstream main.py appends the eos token to task.stop_words, so it also
    # participates in postprocess truncation
    if tokenizer.eos_token and tokenizer.eos_token not in task.stop_words:
        task.stop_words.append(tokenizer.eos_token)

    meta = {
        "model": args.model,
        "dataset": task.NAME,
        "impl": "repro",
        "num_tasks": len(results) + len(docs),
        "watermark": {
            "algorithm": "CodeWM (Guo & Cheng, NeurIPS 2025)",
            "implementation": "wm/codewm port, upstream evaluation protocol "
            "(src/wm/codewm/repro_protocol.py)",
            "variant": "wpn",
            "code_model": str(args.code_model),
            "code_model_ckpt_epoch": ckpt_meta["ckpt_epoch"],
            **config.to_dict(),
        },
        "generation": {**repro.GEN_KWARGS, "n_samples": args.n_samples,
                       "max_length": repro.MAX_LENGTH, "max_time": repro.MAX_TIME},
        "detection_negative_class": "human reference solutions (upstream)",
        "seed": args.seed,
        "started_at": prior_meta.get("started_at", time.strftime("%Y-%m-%d %H:%M:%S")),
    }
    if not args.verify_against:
        utils.assert_resumable(prior_meta, meta, out_path)

    def generate_samples(enc, watermark: bool, doc, prompt: str):
        stopping = repro.EndOfFunctionCriteria(
            enc.input_ids.shape[1], task.stop_words, tokenizer
        )
        with torch.no_grad():
            seqs = model.generate(
                **enc,
                num_return_sequences=args.n_samples,
                max_length=repro.MAX_LENGTH,
                max_time=repro.MAX_TIME,
                stopping_criteria=StoppingCriteriaList([stopping]),
                logits_processor=wm_processor if watermark else None,
                pad_token_id=tokenizer.pad_token_id,
                **repro.GEN_KWARGS,
            )
        prompt_len = enc.input_ids.shape[1]
        texts, raws = [], []
        for s in range(seqs.shape[0]):
            # upstream decodes with special tokens kept (eos is a stop word)
            decoded = tokenizer.decode(
                seqs[s, prompt_len:],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            raws.append(decoded)
            texts.append(task.postprocess_generation(prompt + decoded, doc))
        return texts, raws

    def detect_one(text: str, prompt: str):
        ids = repro.completion_ids_for_detection(tokenizer, prompt, text)
        if ids is None:
            return None
        return detector.detect(torch.tensor(ids))

    def detect_first_valid(texts: list[str], prompt: str):
        for i, text in enumerate(texts):
            det = detect_one(text, prompt)
            if det is None or det.pop("invalid", False):
                continue
            return det, i
        return None, None

    def save():
        out_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2))

    t0 = time.time()
    for di, doc in enumerate(docs):
        prompt = task.get_prompt(doc)
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=repro.MAX_LENGTH
        ).to(device)

        branches = {}
        for name, wm in (("watermarked", True), ("unwatermarked", False)):
            texts, raws = generate_samples(enc, wm, doc, prompt)
            det, det_idx = detect_first_valid(texts, prompt)
            branches[name] = {
                "completions": [t[len(prompt):] for t in texts],
                # pre-postprocess decode (diagnoses upstream's stop-word
                # truncation, e.g. MBPP's regex-split quirk)
                "completions_raw": raws,
                "detection": det,
                "detect_sample_idx": det_idx,
                **repro.pass_at_k(task, doc, texts),
            }
        human_det, _ = detect_first_valid([task.get_full_data(doc)], prompt)

        results.append({
            "task_id": task.task_id(doc),
            "dataset": task.NAME,
            "prompt_text": prompt,
            "valid": branches["watermarked"]["detection"] is not None
            and human_det is not None,
            **branches,
            "human": {"detection": human_det},
        })
        if not args.verify_against:
            save()  # checkpoint after every task
        valid = [r for r in results if r["valid"]]
        wm_z = [r["watermarked"]["detection"]["z_score"] for r in valid]
        print(f"[{di+1}/{len(docs)}] {time.time()-t0:.0f}s | valid {len(valid)} | "
              f"mean z (wm)={sum(wm_z)/max(len(wm_z),1):.2f} | "
              f"pass@1 (wm)={sum(r['watermarked']['pass@1'] for r in results)/len(results):.3f} "
              f"(base)={sum(r['unwatermarked']['pass@1'] for r in results)/len(results):.3f}",
              flush=True)

    if args.verify_against:
        verify(results, Path(args.verify_against))
        return

    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    valid = [r for r in results if r["valid"]]
    meta["summary"] = {
        **utils.detection_summary(valid),
        "num_valid": len(valid),
        "false_positive_rate_human": sum(
            utils.detect_flag(r["human"]["detection"]) for r in valid
        ) / max(len(valid), 1),
        "avg_pass1_watermarked": sum(r["watermarked"]["pass@1"] for r in results)
        / max(len(results), 1),
        "avg_pass1_unwatermarked": sum(r["unwatermarked"]["pass@1"] for r in results)
        / max(len(results), 1),
    }
    save()
    s = meta["summary"]
    print(f"Done. valid={len(valid)} | AUROC={s['auroc']} | "
          f"TPR@5%={utils.fmt_pct(s['tpr@5pct'])} -> {out_path}")


def verify(results: list, stored_path: Path) -> None:
    """Restoration check: every regenerated record must match the stored run."""
    stored = {r["task_id"]: r for r in json.loads(stored_path.read_text())["results"]}
    checked = mismatched = 0
    for r in results:
        old = stored.get(r["task_id"])
        if old is None:
            continue
        checked += 1
        diffs = []
        # the human branch is sampling-free: fixed reference text through the
        # same tokenizer, slicer and CodeModel. It must match on ANY hardware,
        # so a human-branch diff means the restoration is wrong, while a
        # completion diff alone can just be cross-GPU sampling drift.
        zh_new = (r["human"].get("detection") or {}).get("z_score")
        zh_old = (old["human"].get("detection") or {}).get("z_score")
        if zh_new != zh_old:
            diffs.append(f"HUMAN(detector): z {zh_old} -> {zh_new}")
        for br in ("watermarked", "unwatermarked"):
            if r[br]["completions"] != old[br]["completions"]:
                n = sum(a != b for a, b in
                        zip(r[br]["completions"], old[br]["completions"]))
                diffs.append(f"{br}: {n}/{len(old[br]['completions'])} completions differ")
            zn = (r[br].get("detection") or {}).get("z_score")
            zo = (old[br].get("detection") or {}).get("z_score")
            if zn != zo:
                diffs.append(f"{br}: z {zo} -> {zn}")
        if diffs:
            mismatched += 1
            print(f"  task {r['task_id']}: " + "; ".join(diffs))
    verdict = "PASS" if mismatched == 0 else "FAIL"
    print(f"[verify] {verdict}: {checked - mismatched}/{checked} tasks identical "
          f"to {stored_path.name}")
    if mismatched:
        raise SystemExit(1)


def compare_runs(path_a: Path, path_b: Path) -> None:
    """Clean detection statistics of two repro runs on their COMMON tasks.

    Used for two things: quantifying the hardware term between a stored A100 run
    and a re-run on other hardware, and comparing checkpoint operating points.
    Restricting to shared task ids is what makes the two columns comparable —
    a run with `--limit` covers a prefix of the other.
    """
    runs = {}
    for name, p in (("A", path_a), ("B", path_b)):
        data = json.loads(p.read_text())
        runs[name] = {r["task_id"]: r for r in data["results"] if r.get("valid")}
    common = sorted(set(runs["A"]) & set(runs["B"]), key=str)
    print(f"A = {path_a.name}\nB = {path_b.name}\ncommon valid tasks: {len(common)}")
    if not common:
        return

    rows = {k: [runs[k][t] for t in common] for k in runs}
    ident = sum(
        runs["A"][t]["watermarked"]["completions"]
        == runs["B"][t]["watermarked"]["completions"]
        for t in common
    )
    print(f"identical watermarked completion sets: {ident}/{len(common)}")

    stats = {k: utils.detection_summary(v) for k, v in rows.items()}
    keys = ("auroc", "tpr@1pct", "tpr@5pct", "mean_z_watermarked",
            "mean_z_unwatermarked")
    print(f"\n{'metric':22s} {'A':>10s} {'B':>10s} {'B-A':>10s}")
    for k in keys:
        a, b = stats["A"][k], stats["B"][k]
        d = "n/a" if a is None or b is None else f"{b - a:+.4f}"
        fa = "n/a" if a is None else f"{a:.4f}"
        fb = "n/a" if b is None else f"{b:.4f}"
        print(f"{k:22s} {fa:>10s} {fb:>10s} {d:>10s}")
    for k in ("avg_pass1_watermarked", "avg_pass1_unwatermarked"):
        src = k.replace("avg_pass1_", "")
        a = sum(r[src]["pass@1"] for r in rows["A"]) / len(common)
        b = sum(r[src]["pass@1"] for r in rows["B"]) / len(common)
        print(f"{k:22s} {a:10.4f} {b:10.4f} {b - a:+10.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-runs", nargs=2, metavar=("RUN_A", "RUN_B"),
                    help="print matched-subset clean statistics for two runs "
                         "and exit; no generation, no GPU")
    if "--compare-runs" in sys.argv:
        a, b = ap.parse_known_args()[0].compare_runs
        return compare_runs(Path(a), Path(b))
    ap.add_argument("--dataset", choices=sorted(repro.TASKS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--code-model", required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--output")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", type=int)
    ap.add_argument("--num-shards", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=repro.N_SAMPLES)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--switch-threshold", type=float, default=-10.0)
    ap.add_argument("--z-threshold", type=float, default=4.0)
    ap.add_argument("--hash-key", type=int, default=15485863)
    ap.add_argument("--verify-against", metavar="STORED_JSON",
                    help="regenerate and diff against a stored run instead of "
                         "writing output")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
