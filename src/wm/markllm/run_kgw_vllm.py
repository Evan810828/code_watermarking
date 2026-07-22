"""KGW reproduction with the MarkLLM package + vLLM inference.

Generation: vLLM (V1) with MarkLLM's official
`watermark/kgw/kgw_logits_processor_for_vllm.KGWLogitsProcessor`, enabled per
request via SamplingParams.extra_args (kgw_delta=8 as in the pilot).
Detection: MarkLLM's `KGW.detect_watermark` (standalone text scoring), which
uses the same `(hash_key * f) % vocab_size` seeding as the vLLM processor.

Note: current MarkLLM seeds the greenlist PRF with `% vocab_size`; the pilot
implementation used the older `% (2**63 - 1)` formula. Each pipeline is
internally consistent, so aggregate rates are comparable but per-token green
lists are not.

Usage (from this directory):
    python run_kgw_vllm.py --dataset humaneval --delta 8
"""

import argparse
import json
import time

from adapter import MARKLLM_ROOT, load_pilot_utils, make_transformers_config

import dataloader

pilot_utils = load_pilot_utils()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=dataloader.DATASETS, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default: the dataset's MAX_NEW_TOKENS")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=8.0)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from watermark.kgw.kgw import KGW
    from watermark.kgw.kgw_logits_processor_for_vllm import KGWLogitsProcessor

    ds = dataloader.get(args.dataset)
    tasks = ds.load_tasks(limit=args.limit)
    max_new_tokens = args.max_new_tokens or ds.MAX_NEW_TOKENS

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # transformers in this env predates the qwen3_5 config class; read the
    # raw config.json for vocab_size (must equal the logits dim the vLLM
    # processor sees, so the generation and detection PRFs match).
    with open(hf_hub_download(args.model, "config.json")) as f:
        raw_cfg = json.load(f)
    vocab_size = raw_cfg.get("vocab_size") or raw_cfg["text_config"]["vocab_size"]
    print(f"vocab_size={vocab_size}")

    prompts = [
        pilot_utils.chat_text(tokenizer, ds.build_prompt(t), ds.SYSTEM_PROMPT)
        for t in tasks
    ]

    llm = LLM(
        model=args.model,
        logits_processors=[KGWLogitsProcessor],
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        seed=args.seed,
    )

    def gen(watermark: bool):
        sp = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=max_new_tokens,
            seed=args.seed,
            extra_args={
                "kgw_enable": watermark,
                "kgw_gamma": args.gamma,
                "kgw_delta": args.delta,
                "kgw_window_size": 1,
            },
        )
        outs = llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outs], [len(o.outputs[0].token_ids) for o in outs]

    t0 = time.time()
    wm_texts, wm_ntok = gen(True)
    base_texts, base_ntok = gen(False)
    gen_seconds = time.time() - t0
    print(f"generation done in {gen_seconds:.0f}s")

    # Detection with the MarkLLM package (device must match the generation-side
    # PRF device: the vLLM processor seeds its randperm on the GPU).
    detector = KGW(
        str(MARKLLM_ROOT / "config" / "KGW.json"),
        make_transformers_config(tokenizer=tokenizer, vocab_size=vocab_size, device="cuda"),
        gamma=args.gamma,
        delta=args.delta,
    )

    results = []
    for task, wm_t, wm_n, b_t, b_n in zip(tasks, wm_texts, wm_ntok, base_texts, base_ntok):
        results.append(
            {
                "task_id": ds.task_id(task),
                "dataset": ds.NAME,
                **ds.result_extras(task),
                "watermarked": {
                    "completion": wm_t,
                    "extracted_code": pilot_utils.extract_code(wm_t),
                    "num_gen_tokens": wm_n,
                    "detection": detector.detect_watermark(wm_t),
                },
                "unwatermarked": {
                    "completion": b_t,
                    "extracted_code": pilot_utils.extract_code(b_t),
                    "num_gen_tokens": b_n,
                    "detection": detector.detect_watermark(b_t),
                },
            }
        )

    n_wm = sum(r["watermarked"]["detection"]["is_watermarked"] for r in results)
    n_fp = sum(r["unwatermarked"]["detection"]["is_watermarked"] for r in results)
    meta = {
        "model": args.model,
        "dataset": ds.NAME,
        "num_tasks": len(results),
        "engine": "vllm",
        "watermark": {
            "algorithm": "KGW (MarkLLM package, vLLM logits processor)",
            "gamma": args.gamma,
            "delta": args.delta,
            "hash_key": 15485863,
            "prefix_length": 1,
            "z_threshold": 4.0,
        },
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": max_new_tokens,
            "seed": args.seed,
        },
        "gen_seconds": round(gen_seconds, 1),
        "summary": {
            "detect_rate_watermarked": n_wm / len(results),
            "false_positive_rate_unwatermarked": n_fp / len(results),
        },
    }
    tag = pilot_utils.model_tag(args.model)
    out_path = pilot_utils.out_dir("kgw", args.dataset, "markllm") / f"kgw_vllm_{args.dataset}_{tag}.json"
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"Done. TPR={n_wm}/{len(results)}, FP={n_fp}/{len(results)} -> {out_path}")


if __name__ == "__main__":
    main()
