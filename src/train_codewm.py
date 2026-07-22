"""Training pipeline for the wm/codewm CodeModel (Guo & Cheng, NeurIPS 2025
— ../code-watermark). One method-level script, three stages:

    python train_codewm.py gen     # teacher generations on MBPP train prompts
    python train_codewm.py build   # ngram dataset: teacher top-K logits + entropy
    python train_codewm.py train   # distill + CE + switch training

Faithful to upstream (src/datahelper.py gen_mbpp + src/train.py +
modelhelper.compute_loss) with these documented deltas:

  - Corpus: upstream's gen_mbpp = teacher-generated solutions to MBPP
    prompts. Here the teacher is this pilot's model (Qwen3.5-4B) prompted
    exactly like the mbpp dataloader (chat template + SYSTEM_PROMPT incl.
    the no-comments rule, T=0.7/top_p=0.8/top_k=20), on the MBPP
    train+validation+prompt splits (474 tasks) — disjoint from the 500-task
    test split every pilot benchmark run uses.
  - Teacher logits are stored top-K (default 128) + full-vocab logsumexp
    instead of the full 152k-float rows upstream caches (which would be
    ~180 GB here). The distillation KL is computed exactly on the induced
    (K+1)-bucket coarsening (top-K tokens + one tail bucket) — the ranking
    the watermark needs lives entirely in the head of the distribution.
  - compute_loss port fixes upstream's modelhelper.py:197 bare-text
    SyntaxError and its F.kl_div argument swap (upstream passes
    probs/log-probs in the wrong slots); temperature 2.0 and the entropy>=1.2
    switch target with 2x positive weight are kept. count_entropy is dropped
    (upstream weights it 0.0).
  - Hyperparameters follow the upstream README gen_mbpp recipe:
    context_width 2, alpha_distill 0.1, alpha_ce 0.1, alpha_switch 8.0,
    batch 512, lr 2e-4, warmup 1000, cosine; epochs capped with early
    stopping on val loss instead of the README's fixed 1000.
  - ngrams are deduplicated corpus-wide (upstream does the same via its
    seen_ngram_keys set); teacher stats come from the first occurrence.

Artifacts land under --work-dir (default out/codewm_train/, or $CODEWM_WORK_DIR):
    gens.json                       teacher generations (token ids)
    ngrams_cw<w>_k<K>/*.npy         training arrays
    codemodel_cw<w>/best.pt|meta.json   trained checkpoint
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dataloader
import utils
from wm.codewm import CodeModel

# Default work dir is repo-local (gitignored); set CODEWM_WORK_DIR to relocate
# (e.g. to a scratch filesystem — artifacts total several GB).
WORK_DIR_DEFAULT = os.environ.get(
    "CODEWM_WORK_DIR", str(Path(__file__).resolve().parents[1] / "out" / "codewm_train")
)
ENTROPY_THRESHOLD = 1.2  # switch target (upstream compute_loss hardcodes it)


# ------------------------------------------------------------------ corpus


def load_train_tasks(limit=None) -> list[dict]:
    """MBPP train+validation+prompt splits (474 tasks; the pilot benchmarks
    on the disjoint 500-task test split, data/mbpp_test.json)."""
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp")
    tasks = []
    for split in ("train", "validation", "prompt"):
        tasks.extend(dict(t) for t in ds[split])
    test_ids = {t["task_id"] for t in dataloader.get("mbpp").load_tasks()}
    overlap = [t["task_id"] for t in tasks if t["task_id"] in test_ids]
    assert not overlap, f"train corpus overlaps mbpp test split: {overlap[:5]}"
    return tasks[:limit] if limit else tasks


def cmd_gen(args):
    ds = dataloader.get("mbpp")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    tasks = load_train_tasks(args.limit)
    jobs = [(t, s) for t in tasks for s in range(args.samples_per_task)]
    out_path = Path(args.work_dir) / "gens.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    if out_path.exists():
        records = json.loads(out_path.read_text())["records"]
        done = {(r["task_id"], r["sample_idx"]) for r in records}
        jobs = [(t, s) for t, s in jobs if (t["task_id"], s) not in done]
        print(f"Resuming: {len(records)} done, {len(jobs)} remaining")

    tokenizer, model = utils.load_hf_model(args.model, device, padding_side="left")
    meta = {
        "model": args.model,
        "corpus": "mbpp train+validation+prompt",
        "samples_per_task": args.samples_per_task,
        "system_prompt": ds.SYSTEM_PROMPT,
        "generation": {"temperature": args.temperature, "top_p": args.top_p,
                       "top_k": args.top_k, "max_new_tokens": ds.MAX_NEW_TOKENS},
        "seed": args.seed,
    }

    t0 = time.time()
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        chat_texts = [utils.task_chat_text(tokenizer, ds, t) for t, _ in batch]
        enc = tokenizer(chat_texts, return_tensors="pt", padding=True).to(device)
        prompt_len = enc.input_ids.shape[1]
        with torch.no_grad():
            seqs = model.generate(
                **enc,
                max_new_tokens=ds.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=tokenizer.pad_token_id,
            )
        for b, (task, sample_idx) in enumerate(batch):
            gen_ids = seqs[b, prompt_len:]
            eos_pos = (gen_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen_ids = gen_ids[: eos_pos[0] + 1]
            real_prompt = enc.input_ids[b][enc.attention_mask[b].bool()]
            records.append(
                {
                    "task_id": task["task_id"],
                    "sample_idx": sample_idx,
                    "prompt_ids": real_prompt.tolist(),
                    "gen_ids": gen_ids.tolist(),
                }
            )
        out_path.write_text(json.dumps({"meta": meta, "records": records}))
        done = min(start + args.batch_size, len(jobs))
        print(f"[{done}/{len(jobs)}] {time.time()-t0:.0f}s", flush=True)
    print(f"Done: {len(records)} generations -> {out_path}")


# ------------------------------------------------------------------ build


def load_codealpaca(tokenizer, limit: int) -> list[dict]:
    """CodeAlpaca sequences, upstream's third training source (datahelper
    TASK_SOURCES "alpaca"): full_data = instruction + "\n" + output, no
    generation step. Tokenized raw (no chat template), truncated to 1024."""
    from datasets import load_dataset

    ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
    ds = ds.shuffle(seed=42).select(range(limit))
    records = []
    for ex in ds:
        text = f"{ex['instruction']}\n{ex['output']}"
        ids = tokenizer(text, truncation=True, max_length=1024)["input_ids"]
        records.append({"prompt_ids": ids, "gen_ids": []})
    return records


def cmd_build(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cw, topk = args.context_width, args.topk
    records = json.loads((Path(args.work_dir) / "gens.json").read_text())["records"]
    out_dir = Path(args.work_dir) / f"ngrams_cw{cw}_k{topk}"
    if args.corpus == "mbpp+alpaca":
        out_dir = Path(args.work_dir) / f"ngrams_alpaca_cw{cw}_k{topk}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = utils.load_hf_model(args.model, device)
    if args.corpus == "mbpp+alpaca":
        alpaca = load_codealpaca(tokenizer, args.alpaca_limit)
        print(f"Loaded {len(alpaca)} codealpaca sequences")
        records = records + alpaca
    seen = set()
    prefixes, nexts, entropies, top_idx, top_val, lse2 = [], [], [], [], [], []

    t0 = time.time()
    for ri, rec in enumerate(records):
        ids = rec["prompt_ids"] + rec["gen_ids"]
        ids_t = torch.tensor(ids, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids_t, return_dict=True).logits[0].float()  # [L, V]
        # teacher stats at index t-1 describe the distribution over token t
        probs = torch.softmax(logits, dim=-1)
        ent = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(-1)
        tv, ti = torch.topk(logits, topk, dim=-1)
        ls2 = torch.logsumexp(logits / 2.0, dim=-1)
        ent, tv, ti, ls2 = ent.cpu(), tv.cpu(), ti.cpu(), ls2.cpu()

        for t in range(cw, len(ids)):
            key = tuple(ids[t - cw : t + 1])
            if key in seen:
                continue
            seen.add(key)
            prefixes.append(key[:-1])
            nexts.append(ids[t])
            entropies.append(float(ent[t - 1]))
            top_idx.append(ti[t - 1].numpy())
            top_val.append(tv[t - 1].numpy())
            lse2.append(float(ls2[t - 1]))
        if (ri + 1) % 50 == 0:
            print(f"[{ri+1}/{len(records)}] {len(seen)} unique ngrams "
                  f"{time.time()-t0:.0f}s", flush=True)

    np.save(out_dir / "prefix.npy", np.asarray(prefixes, dtype=np.int32))
    np.save(out_dir / "next.npy", np.asarray(nexts, dtype=np.int32))
    np.save(out_dir / "entropy.npy", np.asarray(entropies, dtype=np.float32))
    np.save(out_dir / "top_idx.npy", np.stack(top_idx).astype(np.int32))
    np.save(out_dir / "top_val.npy", np.stack(top_val).astype(np.float16))
    np.save(out_dir / "lse2.npy", np.asarray(lse2, dtype=np.float32))
    frac_high = float(np.mean(np.asarray(entropies) >= ENTROPY_THRESHOLD))
    (out_dir / "meta.json").write_text(json.dumps({
        "model": args.model, "context_width": cw, "topk": topk,
        "num_ngrams": len(nexts), "num_sequences": len(records),
        "frac_entropy_ge_threshold": frac_high,
        "entropy_threshold": ENTROPY_THRESHOLD,
    }, indent=2))
    print(f"Done: {len(nexts)} unique ngrams ({frac_high:.1%} high-entropy) -> {out_dir}")


# ------------------------------------------------------------------ train


def distill_kl(student_logits, t_idx, t_val, t_lse2, tau=2.0):
    """Exact KL(teacher || student) on the (K+1)-bucket coarsening (top-K
    teacher tokens + tail), both at temperature tau, scaled by tau^2."""
    p_k = torch.exp(t_val / tau - t_lse2.unsqueeze(1))  # [B, K]
    p_tail = (1 - p_k.sum(1)).clamp_min(1e-8)
    logq = F.log_softmax(student_logits / tau, dim=-1)  # [B, V]
    logq_k = logq.gather(1, t_idx)
    q_tail = (1 - torch.exp(logq_k).sum(1)).clamp_min(1e-8)
    kl = (p_k * (p_k.clamp_min(1e-12).log() - logq_k)).sum(1) + p_tail * (
        p_tail.log() - q_tail.log()
    )
    return kl.mean() * tau**2


def cmd_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    cw, topk = args.context_width, args.topk
    stem = "ngrams" if args.corpus == "mbpp" else "ngrams_alpaca"
    data_dir = Path(args.work_dir) / f"{stem}_cw{cw}_k{topk}"
    out_dir = Path(args.work_dir) / (
        f"codemodel_cw{cw}" if args.corpus == "mbpp" else f"codemodel_alpaca_cw{cw}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = torch.from_numpy(np.load(data_dir / "prefix.npy")).long()
    nxt = torch.from_numpy(np.load(data_dir / "next.npy")).long()
    entropy = torch.from_numpy(np.load(data_dir / "entropy.npy"))
    t_idx = torch.from_numpy(np.load(data_dir / "top_idx.npy")).long()
    t_val = torch.from_numpy(np.load(data_dir / "top_val.npy")).float()
    t_lse2 = torch.from_numpy(np.load(data_dir / "lse2.npy"))
    n = len(nxt)

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * 0.2)
    val_ix, train_ix = perm[:n_val], perm[n_val:]
    print(f"{n} ngrams: {len(train_ix)} train / {len(val_ix)} val")

    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(args.model)
    model_cfg = getattr(model_cfg, "text_config", model_cfg)
    vocab_size = model_cfg.vocab_size
    model = CodeModel({"vocab_size": vocab_size}).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_ix) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    from transformers import get_cosine_schedule_with_warmup

    sched = get_cosine_schedule_with_warmup(opt, args.warmup_steps, total_steps)

    def batch_loss(ix):
        b_prefix = prefix[ix].to(device)
        out = model(b_prefix)
        s_logits, s_switch = out["logits"], out["switch"]
        b_ent = entropy[ix].to(device)
        target = (b_ent >= ENTROPY_THRESHOLD).float()
        weight = 1.0 + target  # 2x weight on high-entropy positives
        l_distill = distill_kl(
            s_logits, t_idx[ix].to(device), t_val[ix].to(device), t_lse2[ix].to(device)
        )
        l_ce = F.cross_entropy(s_logits, nxt[ix].to(device), label_smoothing=0.1)
        l_switch = F.binary_cross_entropy_with_logits(s_switch, target, weight=weight)
        total = args.alpha_distill * l_distill + args.alpha_ce * l_ce + args.alpha_switch * l_switch
        acc = ((torch.sigmoid(s_switch) > 0.5) == target.bool()).float().mean()
        top1 = (s_logits.argmax(-1) == t_idx[ix, 0].to(device)).float().mean()
        return total, {"distill": l_distill.item(), "ce": l_ce.item(),
                       "switch": l_switch.item(), "switch_acc": acc.item(),
                       "top1_agree": top1.item()}

    best_val, best_epoch, log = float("inf"), -1, []
    # second selection criterion: ranking quality only (distill + ce) — the
    # alpha_switch=8 weighting makes total val loss track the fast-converging
    # switch head, which can early-select a checkpoint whose ranking (the
    # pairing signal the watermark rides on) is still improving
    best_rank, best_rank_epoch = float("inf"), -1
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep = torch.randperm(len(train_ix), generator=g)
        tr_metrics, tr_loss = [], 0.0
        for s in range(steps_per_epoch):
            ix = train_ix[ep[s * args.batch_size : (s + 1) * args.batch_size]]
            total, m = batch_loss(ix)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tr_loss += total.item()
            tr_metrics.append(m)

        model.eval()
        va_loss, va_metrics = 0.0, []
        with torch.no_grad():
            for s in range(0, len(val_ix), args.batch_size):
                ix = val_ix[s : s + args.batch_size]
                total, m = batch_loss(ix)
                va_loss += total.item() * len(ix)
                va_metrics.append(m)
        va_loss /= len(val_ix)

        def avg(ms, k):
            return sum(m[k] for m in ms) / len(ms)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss / steps_per_epoch,
            "val_loss": va_loss,
            "val_switch_acc": avg(va_metrics, "switch_acc"),
            "val_top1_agree": avg(va_metrics, "top1_agree"),
            "val_distill": avg(va_metrics, "distill"),
            "val_ce": avg(va_metrics, "ce"),
            "val_switch": avg(va_metrics, "switch"),
            "lr": sched.get_last_lr()[0],
            "seconds": round(time.time() - t0),
        }
        log.append(row)
        print(json.dumps(row), flush=True)

        if va_loss < best_val:
            best_val, best_epoch = va_loss, epoch
            torch.save(model.state_dict(), out_dir / "best.pt")
        rank_loss = row["val_distill"] + row["val_ce"]
        if rank_loss < best_rank:
            best_rank, best_rank_epoch = rank_loss, epoch
            torch.save(model.state_dict(), out_dir / "best_rank.pt")
        (out_dir / "meta.json").write_text(json.dumps({
            "model": args.model, "vocab_size": vocab_size,
            "context_width": cw, "topk": topk,
            "entropy_threshold": ENTROPY_THRESHOLD,
            "alphas": {"distill": args.alpha_distill, "ce": args.alpha_ce,
                       "switch": args.alpha_switch},
            "batch_size": args.batch_size, "lr": args.lr,
            "warmup_steps": args.warmup_steps, "seed": args.seed,
            "data_dir": str(data_dir), "num_ngrams": n,
            "best_epoch": best_epoch, "best_val_loss": best_val,
            "best_rank_epoch": best_rank_epoch, "best_rank_loss": best_rank,
            "log": log,
        }, indent=2))
        if epoch - best_epoch >= args.patience:
            print(f"Early stop at epoch {epoch} (best {best_epoch})")
            break
    print(f"Done: best val {best_val:.4f} @ epoch {best_epoch} -> {out_dir}/best.pt")


# ------------------------------------------------------------------ cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "build", "train"])
    ap.add_argument("--work-dir", default=WORK_DIR_DEFAULT)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    # gen
    ap.add_argument("--samples-per-task", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="gen: prompt batch (16); train: ngram batch (512)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    # build + train (upstream README gen_mbpp recipe)
    ap.add_argument("--context-width", type=int, default=2)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--corpus", choices=["mbpp", "mbpp+alpaca"], default="mbpp",
                    help="build: add CodeAlpaca sequences (upstream's alpaca "
                    "source); train: reads the matching ngram dir")
    ap.add_argument("--alpaca-limit", type=int, default=20000)
    # train
    # cosine horizon sized to the actual gen_mbpp ngram count (~42k -> 66
    # steps/epoch); the upstream README's warmup 1000 would span 15 epochs
    # and the first training run early-stopped inside it
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--alpha-distill", type=float, default=0.1)
    ap.add_argument("--alpha-ce", type=float, default=0.1)
    ap.add_argument("--alpha-switch", type=float, default=8.0)
    args = ap.parse_args()
    if args.batch_size is None:
        args.batch_size = 16 if args.stage == "gen" else 512

    {"gen": cmd_gen, "build": cmd_build, "train": cmd_train}[args.stage](args)


if __name__ == "__main__":
    main()
