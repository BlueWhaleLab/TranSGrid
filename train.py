"""Train the encoder-decoder Transformer on a board planning split.

    python train.py --level level6 \
        --train data/train/random_N6_30000000.jsonl \
        --eval  data/dev/random_N6_2700.jsonl

--train takes several path[:count] sources and mixes them, so the recipe is
explicit and reproducible:

    --train data/train/atomic_N6_300000.jsonl:20000 \
            data/train/random_N6_30000000.jsonl:80000

--eval is decoded in full after every epoch -- solve rate overall and per
answer length -- and is what best.pt is selected on, so it takes the dev
split. Reporting goes through eval.py on data/test/.

Checkpoints, config.json and a JSONL log land in runs/<name>/. Rerunning
with the same --seed and the same data reproduces the run. Add --wandb to
mirror the log to Weights & Biases (--wandb-mode offline needs no login).

best.pt is selected on solve rate, not loss -- the note beside the
checkpoint says why the two disagree. Cross-entropy is bounded below by the
label entropy here: many action orderings solve the same board, so `loss`
only compares within one labelling convention. `solve_rate` replays the
boards and is not.
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data import check, load_dataset
from utils.model import (MAX_DECODE_TOKENS, MODEL_LEVELS, PAD, TOKENIZERS,
                         BoardDataset, TranSGridModel, collate, parse_source,
                         peek_n)

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", required=True, choices=list(MODEL_LEVELS),
                   help="ladder rung: sets layers, d_model, heads and d_ff")
    p.add_argument("--train", nargs="+", required=True,
                   help="one or more path[:count] sources to mix")
    p.add_argument("--eval", required=True,
                   help="split scored each epoch and used to pick best.pt; "
                        "point it at dev, report on test with eval.py")

    p.add_argument("--name", default=None,
                   help="run directory name (default: built from the config)")
    p.add_argument("--epochs", type=int, default=6,
                   help="passes over the training mix")
    p.add_argument("--batch-size", type=int, default=512,
                   help="items per step, train and eval alike")
    p.add_argument("--lr", type=float, default=1.5e-4,
                   help="peak learning rate, reached at the end of warmup")
    p.add_argument("--src-pos", default="grid", choices=["grid", "flat"],
                   help="encoder positions: board/row/col, or flat absolute")
    p.add_argument("--src-enc", default="pair", choices=list(TOKENIZERS),
                   help="pair: one token per cell holding both digits, n*n "
                        "tokens. board: the two boards end to end")
    p.add_argument("--seed", type=int, default=42,
                   help="seeds init, dropout and the training shuffle")
    p.add_argument("--wandb", action="store_true",
                   help="log to Weights & Biases")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                       else "cpu",
                       help="cuda or cpu; pick a GPU with CUDA_VISIBLE_DEVICES")
    p.add_argument("--dropout", type=float, default=0.1,
                       help="dropout on the embeddings and inside every layer")

    p.add_argument("--min-lr-ratio", type=float, default=0.1,
                   help="floor the decay lands on, as a fraction of --lr")
    p.add_argument("--warmup", type=int, default=1000,
                   help="steps of linear warmup from zero to --lr")
    p.add_argument("--schedule", default="cosine",
                   choices=["constant", "cosine"],
                   help="constant: warmup, hold the peak, then decay over the "
                        "last --decay-frac. cosine stretches with --epochs, "
                        "so a short run is penalised by the schedule rather "
                        "than by its length")
    p.add_argument("--decay-frac", type=float, default=0.3,
                   help="constant schedule: fraction of the post-warmup "
                        "steps spent decaying at the end")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW weight decay")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="max gradient norm per step")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="cross-entropy label smoothing")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader worker processes")
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"],
                   help="autocast dtype; off trains in fp32")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model (slow first epoch)")
    p.add_argument("--wandb-project", default="TranSGrid",
                   help="wandb project the run lands in")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"],
                   help="offline logs to run_dir without a login")
    p.add_argument("--log-every", type=int, default=50,
                   help="steps between per-step wandb points")
    return p.parse_args()


def environment():
    """Code version, hardware and versions, for reproducing a run later.

    argparse already carries every training knob, --seed included, and those
    go up verbatim. What a rerun cannot recover from the flags is which
    commit produced them and what it ran on, so that goes up beside them.
    `git_dirty` is the one to watch: a run started with uncommitted edits
    cannot be reproduced from its commit alone.
    """
    def git(*args):
        try:
            done = subprocess.run(("git", "-C", str(ROOT)) + args,
                                  capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"cmd": " ".join(sys.argv),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
            # every card here is the same model, so the name alone does not
            # say which one ran this; the launch mask does
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "host": platform.node()}


def lr_lambda(step, warmup, total, min_ratio, schedule="constant",
              decay_frac=0.3):
    """Warmup, then either a cosine over the whole run or hold-then-decay.

    Hold-then-decay keeps the peak until the last `decay_frac` of the
    post-warmup steps
    """
    if step < warmup:
        return (step + 1) / warmup
    if schedule == "cosine":
        progress = (step - warmup) / max(1, total - warmup)
    else:
        hold = warmup + (1 - decay_frac) * (total - warmup)
        if step < hold:
            return 1.0
        progress = (step - hold) / max(1, total - hold)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return min_ratio + (1 - min_ratio) * cosine


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype, tokenizer, items):
    """Teacher-forced loss and token accuracy, plus greedy solve rate over
    the whole split, broken down by the golden answer's length.

    `solve_rate` applies the predicted actions to the item's initial board
    and compares with the target, so it is invariant to how the labels are
    written. `exact_match` is a token-for-token match with the golden answer,
    and their difference (`alt_solution`) is a direct reading of how much
    label ambiguity the split carries: many action orderings solve the same
    board, and the model is free to pick another one.
    """
    model.eval()
    loss_sum = tok_seen = tok_right = 0.0
    seen = 0
    agg = defaultdict(float)
    by_k = defaultdict(lambda: defaultdict(float))

    for src, tgt in tqdm(loader, desc="eval", leave=False, dynamic_ncols=True):
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        with torch.autocast(device_type=device, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             tgt_out.reshape(-1))
            pred = model.generate(src, max_new_tokens=MAX_DECODE_TOKENS)
        mask = tgt_out != PAD
        loss_sum += loss.item() * mask.sum().item()
        tok_seen += mask.sum().item()
        tok_right += ((logits.argmax(-1) == tgt_out) & mask).sum().item()

        for row in range(src.size(0)):
            item = items[seen + row]
            actions = tokenizer.decode_actions(pred[row].tolist())
            ok = bool(check(item, actions))
            hit = actions == item["answer"]
            for bucket in (agg, by_k[item["steps"]]):
                bucket["items"] += 1
                bucket["solved"] += ok
                bucket["exact"] += hit
                bucket["alt"] += ok and not hit
        seen += src.size(0)

    model.train()
    n = agg["items"]
    return {"loss": loss_sum / max(1, tok_seen),
            "token_acc": tok_right / max(1, tok_seen),
            "solve_rate": agg["solved"] / n,
            "exact_match": agg["exact"] / n,
            "alt_solution": agg["alt"] / n,
            "decoded": int(n),
            "solve_by_k": {k: by_k[k]["solved"] / by_k[k]["items"]
                           for k in sorted(by_k)}}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "off": None}[args.amp]
    if device == "cpu":
        amp_dtype = None

    train_sources = [parse_source(s, ROOT) for s in args.train]
    eval_path = ROOT / args.eval
    
    # Every split has to agree on the board size, and only the first train
    # source is asked. 
    n = peek_n(train_sources[0][0])
    for path, _ in train_sources[1:] + [(eval_path, None)]:
        if peek_n(path) != n:
            raise ValueError(f"{path} is n={peek_n(path)}, but "
                             f"{train_sources[0][0]} is n={n}")
    tokenizer = TOKENIZERS[args.src_enc](n)

    print(f"board {n}x{n}, {args.src_enc} encoding: {tokenizer.src_len} "
          f"encoder tokens, vocab {tokenizer.src_vocab_size} encoder / "
          f"{tokenizer.tgt_vocab_size} decoder")

    train_set = BoardDataset(train_sources, tokenizer)
    eval_set = BoardDataset(eval_path, tokenizer)
    eval_items = load_dataset(eval_path)   # raw boards, for the solve check
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              collate_fn=collate,
                              num_workers=args.workers, pin_memory=True,
                              generator=torch.Generator().manual_seed(args.seed))
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate,
                             num_workers=args.workers, pin_memory=True)
    for part in train_set.composition:
        share = part["items"] / len(train_set)
        print(f"  {part['items']:>7} items ({share:5.1%})  "
              f"{Path(part['path']).name}")
    print(f"train {len(train_set)} items, eval {len(eval_set)} items")

    cfg = dict(MODEL_LEVELS[args.level])
    model = TranSGridModel.from_config(cfg, tokenizer, args.dropout,
                                       args.src_pos).to(device)
    print(f"{args.level}: {model.n_params() / 1e6:.2f}M params, "
          f"{cfg['layers']}+{cfg['layers']} layers, d={cfg['d_model']}, "
          f"{cfg['heads']} heads (dim {cfg['d_model'] // cfg['heads']}), "
          f"ff={cfg['d_ff']}, {args.src_pos} encoder positions")
    if args.compile:
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD,
                                    label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), eps=1e-9,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, args.warmup, total_steps,
                                       args.min_lr_ratio, args.schedule,
                                       args.decay_frac))
    scaler = torch.amp.GradScaler(device, enabled=amp_dtype is torch.float16)

    # The run directory is all that separates two runs on disk, so every knob
    # that changes what is being compared goes in the name.
    tag = "+".join(Path(p).stem.split("_")[0] for p, _ in train_sources)
    name = args.name or "_".join(
        [args.level, f"N{n}", tag, str(len(train_set)),
         args.src_enc, args.src_pos,
         f"lr{args.lr:g}", f"ep{args.epochs}", f"bs{args.batch_size}"])
    run_dir = ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    settings = {**vars(args), "n": n, "params": model.n_params(),
                "src_vocab": tokenizer.src_vocab_size,
                "tgt_vocab": tokenizer.tgt_vocab_size,
                "src_len": tokenizer.src_len,
                "composition": train_set.composition,
                "train_items": len(train_set),
                "steps_per_epoch": len(train_loader),
                "total_steps": total_steps,
                **cfg, **env}
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2))
    log_file = open(run_dir / "log.jsonl", "a")
    print(f"logging to {run_dir.relative_to(ROOT)}")
    print(f"commit {(env['git_commit'] or '?')[:8]}"
          f"{' +uncommitted edits' if env['git_dirty'] else ''}, "
          f"seed {args.seed}, {env['gpu'] or env['host']}")

    run = None
    if args.wandb:
        import wandb

        
        run = wandb.init(project=args.wandb_project, name=name,
                         mode=args.wandb_mode, dir=str(run_dir),
                         config={**settings,
                                 **{f"mix/{Path(c['path']).stem}": c["items"]
                                    for c in train_set.composition}})
        wandb.watch(model, log="gradients", log_freq=500)  # type: ignore[arg-type]
        print(f"wandb: {getattr(run, 'url', None) or args.wandb_mode}")

    best = -1.0            # best solve rate seen; see the checkpoint note below
    step = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        running = seen = 0.0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}",
                   leave=False, dynamic_ncols=True)
        for src, tgt in bar:
            src, tgt = src.to(device, non_blocking=True), tgt.to(device,
                                                                 non_blocking=True)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            with torch.autocast(device_type=device, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 tgt_out.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tokens = (tgt_out != PAD).sum().item()
            running += loss.item() * tokens
            seen += tokens
            step += 1
            if step % 20 == 0:
                bar.set_postfix(loss=f"{running / seen:.4f}",
                                lr=f"{scheduler.get_last_lr()[0]:.2e}")
            if run is not None and step % args.log_every == 0:
                run.log({"train/loss": loss.item(),
                         "train/lr": scheduler.get_last_lr()[0],
                         "epoch": epoch}, step=step)
        bar.close()

        metrics = evaluate(model, eval_loader, criterion, device, amp_dtype,
                           tokenizer, eval_items)
        row = {"epoch": epoch, "step": step,
               "train_loss": running / max(1, seen),
               "lr": scheduler.get_last_lr()[0],
               "secs": round(time.time() - started, 1), **metrics}
        by_k = " ".join(f"{k}:{v:.2f}" for k, v in metrics["solve_by_k"].items())
        print(f"epoch {epoch:>3}/{args.epochs}  "
              f"train {row['train_loss']:.4f}  eval {row['loss']:.4f}  "
              f"tok {row['token_acc']:.4f}  solve {row['solve_rate']:.4f}  "
              f"exact {row['exact_match']:.4f}  {row['secs']}s\n"
              f"    solve by k  {by_k}")
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()
        if run is not None:
            run.log({"epoch": epoch,
                     "train/epoch_loss": row["train_loss"],
                     "eval/loss": row["loss"],
                     "eval/token_acc": row["token_acc"],
                     "eval/solve_rate": row["solve_rate"],
                     "eval/exact_match": row["exact_match"],
                     "eval/alt_solution": row["alt_solution"],
                     **{f"eval/solve_k{k}": v
                        for k, v in metrics["solve_by_k"].items()},
                     "time/epoch_secs": row["secs"]}, step=step)

        ckpt = {"model": getattr(model, "_orig_mod", model).state_dict(),
                "level": args.level, "cfg": cfg, "n": n, "epoch": epoch,
                "args": vars(args), "metrics": row}
        torch.save(ckpt, run_dir / "last.pt")

        score = metrics["solve_rate"]
        if score > best:
            best = score
            torch.save(ckpt, run_dir / "best.pt")

    log_file.close()
    if run is not None:
        run.summary["best_solve_rate"] = best
        run.finish()
    print(f"done. best solve rate {best:.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
    
    
# CUDA_VISIBLE_DEVICES=0 python3 train.py --level level7 --train data/train/random_N6_30000000.jsonl --eval data/dev/random_N6_2700.jsonl --name Final_level7 --wandb
# python3 eval.py --ckpt runs/Final_level7/best.pt --data data/test/random_N6_18000.jsonl


# CUDA_VISIBLE_DEVICES=2 python3 train.py --level level5 --train data/train/random_N6_30000000.jsonl --eval data/dev/random_N6_2700.jsonl --name Final_level5 --wandb
