from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval import load_checkpoint, load_items, trim_pad
from utils.data import check
from utils.model import (ARG_BASE, BOS, EOS, PAD, MAX_DECODE_TOKENS)


def prefix_ids(tokenizer, answer):
    """``<BOS>`` plus every action but the last, with no ``<EOS>``."""
    ids = [BOS]
    for name, *args in answer[:-1]:
        ids.append(tokenizer.action_to_id[name])
        ids += [ARG_BASE + int(a) for a in args]
    return ids


@torch.no_grad()
def continue_greedy(model, src, ys, cap):
    """Greedy decoding from an already-seeded decoder input."""
    memory, src_pad = model.encode(src)
    done = torch.zeros(ys.size(0), dtype=torch.bool, device=ys.device)
    for _ in range(min(cap, model.max_tgt_positions - ys.size(1))):
        nxt = model.decode(ys, memory, src_pad)[:, -1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        done |= (nxt == EOS) | (nxt == PAD)
        if bool(done.all()):
            break
    return ys, done


@torch.no_grad()
def continue_beam(model, src, ys, beams, cap):
    """Beam search from an already-seeded decoder input."""
    device = ys.device
    memory, src_pad = model.encode(src)
    n, s_len, d = memory.shape
    mem = memory[:, None].expand(n, beams, s_len, d).reshape(n * beams, s_len, d)
    pad = src_pad[:, None].expand(n, beams, s_len).reshape(n * beams, s_len)

    plen = ys.size(1)
    ys = ys[:, None].expand(n, beams, plen).reshape(n * beams, plen).contiguous()
    scores = torch.full((n, beams), float("-inf"), device=device)
    scores[:, 0] = 0.0
    done = torch.zeros(n * beams, dtype=torch.bool, device=device)

    for _ in range(min(cap, model.max_tgt_positions - plen)):
        logp = F.log_softmax(model.decode(ys, mem, pad)[:, -1].float(), -1)
        logp[done] = float("-inf")
        logp[done, PAD] = 0.0
        vocab = logp.size(-1)
        cand = (scores.reshape(-1, 1) + logp).view(n, beams * vocab)
        scores, flat = cand.topk(beams, dim=-1)
        rows = (torch.arange(n, device=device)[:, None] * beams
                + flat // vocab).reshape(-1)
        token = (flat % vocab).reshape(-1)
        ys = torch.cat([ys[rows], token[:, None]], dim=1)
        done = done[rows] | (token == EOS) | (token == PAD)
        if bool(done.all()):
            break
    return ys.view(n, beams, -1), done.view(n, beams)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="runs/level7_grid_pair/best.pt",
                   help="checkpoint to score, e.g. runs/<run>/best.pt")
    p.add_argument("--data", default="data/reasoning/balanced_4800.jsonl",
                   help="split to score, relative to the repository root")
    p.add_argument("--save", default="results/ablation/action_explicit_goal",
                   help="where to write the result JSONL")
    p.add_argument("--beams", type=int, default=1,
                   help="1 is greedy; above that keeps the first beam check() "
                        "verifies, as eval.py does (default: 1)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    return p.parse_args()


def main() -> None:
    a = parse_args()
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16,
           "off": None}[a.amp if a.device != "cpu" else "off"]
    data = Path(a.data)
    data = data if data.is_absolute() else ROOT / data
    ckpt_path = Path(a.ckpt)
    ckpt_path = ckpt_path if ckpt_path.is_absolute() else ROOT / ckpt_path

    items = load_items(data)
    model, tok, _, _ = load_checkpoint(ckpt_path, a.device, items[0]["n"])

    buckets = collections.defaultdict(list)
    for it in items:
        buckets[len(prefix_ids(tok, it["answer"]))].append(it)

    rows = []
    label = f"{ckpt_path.parent.name} prefix"
    for plen in tqdm(sorted(buckets), desc=label, dynamic_ncols=True):
        group = buckets[plen]
        for s in range(0, len(group), a.batch_size):
            batch = group[s:s + a.batch_size]
            src = torch.tensor([tok.encode_source(it["initial"], it["target"])
                                for it in batch], dtype=torch.long,
                               device=a.device)
            ys = torch.tensor([prefix_ids(tok, it["answer"]) for it in batch],
                              dtype=torch.long, device=a.device)
            cap = MAX_DECODE_TOKENS - plen
            with torch.autocast(device_type=a.device, dtype=amp,
                                enabled=amp is not None):
                if a.beams > 1:
                    out, done = continue_beam(model, src, ys, a.beams, cap)
                else:
                    out, done = continue_greedy(model, src, ys, cap)
                    out, done = out[:, None], done[:, None]

            for it, per_beam, stops in zip(batch, out.tolist(), done.tolist()):
                head = [tuple(x) for x in it["answer"][:-1]]
                tails = [tok.decode_actions(ids[plen:]) for ids in per_beam]
                hits = [check(it, head + list(t)) for t in tails]
                pick = hits.index(True) if any(hits) else 0
                tail, ids, stop = tails[pick], per_beam[pick], stops[pick]
                pred = head + list(tail)
                rows.append({**it,
                             "answer": [list(x) for x in it["answer"]],
                             "pred": [list(x) for x in pred],
                             "pred_ids": trim_pad(ids),
                             "pred_tail": [list(x) for x in tail],
                             "given": len(it["answer"]) - 1,
                             "solved": bool(hits[pick]),
                             "exact": [list(x) for x in pred] == [
                                 list(x) for x in it["answer"]],
                             "last_exact": list(tail) == [
                                 tuple(it["answer"][-1])],
                             "solved_top1": bool(hits[0]),
                             "beam_rank": pick + 1 if hits[pick] else 0,
                             "stop_reason": "stop" if stop else "length"})

    rows.sort(key=lambda r: r["id"])
    out_path = Path(a.save)
    out_path = out_path if out_path.is_absolute() else ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    solved = sum(r["solved"] for r in rows)
    last = sum(r["last_exact"] for r in rows)
    print(f"solve rate   {solved}/{n} = {solved / n:.4f}")
    print(f"last action  {last}/{n} = {last / n:.4f}   the withheld action, "
          f"token for token")
    print(f"{n} results -> {out_path}")


if __name__ == "__main__":
    main()
