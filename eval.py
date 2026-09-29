"""Evaluate a trained checkpoint on a held-out split.

    python eval.py --ckpt runs/<run>/best.pt \
                   --data data/test/random_N6_18000.jsonl

Reports two accuracies:
    solve rate    predicted actions turn initial into target (the real metric)
    exact match   predicted actions equal the golden sequence token for token

plus one table per --group-by axis.

--group-by chooses how the items are bucketed. It takes either a name from
GROUPINGS below, for keys that have to be computed, or any field the items
themselves carry -- so a later split only has to write its own metric into
the JSONL to become groupable, with no change here:

    python eval.py ... --group-by steps stop_reason
    python eval.py ... --group-by difficulty      # a field of that split

The decoding and the tables are separate: decode_split() returns one row per
item, and every table is a fold over those rows. The result file holds the
same rows, so a new grouping can also be applied to a finished run without
decoding it again.

--beams picks the decoder. The default 1 is plain greedy. Above 1 it beam
searches and keeps the first beam `check()` verifies, which costs no training
and needs no golden answer -- `check()` reads the item's own initial and
target boards, and those are inputs. The report then carries solve@1, the
top-scoring beam before verification, beside solve@B, so the gain from
verifying is separated from the gain from searching. Note solve@1 is NOT the
greedy number: beam 0 maximises total sequence log-probability while greedy
takes the per-step argmax, and the two paths diverge. To quote greedy, run
with --beams 1.

    python eval.py --ckpt runs/<run>/best.pt \
                   --data data/test/random_N6_18000.jsonl --beams 8

Every run writes a result file: the whole split, unchanged, with the
prediction, whether it solved the board, and why decoding stopped added to
each item. `stop_reason` is "stop" when the model emitted <EOS> or <PAD> and
"length" when it was still going at the decode cap. It lands beside the checkpoint unless --save says
otherwise.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from utils.data import check
from utils.model import (MAX_DECODE_TOKENS, MODEL_LEVELS, PAD, TOKENIZERS,
                         TranSGridModel, peek_n)

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True,
                   help="checkpoint to score, e.g. runs/<run>/best.pt")
    p.add_argument("--data", required=True,
                   help="split to score it on; point it at data/test/")

    p.add_argument("--beams", type=int, default=1,
                   help="1 (default) is greedy; above 1 beam searches and "
                        "keeps the first beam check() verifies")
    p.add_argument("--group-by", nargs="+",
                   default=["stop_reason", "steps", "action"],
                   help="one table per axis: a name from GROUPINGS, or any "
                        "field the items carry")
    p.add_argument("--save", default=None,
                   help="where to write the result JSONL (default: "
                        "eval_<data>.jsonl beside the checkpoint)")

    p.add_argument("--batch-size", type=int, default=256,
                   help="items decoded per forward pass")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help=f"decode cap (default: {MAX_DECODE_TOKENS}, the same "
                        f"constant train.py evaluates with)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu",
                   help="cuda or cpu; pick a GPU with CUDA_VISIBLE_DEVICES")
    p.add_argument("--limit", type=int, default=None,
                       help="score only the first N items (default: all)")
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"],
                   help="autocast dtype; off decodes in fp32")
    return p.parse_args()


def load_items(path, limit=None):
    items = []
    with open(path) as f:
        for line in f:
            if limit is not None and len(items) >= limit:
                break
            item = json.loads(line)
            item["answer"] = [tuple(a) for a in item["answer"]]
            items.append(item)
    return items


def trim_pad(ids):
    """Drop the <PAD> tail generate() writes after a row has stopped."""
    end = len(ids)
    while end and ids[end - 1] == PAD:
        end -= 1
    return ids[:end]


def load_checkpoint(path, device, n):
    """The model a checkpoint was trained as, ready to decode with."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt["n"] != n:
        raise ValueError(f"checkpoint was trained on n={ckpt['n']}, "
                         f"data is n={n}")

    # The encoder is fed at its real source length, the width the checkpoint
    # was trained at; the n check above is what makes that width agree.
    tokenizer = TOKENIZERS[ckpt["args"]["src_enc"]](n)

    # checkpoints saved before train.py recorded `cfg` carry only their level
    # name, which is exactly that level's sizes
    cfg = ckpt.get("cfg") or MODEL_LEVELS[ckpt["level"]]
    model = TranSGridModel.from_config(cfg, tokenizer,
                                       src_pos=ckpt["args"]["src_pos"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), tokenizer, ckpt, cfg


def decode_split(model, tokenizer, items, beams=1, cap=MAX_DECODE_TOKENS,
                 batch_size=256, device="cpu", amp_dtype=None):
    """Decode every item; returns one result row each.

    A row is the item verbatim plus what the model did with it, which is
    also exactly what lands in the result JSONL. Keeping the two the same
    means every table below can be recomputed from the file alone, and a
    later split only has to carry its own metric to be grouped by it.
    """
    rows = []
    for start in tqdm(range(0, len(items), batch_size), desc="decoding",
                      dynamic_ncols=True):
        batch = items[start:start + batch_size]
        src = torch.tensor(
            [tokenizer.encode_source(it["initial"], it["target"])
             for it in batch], dtype=torch.long, device=device)
        with torch.autocast(device_type=device, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            if beams > 1:
                out, stopped = model.beam_generate(src, beams,
                                                   max_new_tokens=cap)
            else:
                out, stopped = model.generate(src, max_new_tokens=cap,
                                              return_stop=True)
                out, stopped = out[:, None], stopped[:, None]

        for item, beams_ids, stops in zip(batch, out.tolist(),
                                          stopped.tolist()):
            
            
            cands = [tokenizer.decode_actions(ids) for ids in beams_ids]
            hits = [check(item, c) for c in cands]
            pick = hits.index(True) if any(hits) else 0
            pred, ids, stop = cands[pick], beams_ids[pick], stops[pick]
            # beam_rank is 1 for a greedy-equivalent hit and 0 for a miss, so
            # the beam budget can be re-tuned from the file alone
            rows.append({**item,
                         "answer": [list(a) for a in item["answer"]],
                         "pred": [list(a) for a in pred],
                         "pred_ids": trim_pad(ids),
                         "solved": bool(hits[pick]),
                         "exact": pred == item["answer"],
                         "solved_top1": bool(hits[0]),
                         "beam_rank": pick + 1 if hits[pick] else 0,
                         "stop_reason": "stop" if stop else "length"})
    return rows


def _gold_first_action(row):
    """The gold action, on single-step items only; None drops the rest."""
    return tuple(row["answer"][0]) if row["steps"] == 1 else None


GROUPINGS = {
    "steps": (lambda row: row["steps"], "answer length"),
    "action": (_gold_first_action, "action (single-step items)"),
    "stop_reason": (lambda row: row["stop_reason"], "stop reason"),
}


def resolve_grouping(name, rows):
    """--group-by <name> -> (key function, table heading)."""
    if name in GROUPINGS:
        key_of, heading = GROUPINGS[name]
    elif rows and name in rows[0]:
        key_of, heading = (lambda row: row.get(name)), name
    else:
        raise SystemExit(f"--group-by {name!r} is neither a computed grouping "
                         f"({', '.join(GROUPINGS)}) nor a field of the split")
    try:
        if rows:
            key_of(rows[0])
    except KeyError as missing:
        raise SystemExit(f"--group-by {name!r} needs field {missing}, which "
                         f"this split does not carry") from None
    return key_of, heading


def label(key):
    """A grouping key as a printable label; actions carry their arguments."""
    if isinstance(key, tuple):
        return f"{key[0]}({','.join(map(str, key[1:]))})"
    return str(key)


def breakdown(rows, key_of):
    """{key: [items, solved, solved_top1]}; a key of None drops the row."""
    groups = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        key = key_of(row)
        if key is None:
            continue
        group = groups[key]
        group[0] += 1
        group[1] += row["solved"]
        group[2] += row["solved_top1"]
    return groups


def print_breakdown(heading, groups, total, beams):
    """One table: share of the split, solve rate, and the beam gain.

    Rows come out in key order -- numerically for answer lengths, so the
    table still reads 9, 10, 11 rather than 1, 10, 11 once a split goes
    past nine actions.
    """
    if not groups:
        return
    width = max(len(label(k)) for k in groups)
    print(f"\nby {heading}:")
    for key in sorted(groups):
        n, solved, top1 = groups[key]
        line = (f"  {label(key):<{width}} {solved:>6}/{n:<6} = "
                f"{solved / n:.4f}   {n / total:6.1%} of items")
        if beams > 1:
            line += f"   solve@1 {top1 / n:.4f}  {(solved - top1) / n:+.4f}"
        print(line)


def print_spread(rows):
    """How many distinct first actions the model produces.

    A model that has not learned to read the boards falls back on the label
    marginal and emits a handful of actions no matter what it is shown, so
    this says more than the loss does.
    """
    pred = Counter(tuple(r["pred"][0]) if r["pred"] else None for r in rows)
    gold = Counter(tuple(r["answer"][0]) for r in rows)
    top, count = pred.most_common(1)[0]
    print(f"\nprediction spread  {len(pred)} distinct first actions vs "
          f"{len(gold)} in the data; most common is {label(top)} at "
          f"{count / len(rows):.1%} of items")


def report(rows, group_by, beams):
    """The headline accuracies, then one table per --group-by axis."""
    total = len(rows)
    solved = sum(r["solved"] for r in rows)
    top1 = sum(r["solved_top1"] for r in rows)
    exact = sum(r["exact"] for r in rows)

    print(f"solve rate   {solved}/{total} = {solved / total:.4f}")
    if beams > 1:
        print(f"  solve@1    {top1}/{total} = {top1 / total:.4f}"
              f"   top beam before verifying")
        print(f"  gain       {(solved - top1) / total:+.4f}"
              f"   from verifying {beams} beams")
    print(f"exact match  {exact}/{total} = {exact / total:.4f}")

    print_spread(rows)
    for name in group_by:
        key_of, heading = resolve_grouping(name, rows)
        print_breakdown(heading, breakdown(rows, key_of), total, beams)


def main():
    args = parse_args()
    device = args.device
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "off": None}[args.amp]
    if device == "cpu":
        amp_dtype = None

    data_path = ROOT / args.data
    items = load_items(data_path, args.limit)
    if not items:
        raise SystemExit(f"{data_path} holds no items")
    n = peek_n(data_path)

    model, tokenizer, ckpt, cfg = load_checkpoint(ROOT / args.ckpt, device, n)
    enc, src_pos = ckpt["args"]["src_enc"], ckpt["args"]["src_pos"]
    print(f"{args.ckpt}: {ckpt['level']} (d={cfg['d_model']}, "
          f"{cfg['heads']} heads, ff={cfg['d_ff']}), "
          f"{model.n_params() / 1e6:.2f}M params, "
          f"{src_pos} positions, epoch {ckpt['epoch']}")
    print(f"{data_path.name}: {len(items)} items, board {n}x{n}\n")

    cap = args.max_new_tokens or MAX_DECODE_TOKENS
    print(f"encoder: {enc} encoding, {tokenizer.src_len} ids")
    how = f"beam {args.beams} + verify" if args.beams > 1 else "greedy"
    print(f"decoder: {how}, cap {cap} tokens\n")

    rows = decode_split(model, tokenizer, items, args.beams, cap,
                        args.batch_size, device, amp_dtype)

    # written before the tables: decoding is the expensive half, and a
    # grouping that raises must not take the results down with it
    out_path = (ROOT / args.save if args.save else
                (ROOT / args.ckpt).parent / f"eval_{data_path.stem}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    report(rows, args.group_by, args.beams)
    print(f"\n{len(rows)} results -> {out_path}")


if __name__ == "__main__":
    main()
