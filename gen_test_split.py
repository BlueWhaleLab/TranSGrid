from __future__ import annotations

import argparse
from array import array
from hashlib import blake2b
import json
from pathlib import Path
import random

import numpy as np

from utils.dataset_io import AtomicJsonlWriter, DatasetItemValidator
from utils.induction import InductionRuleCatalog
from utils.reasoning import ReasoningScorer
from utils.sample_factory import BoardSampleFactory

ROOT = Path(__file__).resolve().parent

DEFAULT_EXCLUDE = ("data/train.jsonl",
                   "data/dev.jsonl")


SEQUENCE_TRIES = 100
BOARD_TRIES = 100
DISTINCT_TRIES = 1000


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True,
                   help="destination JSONL, read relative to the repo")
    p.add_argument("--per-length", type=int, default=400, help="items at each answer length (default: 400)")
    p.add_argument("--lengths", default="1-12", help="answer lengths to cover, e.g. 1-12 or 1-9 (default: 1-12)")
    p.add_argument("--exclude", nargs="*", default=None,
                   help=f"splits the result must be disjoint from (default: {' '.join(DEFAULT_EXCLUDE)}); pass with no values to skip the check entirely")
    p.add_argument("--n", type=int, default=6, help="board size (default: 6)")
    p.add_argument("--seed", type=int, default=1234, help="generator seed; keep it away from the seed the training pool was drawn with (default: 1234)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace the destination if it exists")
    return p.parse_args()


def parse_lengths(spec):
    """"1-12" or "1,3,5" or "1-4,9-12" -> a sorted tuple of lengths."""
    lengths = set()
    for part in spec.split(","):
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            lengths.update(range(low, high + 1))
        else:
            lengths.add(int(part))
    if not lengths or min(lengths) < 1:
        raise SystemExit(f"--lengths {spec!r} does not name a positive range")
    return tuple(sorted(lengths))


def digest(key: bytes) -> int:
    """A 64-bit digest of a question key, small enough to hold 30M of."""
    return int.from_bytes(blake2b(key, digest_size=8).digest(), "big")


def load_excluded(paths):
    """Sorted digests of every question in the splits we must avoid."""
    digests = array("Q")
    for path in paths:
        full = ROOT / path
        if not full.exists():
            raise SystemExit(f"--exclude names {path}, which does not exist")
        print(f"reading {path} ...", flush=True)
        with open(full) as f:
            for i, line in enumerate(f, 1):
                item = json.loads(line)
                digests.append(digest(
                    BoardSampleFactory.question_key(item)))
                if i % 2_000_000 == 0:
                    print(f"  {i:,} rows", flush=True)
        print(f"  {i:,} rows total", flush=True)
    table = np.frombuffer(digests, dtype=np.uint64).copy()
    table.sort()
    print(f"excluding {len(table):,} questions\n", flush=True)
    return table


def excluded(table, value: int) -> bool:
    if table is None or not len(table):
        return False
    where = np.searchsorted(table, np.uint64(value))
    return where < len(table) and table[where] == np.uint64(value)


def draw_board(factory, scorer, actions, hits):
    for _ in range(BOARD_TRIES):
        initial = factory._random_grid()
        target, _ = scorer.score_trajectory(initial, actions)
        if target != initial:
            return initial, target
        hits["identity"] += 1
    raise RuntimeError(f"{BOARD_TRIES} random boards were all left unchanged by {actions}")


def draw_actions(factory, length, rng):
    for _ in range(SEQUENCE_TRIES):
        actions = [factory._random_action() for _ in range(length)]
        if factory.moved_position_count(actions):
            return actions
    raise RuntimeError(f"could not draw a non-identity {length}-action sequence in {SEQUENCE_TRIES} tries")


def main():
    args = parse_args()
    lengths = parse_lengths(args.lengths)
    paths = DEFAULT_EXCLUDE if args.exclude is None else tuple(args.exclude)
    table = load_excluded(paths) if paths else None
    if table is None:
        print("warning: no exclusion check, the result may overlap training\n")

    rng = random.Random(args.seed)
    scorer = ReasoningScorer(args.n)
    catalog = InductionRuleCatalog(args.n, rng)
    factory = BoardSampleFactory(args.n, rng, catalog, scorer)
    validator = DatasetItemValidator(args.n, scorer, set(catalog.by_name))

    seen: set[int] = set()
    hits = {"excluded": 0, "duplicate": 0, "identity": 0}
    item_id = 0
    out = ROOT / args.output

    with AtomicJsonlWriter(out, overwrite=args.overwrite) as writer:
        for length in lengths:
            for _ in range(args.per_length):
                for _ in range(DISTINCT_TRIES):
                    actions = draw_actions(factory, length, rng)
                    initial, target = draw_board(
                        factory, scorer, actions, hits)
                    
                    item = {"id": item_id, "n": args.n, "steps": length, "initial": initial, "target": target, "answer": [list(a) for a in actions]}
                    value = digest(BoardSampleFactory.question_key(item))
                    if excluded(table, value):
                        hits["excluded"] += 1
                        continue
                    if value in seen:
                        hits["duplicate"] += 1
                        continue
                    line = json.dumps(item)
                    if not validator.validate(line):
                        raise RuntimeError(
                            f"generated item {item_id} failed validation")
                    seen.add(value)
                    writer.write_line(line)
                    item_id += 1
                    break
                else:
                    raise RuntimeError(
                        f"could not place a distinct L={length} item in "
                        f"{DISTINCT_TRIES} tries")
            print(f"L={length:2d}: {args.per_length} items", flush=True)
        writer.commit()

    print(f"\nwrote {args.output}: {item_id:,} items, "
          f"{len(lengths)} lengths x {args.per_length}")
    print(f"rejected -- overlapping an excluded split: {hits['excluded']}, duplicate within this split: {hits['duplicate']}, identity board: {hits['identity']}")


if __name__ == "__main__":
    main()
