from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from utils.board import ACTIONS, Board
from utils.dataset_io import AtomicJsonlWriter

N = 6
BOARD = Board([[0] * N for _ in range(N)])


def all_actions() -> list[tuple]:
    """Every legal action on the board, as utils/data.py enumerates them."""
    out = []
    for name in ACTIONS:
        if name.startswith("BLOCK"):
            out += [(name, i, j) for i in range(N - 1) for j in range(N - 1)]
        else:
            out += [(name, i) for i in range(N)]
    return out


def footprint(action: tuple) -> frozenset:
    """The cells an action moves, read off a board of distinct labels."""
    probe = [[r * N + c for c in range(N)] for r in range(N)]
    grid = [row[:] for row in probe]
    BOARD.apply(grid, action)
    return frozenset((r, c) for r in range(N) for c in range(N)
                     if grid[r][c] != probe[r][c])


ALL = all_actions()
FOOTPRINT = {action: footprint(action) for action in ALL}


def build(length: int, rng: random.Random) -> tuple[list[tuple], int]:
    actions: list[tuple] = []
    regions: list[frozenset] = []
    while len(actions) < length:
        action = rng.choice(ALL)
        cells = FOOTPRINT[action]
        if any(cells != region and cells & region for region in regions):
            continue
        if cells not in regions:
            regions.append(cells)
        actions.append(action)
    rng.shuffle(actions)
    return actions, len(regions)


def decouple(item: dict, rng: random.Random, tries: int) -> dict | None:
    """Rewrite one row on its own initial board, keeping its answer length."""
    grid = item["initial"]
    for _ in range(tries):
        actions, regions = build(item["steps"], rng)
        target = Board(grid).execute(actions)
        if target == grid:
            continue
        return dict(id=item["id"], n=item["n"], steps=item["steps"],
                    initial=[row[:] for row in grid], target=target,
                    answer=[list(a) for a in actions],
                    variant="decoupled", regions=regions)
    return None


def replays(item: dict) -> bool:
    """Whether the answer reaches the target."""
    grid = [row[:] for row in item["initial"]]
    for action in item["answer"]:
        BOARD.apply(grid, tuple(action))
    return grid == item["target"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/reasoning/balanced_4800.jsonl",
                   help="split to rewrite, relative to the repository root")
    p.add_argument("--output",
                   default="data/ablation/decoupling/balanced_4800_decoupled_1-9.jsonl",
                   help="destination JSONL, relative to the repository root")
    p.add_argument("--lengths", default="1-9", metavar="LO-HI",
                   help="inclusive answer-length range to decouple; rows "
                        "outside it are copied through (default: 1-9)")
    p.add_argument("--tries", type=int, default=400,
                   help="draws per row before giving up on its board "
                        "(default: 400)")
    p.add_argument("--seed", type=int, default=42,
                   help="sampling seed (default: 42)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace --output if it already exists")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    args = parse_args()
    try:
        lo, _, hi = args.lengths.partition("-")
        lengths = set(range(int(lo), int(hi or lo) + 1))
    except ValueError:
        raise SystemExit(f"--lengths wants LO-HI, got {args.lengths!r}")

    rng = random.Random(args.seed)
    data, output = resolve(args.data), resolve(args.output)
    try:
        source = [json.loads(line) for line in open(data)]
    except FileNotFoundError:
        raise SystemExit(f"{data} does not exist") from None

    rows, failed = [], Counter()
    for item in source:
        if item["steps"] not in lengths:
            rows.append(dict(item, variant="source"))
            continue
        rewritten = decouple(item, rng, args.tries)
        if rewritten is None:
            failed[item["steps"]] += 1
            rows.append(dict(item, variant="source"))
        else:
            rows.append(rewritten)

    broken = [r for r in rows if r["variant"] == "decoupled" and
              not replays(r)]
    if broken:
        raise SystemExit(f"{len(broken)} decoupled rows failed replay; "
                         f"refusing to write {output}")

    counts = Counter(r["steps"] for r in rows)
    print(f"{len(rows):,} rows, "
          f"{sum(r['variant'] == 'decoupled' for r in rows):,} decoupled")
    print("  per length:", " ".join(f"{L}:{counts[L]}" for L in sorted(counts)))
    if failed:
        print("  could not decouple (copied through):",
              " ".join(f"{L}:{n}" for L, n in sorted(failed.items())))

    with AtomicJsonlWriter(output, args.overwrite) as writer:
        for row in rows:
            writer.write_line(json.dumps(row))
        writer.commit()
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
