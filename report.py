import argparse
import json
import re
from pathlib import Path

from eval import GROUPINGS, report

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

AXES = ["stop_reason"]  # "deductive_score", "inductive_score", "abductive_level", 

# what a no-argument run reports: a result file, or a directory of them
DEFAULT = RESULTS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="*", type=Path,
                   help=f"result JSONLs, or directories of them; paths are "
                        f"read relative to the repo, not the shell's cwd "
                        f"(default: {DEFAULT.relative_to(ROOT)})")
    p.add_argument("--axes", nargs="+", default=AXES, metavar="FIELD",
                   help=f"one table per axis: a name from GROUPINGS, or any "
                        f"field the rows carry; an axis a file does not carry "
                        f"is skipped (default: {' '.join(AXES)})")
    return p.parse_args()


def expand(paths):
    files = []
    for path in paths:
        path = ROOT / path
        files.extend(sorted(path.glob("*.jsonl")) if path.is_dir() else [path])
    return files


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def infer_beams(path, rows):
    """How many beams the run decoded with; 1 reports as greedy."""
    named = re.search(r"beam(\d+)", path.name)
    if named:
        return int(named.group(1))
    
    return max((row.get("beam_rank", 1) for row in rows), default=1) or 1


def main():
    args = parse_args()
    asked = args.files or [DEFAULT]
    paths = expand(asked)
    if not paths:
        raise SystemExit(f"no result files in "
                         f"{', '.join(str(p) for p in asked)}")

    for i, path in enumerate(paths):
        rows = load_rows(path)
        if not rows:
            raise SystemExit(f"{path} holds no results")
        beams = infer_beams(path, rows)

        how = f"beam {beams} + verify" if beams > 1 else "greedy"
        print(f"{'' if i == 0 else chr(10)}{path.name}: {len(rows)} items, board {rows[0]['n']}x{rows[0]['n']}, {how}\n")

        axes = [a for a in args.axes if a in GROUPINGS or a in rows[0]]
        for missing in [a for a in args.axes if a not in axes]:
            print(f"(no {missing} in this file; skipping that table)")
        report(rows, axes, beams)


if __name__ == "__main__":
    main()
