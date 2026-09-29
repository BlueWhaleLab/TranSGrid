"""Data generation script for the task

A sample is an NxN board of random digits plus an action sequence; applying
the sequence gives the target board, and the sequence itself is the golden
answer.

Running this file generates the pools under data/raw/ and splits each into
data/train/, data/dev/ and data/test/. The three share no question: pools are
deduped on (initial board, target board) and the split partitions them, and
`assert_disjoint` re-checks it before anything is written. Select checkpoints
on dev, report on test -- doing both on one split inflates the number, and
here `best.pt` is chosen on solve_rate, which is the headline number itself.

Read data back with `load_dataset`, score a prediction with `check`.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

try:
    from .board import ACTIONS, Board
except ImportError:      # running as a script: python utils/data.py
    from board import ACTIONS, Board


class BoardGenerator:
    """Build (initial board, target board, golden answer) triples for eval.

    An item looks like:
        {"id": 0, "n": 4, "steps": 2,
         "initial": [[...], ...], "target": [[...], ...],
         "answer": [["ROW_LEFT", 0], ["BLOCK_CW", 1, 1]]}

    The answer is *a* solution, not necessarily the shortest one: random
    digits repeat, so a shorter sequence may reach the same target. Score
    with `check`, which compares boards rather than action sequences.

    Both generators dedupe on `_question_key`, so a pool never contains the
    same question twice and any partition of it is disjoint by construction.
    """

    def __init__(self, n=4, seed=None):
        if n < 2:
            raise ValueError(f"n must be >= 2 to allow BLOCK actions, got {n}")
        self.n = n
        self.rng = random.Random(seed)
        self._rules = Board([[0] * n for _ in range(n)])   # for apply() only

    # ---- public API ----
    def generate_random_dataset(self, size, steps, path=None):
        """`size` boards, each scrambled by a random action sequence.

        `steps` is the sequence length, given three ways:

            5             every item is 5 actions long
            (1, 9)        drawn uniformly from the inclusive range
            {1: 4, ...}   drawn by weight, one entry per length

        The weighted form is there because uniform lengths spend half the
        data on lengths the model already solves perfectly -- see
        STEP_WEIGHTS below. Note the length is that of the golden answer,
        not the true difficulty: actions can cancel out, so a shorter
        solution may exist.
        """
        if isinstance(steps, dict):
            lengths = sorted(steps)
            weights = [steps[k] for k in lengths]
            if min(lengths) < 1 or min(weights) < 0 or not sum(weights):
                raise ValueError(f"step weights must be non-negative with a "
                                 f"positive total over lengths >= 1: {steps}")
            draw = lambda: self.rng.choices(lengths, weights)[0]
        elif isinstance(steps, tuple):
            draw = lambda: self.rng.randint(*steps)
        else:
            draw = lambda: steps

        items, seen, stale = [], set(), 0
        with tqdm(total=size, desc="generating", unit=" items",
                  unit_scale=True, leave=False) as bar:
            while len(items) < size:
                item = self._sample(self._random_sequence(draw()))
                key = _question_key(item)
                if key in seen:
                    stale += 1
                    if stale > 1000:
                        raise RuntimeError(f"ran out of distinct questions after "
                                           f"{len(items)} items; a {self.n}x{self.n} "
                                           f"board cannot fill size={size}")
                    continue
                seen.add(key)
                stale = 0
                items.append({"id": len(items), **item})
                bar.update(1)
        return self._finish(items, path)

    def generate_atomic_action_dataset(self, repeats=1, path=None):
        """One sample per legal single action, `repeats` boards each.

        Covers every action with every valid argument, so per-action accuracy
        can be read off directly. An NxN board has 8n + 2(n-1)^2 legal
        actions -- 50 for n=4, 242 for n=10 -- and each gets `repeats`
        samples, so the dataset is balanced across actions by construction.
        """
        actions = self._all_actions()
        items, seen = [], set()
        with tqdm(total=len(actions) * repeats, desc="generating",
                  unit=" items", unit_scale=True, leave=False) as bar:
            for action in actions:
                for _ in range(repeats):
                    for _ in range(1000):
                        item = self._sample([action])
                        key = _question_key(item)
                        if key not in seen:
                            break
                    else:
                        raise RuntimeError(f"ran out of distinct boards for "
                                           f"{action} after {len(items)} items")
                    seen.add(key)
                    items.append({"id": len(items), **item})
                bar.update(repeats)
        return self._finish(items, path)

    # ---- internals ----
    def _random_grid(self):
        """An NxN grid of random digits 0-9."""
        return [[self.rng.randrange(10) for _ in range(self.n)]
                for _ in range(self.n)]

    def _random_action(self):
        """One uniformly random legal action for an NxN board."""
        name = self.rng.choice(ACTIONS)
        if name.startswith("BLOCK"):
            return (name, self.rng.randrange(self.n - 1),
                    self.rng.randrange(self.n - 1))
        return (name, self.rng.randrange(self.n))

    def _random_sequence(self, k, max_tries=100):
        """`k` random actions that together are not the identity.

        A sequence like [ROW_LEFT(0), ROW_RIGHT(0)] cancels itself out and
        would leave every board untouched, so it is redrawn.
        """
        for _ in range(max_tries):
            answer = [self._random_action() for _ in range(k)]
            if k == 0 or not self._is_identity(answer):
                return answer
        raise RuntimeError(f"could not draw a non-identity sequence of "
                           f"{k} actions in {max_tries} tries")

    def _all_actions(self):
        """Every legal action on an NxN board, in ACTIONS order."""
        out = []
        for name in ACTIONS:
            if name.startswith("BLOCK"):
                out += [(name, i, j) for i in range(self.n - 1)
                        for j in range(self.n - 1)]
            else:
                out += [(name, i) for i in range(self.n)]
        return out

    def _run(self, grid, answer):
        """Apply `answer` to a copy of `grid` and return the result."""
        g = [row[:] for row in grid]
        for action in answer:
            self._rules.apply(g, action)
        return g

    def _is_identity(self, answer):
        """True if `answer` leaves *every* NxN board unchanged.

        Checked on a grid of distinct cell labels rather than digits, so a
        board that merely happens to be symmetric cannot fool it.
        """
        probe = [[r * self.n + c for c in range(self.n)] for r in range(self.n)]
        return self._run(probe, answer) == probe

    def _sample(self, answer, max_tries=100):  # ensures target != initial
        """One item: a fresh random board scrambled by `answer`.

        Redraws the board while the digits happen to make the target equal
        the initial (e.g. ROW_LEFT on a row of identical digits), which
        would make the task trivial.
        """
        for _ in range(max_tries):
            grid = self._random_grid()
            target = self._run(grid, answer)
            if not answer or target != grid:
                return {"n": self.n, "steps": len(answer),
                        "initial": grid, "target": target, "answer": answer}
        raise RuntimeError(f"{max_tries} random boards were all left unchanged "
                           f"by {answer}; it acts as the identity on a "
                           f"{self.n}x{self.n} board")

    @staticmethod
    def _finish(items, path):
        """Write the dataset as JSON Lines if asked; return the items."""
        if path is not None:
            with open(path, "w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
        return items


def _question_key(item):
    """The identity of a question: its two boards, packed into bytes.
    """
    flat = [d for row in item["initial"] for d in row]
    flat += [d for row in item["target"] for d in row]
    return bytes(flat)


def assert_disjoint(named_keys, label):
    """Fail unless no question appears in two of the named key sets.
    """
    named = list(named_keys)
    for i, (a, ka) in enumerate(named):
        for b, kb in named[i + 1:]:
            shared = ka & kb
            assert not shared, \
                f"{label}: {len(shared)} questions appear in both {a} and {b}"
    sizes = ", ".join(f"{n} {len(k)}" for n, k in named)
    print(f"{label}: splits are disjoint ({sizes} distinct questions)")


def load_dataset(path):
    """Read a JSON Lines dataset, restoring actions as tuples."""
    items = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            item["answer"] = [tuple(a) for a in item["answer"]]
            items.append(item)
    return items


def check(item, answer):
    """True if `answer` turns the item's initial board into its target."""
    try:
        return Board(item["initial"]).execute(answer) == item["target"]
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Split train / dev / test
# ---------------------------------------------------------------------------

def _write_split(items, name, split, n):
    """Write random train/dev to data/<split>.jsonl. 
    Write all other splits to data/<split>/<name>_N<n>_<num>.jsonl.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    if name == "random" and split in {"train", "dev"}:
        path = data_dir / f"{split}.jsonl"
    else:
        path = data_dir / split / f"{name}_N{n}_{len(items)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        for item in tqdm(items, desc=f"writing {path.name}", unit=" items",
                         unit_scale=True, leave=False):
            f.write(json.dumps(item) + "\n")
    print(f"{len(items):>9} items -> {path}")


def _stratify(items, key_of, dev_per_key, test_per_key, train_num, rng, label):
    """Hold out `test_per_key` + `dev_per_key` per group, rest to train.

    Groups `items` by `key_of`, shuffles each group with `rng`, then takes
    `test_per_key` items from every group for test and a further
    `dev_per_key` for dev, so dev and test carry the same shape as each other
    and as the data. What is left over is pooled, shuffled and cut to
    `train_num` -- all of it when `train_num` is None. Returns
    `(test, dev, train)`; `label` only names the group in messages.
    """
    groups = defaultdict(list)
    for item in items:
        groups[key_of(item)].append(item)

    test, dev, rest = [], [], []
    need = test_per_key + dev_per_key
    for key in sorted(groups):
        group = groups[key]
        rng.shuffle(group)
        assert len(group) >= need, \
            f"{label} {key} has only {len(group)} items, need {need}"
        test += group[:test_per_key]
        dev += group[test_per_key:need]
        rest += group[need:]
    for split in (test, dev, rest):
        rng.shuffle(split)
    if train_num is None:
        train_num = len(rest)
    assert len(rest) >= train_num, \
        f"{label}: only {len(rest)} items left for train, need {train_num}"
    print(f"{label}: {len(groups)} groups x ({test_per_key} test + "
          f"{dev_per_key} dev) = {len(test)} test, {len(dev)} dev")
    return test, dev, rest[:train_num]


def split_datasets(seed, datasets):
    """Split each (kind, items, out) triple into data/{train,dev,test}.

    `kind` picks the stratification from SPLIT_SPECS: atomic groups by
    (action, args), random by answer length, so dev and test both contribute
    the same number of items per group. `out` names the files. Each dataset
    gets its own rng, so its split does not depend on the others.

    dev is what train.py should select checkpoints on; test is what goes in
    the paper. Reporting on the split that picked the checkpoint inflates it,
    doubly so here where best.pt is chosen on solve_rate -- the headline
    number itself.
    """
    for kind, items, out in datasets:
        if not items:
            continue
        key_of, dev_per_key, test_per_key, train_num = SPLIT_SPECS[kind]
        test, dev, train = _stratify(items, key_of, dev_per_key, test_per_key,
                                     train_num, random.Random(seed), out)
        assert_disjoint([(name, {_question_key(it) for it in split})
                         for name, split in (("train", train), ("dev", dev),
                                             ("test", test))], out)
        n = items[0]["n"]
        _write_split(train, out, "train", n)
        _write_split(dev, out, "dev", n)
        _write_split(test, out, "test", n)


# How often each answer length is drawn, as a {length: weight} table for
# generate_random_dataset. Weights sum to 100, so they read as percentages.
STEP_WEIGHTS = {1: 4, 2: 4, 3: 5, 4: 6, 5: 9, 6: 13, 7: 17, 8: 20, 9: 22}

N = 6
SEED = 42

# What you control: how many train items, and how many dev and test items
# per group. Set a *_TRAIN_NUM to None to give train everything left over.
#
# On sizing the held-out splits: the binomial standard error on a per-group
# solve rate near 0.85 is 2.1 points at 300 items and 0.8 at 2000, so 300 per
# answer length cannot resolve the 1-2 point gaps between neighbouring rungs
# of the model ladder. Raise the test figure if the paper has to separate
# models that close; dev can stay small, it only has to rank checkpoints of
# one run against each other.
ATOMIC_TRAIN_NUM = 300000
ATOMIC_DEV_PER_ACTION = 30
ATOMIC_TEST_PER_ACTION = 30
RANDOM_TRAIN_NUM = 30000000
RANDOM_DEV_PER_K = 300
RANDOM_TEST_PER_K = 2000

SPLIT_SPECS = {
    "atomic": (lambda it: tuple(it["answer"][0]), ATOMIC_DEV_PER_ACTION,
               ATOMIC_TEST_PER_ACTION, ATOMIC_TRAIN_NUM),
    "random": (lambda it: it["steps"], RANDOM_DEV_PER_K,
               RANDOM_TEST_PER_K, RANDOM_TRAIN_NUM),
}

# How much to generate, derived so that train + eval exactly fit.
# An NxN board has 8n + 2(n-1)^2 legal actions.
N_ACTIONS = 8 * N + 2 * (N - 1) ** 2

# What to generate; comment out a line to skip that dataset.
# (kind, count, output name): kind picks the split rules, out names the files
JOBS = [
    ("atomic", ATOMIC_DEV_PER_ACTION + ATOMIC_TEST_PER_ACTION
              + (ATOMIC_TRAIN_NUM + N_ACTIONS - 1) // N_ACTIONS, "atomic"),
    ("random", RANDOM_TRAIN_NUM
              + len(STEP_WEIGHTS) * (RANDOM_DEV_PER_K + RANDOM_TEST_PER_K),
     "random"),
]



if __name__ == "__main__":
    gen = BoardGenerator(n=N, seed=SEED)
    
    # Raw pools go to <repo>/data/raw/, the splits to
    # <repo>/data/{train,dev,test}/
    raw = Path(__file__).resolve().parent.parent / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # 1. Generate each dataset and write it to data/raw/
    made = []
    for kind, num, out in JOBS:
        tag = f"{N_ACTIONS}x{num}" if kind == "atomic" else f"{num}"
        path = raw / f"{out}_N{gen.n}_{tag}.jsonl"
        if kind == "atomic":
            items = gen.generate_atomic_action_dataset(repeats=num, path=path)
            print(f"{out} dataset: {len(items)} items "
                  f"({len(items) // num} distinct actions x {num}) -> {path}")
        else:
            items = gen.generate_random_dataset(num, steps=STEP_WEIGHTS, path=path)
            print(f"{out} dataset: {len(items)} items -> {path}")
        made.append((kind, items, path, out))
        
    # 2. Read each dataset back from disk and check every golden answer
    for kind, items, path, out in made:
        total = ok = 0
        with open(path) as f:
            for line in tqdm(f, total=len(items), desc=f"verifying {out}",
                             unit=" items", unit_scale=True, leave=False):
                item = json.loads(line)
                ok += check(item, item["answer"])
                total += 1
        assert total == len(items), \
            f"{out}: wrote {len(items)} items but read back {total}"
        print(f"{out}: verified {total} items, {ok} passed, {total - ok} failed")

    # 3. Split into train / dev / test under data/
    split_datasets(seed=SEED,
                   datasets=[(kind, items, out) for kind, items, _, out in made])

