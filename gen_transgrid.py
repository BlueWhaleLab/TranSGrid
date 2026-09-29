"""Generate normal or inductive board-planning datasets.

The generator exposes twelve short induction rules whose large intermediate
moves cancel to leave small local permutations. Every rule runs at least
three actions. ``none`` produces ordinary
random 1--9 action samples using the same weights as ``utils/data.py``.

CLI examples::

    python gen_transgrid.py --list-rules
    python gen_transgrid.py --rules none --count 1000 \
        --output data/inductive/normal.jsonl
    python gen_transgrid.py --rules 1:100 horizontal_three_cycle:250 \
        --output data/inductive/selected.jsonl
    python gen_transgrid.py --rules none:1000 rule3:200 rule7:300 \
        --output data/inductive/mixed.jsonl
    python gen_transgrid.py --balanced-count 1800 \
        --output data/reasoning/balanced.jsonl

Every row retains the existing six core test fields and adds deductive,
inductive and abductive scores, ``abductive_level``, auditable H/N counts and
applied induction-rule names. Every serialized row is replayed and printed as
PASS before it is staged. The destination is installed only after all rows and
distribution audits pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from pathlib import Path

from utils.dataset_io import AtomicJsonlWriter, DatasetItemValidator
from utils.induction import (
    Action, InductionRule, InductionRuleCatalog,
)
from utils.reasoning import BalancedQuotaPlanner, ReasoningScorer
from utils.sample_factory import BoardSampleFactory

ROOT = Path(__file__).resolve().parent


class TranSGridDatasetGenerator:
    """Generate verified normal and induction-rule TranSGrid datasets.

    This is a compatibility façade over dedicated rule, scoring, planning,
    sampling, validation and output classes. Reusing an instance advances the
    shared RNG; call :meth:`reset` to reproduce from its configured seed.
    """

    Rule = InductionRule
    NORMAL_SOURCE = InductionRuleCatalog.NORMAL_SOURCE
    REQUIRED_FIELDS = DatasetItemValidator.REQUIRED_FIELDS
    REASONING_FIELDS = DatasetItemValidator.REASONING_FIELDS
    DEDUCTIVE_LEVELS = BalancedQuotaPlanner.DEDUCTIVE_LEVELS
    INDUCTIVE_LEVELS = BalancedQuotaPlanner.INDUCTIVE_LEVELS
    ABDUCTIVE_LEVELS = BalancedQuotaPlanner.ABDUCTIVE_LEVELS
    MAX_ACTION_TRIES = BoardSampleFactory.MAX_ACTION_TRIES
    MAX_BOARD_TRIES = BoardSampleFactory.MAX_BOARD_TRIES
    MAX_DISTINCT_TRIES = 1000
    MIN_BALANCED_COUNT = BalancedQuotaPlanner.MIN_COUNT

    def __init__(self, n: int = 6, seed: int = 42):
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        self.n = n
        self.seed = seed
        self.rng = random.Random(seed)
        self.rule_catalog = InductionRuleCatalog(n, self.rng)
        self.scorer = ReasoningScorer(n)
        self.quota_planner = BalancedQuotaPlanner(self.rng)
        self.rules = self.rule_catalog.rules
        self.rules_by_name = self.rule_catalog.by_name
        self.validator = DatasetItemValidator(
            n, self.scorer, set(self.rules_by_name))
        self.sample_factory = BoardSampleFactory(
            n, self.rng, self.rule_catalog, self.scorer)

    def reset(self, seed: int | None = None) -> None:
        """Reset the shared RNG used by the generator and its collaborators."""
        if seed is not None:
            self.seed = seed
        self.rng.seed(self.seed)

    def resolve_rule(self, token: str | int) -> InductionRule:
        return self.rule_catalog.resolve_rule(token)

    def resolve_source(
            self, source: str | int | InductionRule | None
            ) -> InductionRule | None:
        return self.rule_catalog.resolve_source(source)

    @staticmethod
    def source_name(rule: InductionRule | None) -> str:
        return InductionRuleCatalog.source_name(rule)

    def parse_source_counts(
            self, specs: list[str], default_count: int
            ) -> list[tuple[InductionRule | None, int]]:
        return self.rule_catalog.parse_source_counts(specs, default_count)

    # ---- compatibility delegates to reasoning collaborators ----
    @staticmethod
    def abductive_level(score: float) -> int:
        return ReasoningScorer.abductive_level(score)

    def trajectory_reasoning(
            self, initial: list[list[int]], actions: list[Action]
            ) -> tuple[list[list[int]], dict[str, int | float]]:
        return self.scorer.score_trajectory(initial, actions)

    def make_reasoning_item(
            self, item_id: int, length: int,
            induction_count: int) -> tuple[dict, int]:
        return self.sample_factory.make_reasoning_item(
            item_id, length, induction_count)

    def moved_position_count(self, actions: list[Action]) -> int:
        return self.sample_factory.moved_position_count(actions)

    def make_item(
            self, item_id: int = 0,
            source: str | int | InductionRule | None = None
            ) -> tuple[dict, int]:
        return self.sample_factory.make_item(item_id, source)

    def validate_serialized_item(self, line: str) -> bool:
        return self.validator.validate(line)

    # ---- complete verified dataset workflow ----
    def generate(
            self, source_counts: list[tuple[InductionRule | None, int]],
            output: str | Path, overwrite: bool = False) -> int:
        """Generate, verify, then atomically install one JSONL dataset."""
        if any(count < 0 for _, count in source_counts):
            raise ValueError("source counts must be >= 0")
        total = sum(count for _, count in source_counts)
        if total <= 0:
            raise ValueError("source counts must sum to a positive number")

        output = Path(output)
        seen: set[bytes] = set()
        item_id = 0
        with AtomicJsonlWriter(output, overwrite) as writer:
            for rule, count in source_counts:
                for _ in range(count):
                    for _ in range(self.MAX_DISTINCT_TRIES):
                        item, moved = self.make_item(item_id, rule)
                        key = self.sample_factory.question_key(item)
                        if key not in seen:
                            break
                    else:
                        raise RuntimeError(
                            f"could not make a distinct question for item "
                            f"{item_id} ({self.source_name(rule)})")
                    seen.add(key)

                    line = json.dumps(item)
                    passed = self.validate_serialized_item(line)
                    status = "PASS" if passed else "FAIL"
                    print(f"VERIFY {item_id + 1:>7}/{total:<7} "
                          f"id={item_id:<7} "
                          f"source={self.source_name(rule):<27} "
                          f"steps={item['steps']} moved={moved:<3} "
                          f"{status}", flush=True)
                    if not passed:
                        raise RuntimeError(
                            f"gold answer failed for item {item_id} "
                            f"({self.source_name(rule)})")
                    writer.write_line(line)
                    item_id += 1
            writer.commit()

        print(f"VERIFIED {item_id}/{total} gold answers: all PASS", flush=True)
        print(f"SAVED {item_id} items -> {output}", flush=True)
        return item_id

    def generate_balanced(
            self, total: int, output: str | Path,
            overwrite: bool = False) -> int:
        """Generate a dataset balanced on the deductive and inductive axes.
        """
        cells, quotas = self.quota_planner.plan(total, self.n)
        print("BALANCED TARGET QUOTAS")
        for axis in ("deductive", "inductive"):
            values = " ".join(
                f"{level}:{count}"
                for level, count in sorted(quotas[axis].items()))
            print(f"  {axis:<10} {values}")
        print(f"  {'abductive':<10} not constrained; reported as observed")

        output = Path(output)
        seen: set[bytes] = set()
        observed = {
            "deductive": Counter(),
            "inductive": Counter(),
            "abductive": Counter(),
        }
        item_id = 0
        with AtomicJsonlWriter(output, overwrite) as writer:
            order = sorted(cells, key=lambda cell: (-cell[1], -cell[0]))
            for length, induction_count in order:
                for _ in range(cells[(length, induction_count)]):
                    total_tries = 0
                    for _ in range(self.MAX_DISTINCT_TRIES):
                        item, tries = self.make_reasoning_item(
                            item_id, length, induction_count)
                        total_tries += tries
                        key = self.sample_factory.question_key(item)
                        if key not in seen:
                            break
                    else:
                        raise RuntimeError(
                            f"could not make a distinct reasoning item for "
                            f"L={length}, K={induction_count}")
                    seen.add(key)

                    line = json.dumps(item)
                    passed = self.validate_serialized_item(line)
                    status = "PASS" if passed else "FAIL"
                    print(
                        f"VERIFY {item_id + 1:>7}/{total:<7} "
                        f"id={item_id:<7} L={length} K={induction_count} "
                        f"A={item['abductive_level']} "
                        f"score={item['abductive_score']:.6f} "
                        f"Leff={item['effective_length']:.2f} "
                        f"N={item['net_changed']:<2} "
                        f"draws={total_tries:<5} {status}", flush=True)
                    if not passed:
                        raise RuntimeError(
                            f"gold answer or reasoning metadata failed for "
                            f"item {item_id}")
                    writer.write_line(line)
                    observed["deductive"][length] += 1
                    observed["inductive"][induction_count] += 1
                    observed["abductive"][item["abductive_level"]] += 1
                    item_id += 1

            for axis, expected in quotas.items():
                actual = {level: observed[axis][level]
                          for level in expected}
                if actual != expected:
                    raise AssertionError(
                        f"{axis} quota audit failed: expected "
                        f"{expected}, got {actual}")
            writer.commit()

        print(f"VERIFIED {item_id}/{total} gold answers: all PASS", flush=True)
        for axis in ("deductive", "inductive"):
            print(f"AUDIT {axis}: {dict(sorted(observed[axis].items()))}")
        abductive = dict(sorted(observed["abductive"].items()))
        shares = " ".join(f"{level}:{count} ({count / item_id:.1%})"
                          for level, count in abductive.items())
        print(f"OBSERVED abductive: {shares}")
        print(f"SAVED {item_id} balanced items -> {output}", flush=True)
        return item_id

    def generate_from_specs(
            self, specs: list[str], output: str | Path,
            default_count: int = 100, overwrite: bool = False) -> int:
        """Parse SOURCE[:COUNT] specs and generate a verified dataset."""
        return self.generate(
            self.parse_source_counts(specs, default_count), output, overwrite)

    def print_rules(self) -> None:
        self.rule_catalog.print_rules()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rules", nargs="+", default=["none"],
                   metavar="SOURCE[:COUNT]",
                   help="none for normal generation, all for all twelve rules, "
                        "or rules by name/number; :COUNT overrides --count "
                        "(default: none)")
    p.add_argument("--count", type=int, default=100,
                   help="items per source unless SOURCE:COUNT overrides it "
                        "(default: 100)")
    p.add_argument("--balanced-count", type=int,
                   help="generate this many samples with near-equal "
                        "deductive 1-9, inductive 0-3 and abductive 1-5 "
                        "marginals (minimum: 180)")
    p.add_argument("--n", type=int, default=6,
                   help="board side length (default: 6)")
    p.add_argument("--seed", type=int, default=42,
                   help="rule parameters and board RNG seed (default: 42)")
    p.add_argument("--output",
                   help="destination JSONL, relative to the repository root")
    p.add_argument("--overwrite", action="store_true",
                   help="replace --output if it already exists")
    p.add_argument("--list-rules", action="store_true",
                   help="print normal mode and the ten induction rules")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        generator = TranSGridDatasetGenerator(args.n, args.seed)
        if args.list_rules:
            generator.print_rules()
            return
        if not args.output:
            raise ValueError(
                "--output is required unless --list-rules is used")

        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output
        if args.balanced_count is not None:
            if args.rules != ["none"]:
                raise ValueError(
                    "--balanced-count cannot be combined with --rules; "
                    "the balanced generator selects rule compositions")
            print(f"board {generator.n}x{generator.n}, seed {generator.seed}")
            generator.generate_balanced(
                args.balanced_count, output, args.overwrite)
        else:
            source_counts = generator.parse_source_counts(
                args.rules, args.count)
            print(f"board {generator.n}x{generator.n}, seed {generator.seed}")
            for rule, count in source_counts:
                print(f"  {count:>7}  {generator.source_name(rule)}")
            generator.generate(source_counts, output, args.overwrite)
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
