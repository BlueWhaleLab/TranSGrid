"""Validation and atomic JSONL output helpers for generated datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import TextIO

from .data import check
from .reasoning import BalancedQuotaPlanner, ReasoningScorer


class DatasetItemValidator:
    """Replay serialized items and independently verify reasoning metadata."""

    REQUIRED_FIELDS = {"id", "n", "steps", "initial", "target", "answer"}
    REASONING_FIELDS = {
        "deductive_score", "inductive_score", "induction_rules",
        "abductive_score", "abductive_level", "effective_length",
        "abductive_hidden", "net_changed", "visibly_affected",
    }

    def __init__(
            self, n: int, scorer: ReasoningScorer,
            rule_names: set[str]):
        self.n = n
        self.scorer = scorer
        self.rule_names = rule_names

    def validate(self, line: str) -> bool:
        """Replay the exact JSON representation entering the dataset."""
        item = json.loads(line)
        fields = set(item)
        if fields not in (self.REQUIRED_FIELDS,
                          self.REQUIRED_FIELDS | self.REASONING_FIELDS):
            return False
        if (item["n"] != self.n
                or item["steps"] != len(item["answer"])
                or not check(item, item["answer"])):
            return False
        if fields == self.REQUIRED_FIELDS:
            return True

        actions = [tuple(action) for action in item["answer"]]
        target, metrics = self.scorer.score_trajectory(
            item["initial"], actions)
        # bounds track the planner's levels rather than repeating them: the deductive range moved to 12 so that K=3 compositions, which need MIN_RULE_STEPS * 3 actions, have more than a single length to sit in
        lengths = BalancedQuotaPlanner.DEDUCTIVE_LEVELS
        counts = BalancedQuotaPlanner.INDUCTIVE_LEVELS
        if (target != item["target"]
                or item["deductive_score"] != item["steps"]
                or not min(lengths) <= item["deductive_score"] <= max(lengths)
                or item["inductive_score"] != len(
                    item["induction_rules"])
                or not min(counts) <= item["inductive_score"] <= max(counts)):
            return False
        if any(name not in self.rule_names
               for name in item["induction_rules"]):
            return False
        for key in ("abductive_level", "abductive_hidden",
                    "net_changed", "visibly_affected"):
            if item[key] != metrics[key]:
                return False
        return (abs(item["abductive_score"]
                    - metrics["abductive_score"]) < 1e-12
                and abs(item["effective_length"]
                        - metrics["effective_length"]) < 1e-12)


class AtomicJsonlWriter:
    """Stage JSONL output and expose it only after an explicit commit."""

    def __init__(self, output: str | Path, overwrite: bool = False):
        self.output = Path(output)
        self.overwrite = overwrite
        self.temp_path: Path | None = None
        self.stream: TextIO | None = None
        self.committed = False

    def __enter__(self) -> AtomicJsonlWriter:
        if self.output.exists() and not self.overwrite:
            raise FileExistsError(
                f"{self.output} already exists; pass --overwrite to replace it")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.output.name}.", suffix=".tmp",
            dir=self.output.parent, text=True)
        self.temp_path = Path(temp_name)
        self.stream = os.fdopen(fd, "w")
        return self

    def write_line(self, line: str) -> None:
        if self.stream is None or self.stream.closed:
            raise RuntimeError("JSONL writer is not open")
        self.stream.write(line + "\n")

    def commit(self) -> None:
        if self.stream is None or self.temp_path is None:
            raise RuntimeError("JSONL writer is not open")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        if self.output.exists() and not self.overwrite:
            raise FileExistsError(
                f"{self.output} appeared during generation; pass --overwrite "
                "to replace it")
        os.replace(self.temp_path, self.output)
        self.committed = True

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.stream is not None and not self.stream.closed:
            self.stream.close()
        if (not self.committed and self.temp_path is not None
                and self.temp_path.exists()):
            self.temp_path.unlink()

