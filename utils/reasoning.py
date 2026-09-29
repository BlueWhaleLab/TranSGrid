"""Reasoning scores and balanced marginal planning for TranSGrid."""

from __future__ import annotations

from collections import defaultdict
import math
from random import Random

from .board import Board
from .induction import Action


class ReasoningScorer:
    """Replay solutions and compute deductive/abductive metadata."""

    # Expected number of cells whose value differs between the initial and the
    # final board after L purely random actions. The curve saturates because
    # later actions overwrite earlier ones, and it is a property of the board
    # size and the action set alone -- not of any model. Estimated from 4800
    # rule-free trajectories, 400 per length; standard errors are at most 0.25
    # cells. Re-estimate whenever the board size or the action set changes.
    EXPECTED_NET_CHANGE = {
        1: 7.2, 2: 12.5, 3: 16.0, 4: 19.1, 5: 21.3, 6: 22.9,
        7: 24.4, 8: 25.0, 9: 26.3, 10: 27.0, 11: 27.3, 12: 27.8,
    }
    EXPECTED_BOARD_SIZE = 6

    HIDDEN_MIDPOINT = 6.0
    HIDDEN_SCALE = 2.0

    def __init__(self, n: int):
        self.n = n

    @staticmethod
    def abductive_level(score: float) -> int:
        """Map [0, 1] to five intervals: [0,.2), ..., [.8,1]."""
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"abductive score must be in [0, 1], got {score}")
        return min(5, int(score * 5) + 1)

    @classmethod
    def effective_length(cls, net_changed: int) -> float:
        """How many random actions would leave a footprint this large.
        """
        table = cls.EXPECTED_NET_CHANGE
        lengths = sorted(table)
        first = lengths[0]
        if net_changed <= table[first]:
            return net_changed / table[first] * first
        for low, high in zip(lengths, lengths[1:]):
            if table[low] <= net_changed <= table[high]:
                span = table[high] - table[low]
                return low + (net_changed - table[low]) / span * (high - low)
        return float(lengths[-1])

    @classmethod
    def abductive_score(cls, steps: int, net_changed: int) -> float:
        """Fraction of the trajectory the observation fails to witness.
        """
        hidden_actions = steps - cls.effective_length(net_changed)
        z = (hidden_actions - cls.HIDDEN_MIDPOINT) / cls.HIDDEN_SCALE
        return 1.0 / (1.0 + math.exp(-z))

    def score_trajectory(
            self, initial: list[list[int]], actions: list[Action]
            ) -> tuple[list[list[int]], dict[str, int | float]]:
        """Replay a trajectory and compute the paper's H, N and A_abd."""
        board = Board(initial)
        if board.n != self.n:
            raise ValueError(
                f"expected a {self.n}x{self.n} board, got {board.n}x{board.n}")
        grid = [row[:] for row in initial]
        visible: set[tuple[int, int]] = set()
        for action in actions:
            before = [row[:] for row in grid]
            board.apply(grid, action)
            visible.update(
                (r, c) for r in range(self.n) for c in range(self.n)
                if before[r][c] != grid[r][c])

        net_changed = sum(
            initial[r][c] != grid[r][c]
            for r in range(self.n) for c in range(self.n))
        hidden = sum(initial[r][c] == grid[r][c] for r, c in visible)
        score = self.abductive_score(len(actions), net_changed)
        return grid, {
            "abductive_score": score,
            "abductive_level": self.abductive_level(score),
            "effective_length": self.effective_length(net_changed),
            "abductive_hidden": hidden,
            "net_changed": net_changed,
            "visibly_affected": hidden + net_changed,
        }


class BalancedQuotaPlanner:
    """Allocate uniform (deductive, inductive) quotas over feasible cells.

    Only the deductive and inductive marginals are balanced.
    """

    DEDUCTIVE_LEVELS = tuple(range(1, 13))
    INDUCTIVE_LEVELS = tuple(range(4))
    ABDUCTIVE_LEVELS = tuple(range(1, 6))
    
    MIN_COUNT = 240
    MIN_RULE_STEPS = 3

    def __init__(self, rng: Random):
        self.rng = rng

    def plan(
            self, total: int, n: int
            ) -> tuple[dict[tuple[int, int], int], dict[str, dict[int, int]]]:
        """Allocate exact deductive/inductive quotas to feasible (L, K) cells."""
        if n < 4:
            raise ValueError(
                "balanced generation needs n >= 4: on at most nine cells a "
                "non-trivial permutation cannot leave the target differing "
                "from the initial board often enough to sample")
        if total < self.MIN_COUNT:
            raise ValueError(
                f"balanced count must be >= {self.MIN_COUNT}; this "
                "guarantees stable coverage of all 12x4 levels")

        quotas = {
            "deductive": self._uniform_quotas(
                total, self.DEDUCTIVE_LEVELS),
            "inductive": self._uniform_quotas(
                total, self.INDUCTIVE_LEVELS),
        }
        if not self.pair_marginals_feasible(
                quotas["deductive"], quotas["inductive"]):
            raise ValueError(
                "the rounded deductive/inductive quotas are infeasible; "
                "try a nearby count or a different seed")

        for _restart in range(200):
            left_l = quotas["deductive"].copy()
            left_k = quotas["inductive"].copy()
            cells: defaultdict[tuple[int, int], int] = defaultdict(int)
            failed = False
            for _ in range(total):
                candidates = []
                weights = []
                for length in self.DEDUCTIVE_LEVELS:
                    if not left_l[length]:
                        continue
                    for count in self.INDUCTIVE_LEVELS:
                        if (not left_k[count]
                                or not self.cell_supported(length, count)):
                            continue
                        left_l[length] -= 1
                        left_k[count] -= 1
                        feasible = self.pair_marginals_feasible(left_l, left_k)
                        left_l[length] += 1
                        left_k[count] += 1
                        if not feasible:
                            continue
                        candidates.append((length, count))
                        # drain the levels that have the most left to place,
                        # so no level is stranded holding the last few slots
                        weights.append(left_l[length] * left_k[count])
                if not candidates:
                    failed = True
                    break
                cell = self.rng.choices(candidates, weights=weights)[0]
                left_l[cell[0]] -= 1
                left_k[cell[1]] -= 1
                cells[cell] += 1
            if (not failed and not any(left_l.values())
                    and not any(left_k.values())):
                return dict(cells), quotas

        raise RuntimeError(
            "could not allocate feasible balanced reasoning quotas after "
            "200 schedules; try a different seed")

    def _uniform_quotas(
            self, total: int, levels: tuple[int, ...]) -> dict[int, int]:
        """Split total with a maximum one-item difference between levels."""
        base, remainder = divmod(total, len(levels))
        order = list(levels)
        self.rng.shuffle(order)
        quotas = {level: base for level in levels}
        for level in order[:remainder]:
            quotas[level] += 1
        return quotas

    @classmethod
    def pair_marginals_feasible(
            cls,
            length_quota: dict[int, int],
            induction_quota: dict[int, int]) -> bool:
        """Check whether remaining L/K quotas can satisfy L >= MIN_RULE_STEPS*K."""
        if min((*length_quota.values(), *induction_quota.values()),
               default=0) < 0:
            return False
        if sum(length_quota.values()) != sum(induction_quota.values()):
            return False
        top = max(cls.INDUCTIVE_LEVELS)
        for k in range(1, top + 1):
            need = sum(induction_quota[level] for level in range(k, top + 1))
            capacity = sum(count for length, count in length_quota.items()
                           if length >= cls.MIN_RULE_STEPS * k)
            if need > capacity:
                return False
        return True

    @classmethod
    def cell_supported(cls, length: int, induction_count: int) -> bool:
        """Whether K intact rule chunks fit inside an answer of L actions."""
        return length >= cls.MIN_RULE_STEPS * induction_count
