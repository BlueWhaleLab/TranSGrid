"""Board/action construction for ordinary and reasoning-controlled samples."""

from __future__ import annotations

from itertools import product
from random import Random

from .board import ACTIONS, Board
from .data import STEP_WEIGHTS
from .induction import Action, InductionRule, InductionRuleCatalog
from .reasoning import BalancedQuotaPlanner, ReasoningScorer


class BoardSampleFactory:
    """Construct samples while enforcing action and rule contracts."""

    MAX_ACTION_TRIES = 500
    MAX_BOARD_TRIES = 1000

    def __init__(
            self, n: int, rng: Random, rule_catalog: InductionRuleCatalog,
            scorer: ReasoningScorer):
        self.n = n
        self.rng = rng
        self.rule_catalog = rule_catalog
        self.scorer = scorer
        self.rules = rule_catalog.rules
        self._rule_combo_cache: dict[
            tuple[int, int], tuple[tuple[InductionRule, ...], ...]] = {}

    def make_reasoning_item(
            self, item_id: int, length: int,
            induction_count: int) -> tuple[dict, int]:
        """Sample one item in an exact (L, K) cell.

        """
        if not BalancedQuotaPlanner.cell_supported(length, induction_count):
            raise ValueError(
                f"unsupported reasoning cell L={length}, "
                f"K={induction_count}")
        for attempt in range(1, self.MAX_BOARD_TRIES + 1):
            actions, rule_names = self._reasoning_actions(
                length, induction_count)
            initial = self._random_grid()
            target, metrics = self.scorer.score_trajectory(initial, actions)
            if target == initial:
                continue
            item = {
                "id": item_id,
                "n": self.n,
                "steps": length,
                "initial": initial,
                "target": target,
                "answer": [list(action) for action in actions],
                "deductive_score": length,
                "inductive_score": induction_count,
                "induction_rules": rule_names,
                **metrics,
            }
            return item, attempt
        raise RuntimeError(
            f"could not sample a non-identity L={length}, "
            f"K={induction_count} item in {self.MAX_BOARD_TRIES} tries")

    def make_item(
            self, item_id: int = 0,
            source: str | int | InductionRule | None = None
            ) -> tuple[dict, int]:
        """Generate one visible ordinary or single-rule item."""
        rule = self.rule_catalog.resolve_source(source)
        if rule is None:
            actions = self._normal_actions()
            moved = self.moved_position_count(actions)
        else:
            if self.n < rule.min_n:
                raise ValueError(
                    f"rule {rule.name} needs n >= {rule.min_n}, got {self.n}")
            actions = rule.build()
            if len(actions) != rule.steps or len(actions) > 6:
                raise AssertionError(
                    f"{rule.name} built {len(actions)} actions, expected "
                    f"{rule.steps} and at most 6")
            moved = self.moved_position_count(actions)
            if moved != rule.moved_positions:
                raise AssertionError(
                    f"{rule.name} moved {moved} positions, expected "
                    f"{rule.moved_positions}: {actions}")

        induction_count = 0 if rule is None else 1
        rule_names = [] if rule is None else [rule.name]
        for _ in range(self.MAX_BOARD_TRIES):
            initial = self._random_grid()
            target, metrics = self.scorer.score_trajectory(initial, actions)
            if target != initial:
                length = len(actions)
                return ({"id": item_id,
                         "n": self.n,
                         "steps": length,
                         "initial": initial,
                         "target": target,
                         "answer": [list(action) for action in actions],
                         "deductive_score": length,
                                  "inductive_score": induction_count,
                                  "induction_rules": rule_names,
                         **metrics},
                        moved)
        label = self.rule_catalog.source_name(rule)
        raise RuntimeError(
            f"{label}: {self.MAX_BOARD_TRIES} random boards all hid the "
            "action sequence's non-identity effect")

    def moved_position_count(self, actions: list[Action]) -> int:
        """Count moved positions on a distinct-label probe board."""
        rules = Board([[0] * self.n for _ in range(self.n)])
        probe = [[r * self.n + c for c in range(self.n)]
                 for r in range(self.n)]
        for action in actions:
            rules.apply(probe, action)
        return sum(probe[r][c] != r * self.n + c
                   for r in range(self.n) for c in range(self.n))

    @staticmethod
    def question_key(item: dict) -> bytes:
        digits = [v for row in item["initial"] for v in row]
        digits += [v for row in item["target"] for v in row]
        return bytes(digits)

    def _rule_combinations(
            self, length: int, induction_count: int
            ) -> tuple[tuple[InductionRule, ...], ...]:
        key = (length, induction_count)
        if key not in self._rule_combo_cache:
            combos = product(self.rules, repeat=induction_count)
            self._rule_combo_cache[key] = tuple(
                combo for combo in combos
                if all(self.n >= rule.min_n for rule in combo)
                and sum(rule.steps for rule in combo) <= length)
        return self._rule_combo_cache[key]

    def _reasoning_actions(
            self, length: int, induction_count: int
            ) -> tuple[list[Action], list[str]]:
        """Build exactly L actions containing exactly K intact rule chunks."""
        combos = self._rule_combinations(length, induction_count)
        if not combos:
            raise ValueError(
                f"no {induction_count}-rule composition fits action length "
                f"{length} on a {self.n}x{self.n} board")
        for _ in range(self.MAX_ACTION_TRIES):
            combo = self.rng.choice(combos)
            chunks: list[list[Action]] = []
            for rule in combo:
                actions = rule.build()
                if (len(actions) != rule.steps
                        or self.moved_position_count(actions)
                        != rule.moved_positions):
                    raise AssertionError(
                        f"induction rule contract failed: {rule.name}")
                chunks.append(actions)
            filler = length - sum(len(chunk) for chunk in chunks)
            chunks.extend([[self._random_action()] for _ in range(filler)])
            self.rng.shuffle(chunks)
            actions = [action for chunk in chunks for action in chunk]
            if len(actions) == length and self.moved_position_count(actions):
                return actions, [rule.name for rule in combo]
        raise RuntimeError(
            f"could not build a non-identity L={length}, "
            f"K={induction_count} sequence")

    def _random_grid(self) -> list[list[int]]:
        return [[self.rng.randrange(10) for _ in range(self.n)]
                for _ in range(self.n)]

    def _random_action(self) -> Action:
        name = self.rng.choice(ACTIONS)
        if name.startswith("BLOCK"):
            return (name, self.rng.randrange(self.n - 1),
                    self.rng.randrange(self.n - 1))
        return name, self.rng.randrange(self.n)

    def _normal_actions(self) -> list[Action]:
        """A non-identity ordinary 1--9 step sequence using STEP_WEIGHTS."""
        lengths = sorted(STEP_WEIGHTS)
        weights = [STEP_WEIGHTS[k] for k in lengths]
        steps = self.rng.choices(lengths, weights)[0]
        for _ in range(self.MAX_ACTION_TRIES):
            actions = [self._random_action() for _ in range(steps)]
            if self.moved_position_count(actions):
                return actions
        raise RuntimeError(
            f"could not draw a non-identity {steps}-action sequence in "
            f"{self.MAX_ACTION_TRIES} tries")
