"""Induction-rule definitions and source selection for TranSGrid."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable


Action = tuple[str | int, ...]


@dataclass(frozen=True)
class InductionRule:
    """A parameterized short program and its local-effect contract."""

    name: str
    description: str
    steps: int
    moved_positions: int
    min_n: int
    build: Callable[[], list[Action]]


class InductionRuleCatalog:
    """Own the rule strategies and resolve user-facing selectors."""

    NORMAL_SOURCE = "none"

    def __init__(self, n: int, rng: Random):
        self.n = n
        self.rng = rng
        self.rules = self._build_registry()
        self.by_name = {rule.name: rule for rule in self.rules}

    def _build_registry(self) -> tuple[InductionRule, ...]:
        """The stable public rule order used by 1, rule1, ..., rule12.
        """
        return (
            InductionRule(
                "horizontal_swap_mixed",
                "swap horizontal neighbors with block + row/column actions",
                5, 2, 2, self._horizontal_swap_mixed),
            InductionRule(
                "diagonal_swap_blocks",
                "swap one diagonal pair inside a 3x3 neighborhood",
                5, 2, 3, self._diagonal_swap_blocks),
            InductionRule(
                "bent_four_cycle",
                "cycle four cells in a bent shape across a 2x3 patch",
                3, 4, 3, self._bent_four_cycle),
            InductionRule(
                "wide_bent_four_cycle",
                "cycle four cells in a wide bent 2x3 patch",
                3, 4, 3, self._wide_bent_four_cycle),
            InductionRule(
                "tall_bent_four_cycle",
                "cycle four cells in a tall bent 3x2 patch",
                3, 4, 3, self._tall_bent_four_cycle),
            InductionRule(
                "tall_bent_flipped",
                "cycle four cells in a mirrored tall bent 3x2 patch",
                3, 4, 3, self._tall_bent_flipped),
            InductionRule(
                "horizontal_three_cycle",
                "cycle three consecutive cells in one row",
                4, 3, 3, self._horizontal_three_cycle),
            InductionRule(
                "vertical_three_cycle",
                "cycle three consecutive cells in one column",
                4, 3, 3, self._vertical_three_cycle),
            InductionRule(
                "l_three_cycle",
                "cycle three cells in an L shape",
                4, 3, 2, self._l_three_cycle),
            InductionRule(
                "double_horizontal_swap",
                "make two offset horizontal swaps in a 2x3 patch",
                4, 4, 3, self._double_horizontal_swap),
            InductionRule(
                "five_cycle_2x3",
                "cycle five boundary cells in a 2x3 patch",
                4, 5, 3, self._five_cycle_2x3),
            InductionRule(
                "vertical_swap_blocks",
                "swap vertical neighbors using five block rotations",
                5, 2, 3, self._vertical_swap_blocks),
        )

    def _origin(self, height: int, width: int) -> tuple[int, int]:
        """Uniform top-left coordinate for a non-wrapping local rectangle."""
        if self.n < max(height, width):
            raise ValueError(
                f"board {self.n}x{self.n} is too small for a "
                f"{height}x{width} rule")
        return (self.rng.randrange(self.n - height + 1),
                self.rng.randrange(self.n - width + 1))

    def _horizontal_swap_mixed(self) -> list[Action]:
        """Swap horizontal neighbors 7 and 2 with five broad actions.

        Initial (r=0, c=0)::

            7 2 9 4
            1 8 3 6
            5 0 7 2
            9 4 1 8

        Target::

            2 7 9 4
            1 8 3 6
            5 0 7 2
            9 4 1 8
        """
        r, c = self._origin(2, 2)
        return [("BLOCK_CW", r, c),
                ("COL_DOWN", c),
                ("ROW_LEFT", r + 1),
                ("COL_UP", c),
                ("ROW_RIGHT", r + 1)]

    def _diagonal_swap_blocks(self) -> list[Action]:
        """Swap diagonal cells 7 and 8 with five block rotations.

        Initial (r=0, c=0)::

            7 2 9 4
            1 8 3 6
            5 0 7 2
            9 4 1 8

        Target::

            8 2 9 4
            1 7 3 6
            5 0 7 2
            9 4 1 8
        """
        r, c = self._origin(3, 3)
        return [("BLOCK_CW", r, c + 1),
                ("BLOCK_CCW", r + 1, c),
                ("BLOCK_CCW", r, c),
                ("BLOCK_CCW", r, c + 1),
                ("BLOCK_CW", r + 1, c)]

    def _bent_four_cycle(self) -> list[Action]:
        """Cycle 7, 2, 9 and 3 across a bent 2x3 patch.

        Initial (r=0, c=0)::

            7 2 9 4
            1 8 3 6
            5 0 7 2
            9 4 1 8

        Target::

            2 3 7 4
            1 8 9 6
            5 0 7 2
            9 4 1 8
        """
        r, c = self._origin(2, 3)
        return [("BLOCK_CW", r, c),
                ("BLOCK_CW", r, c + 1),
                ("BLOCK_CCW", r, c)]

    def _wide_bent_four_cycle(self) -> list[Action]:
        """Cycle 7, 1, 3 and 8 across a wide bent 2x3 patch.

        The mirror image of :meth:`_bent_four_cycle`: the same three block
        rotations run right to left, so the bend opens the other way.

        Initial (r=0, c=0)::

            7 2 9 4
            1 8 3 6
            5 0 7 2
            9 4 1 8

        Target::

            1 2 9 4
            3 7 8 6
            5 0 7 2
            9 4 1 8
        """
        r, c = self._origin(2, 3)
        return [("BLOCK_CW", r, c + 1),
                ("BLOCK_CW", r, c),
                ("BLOCK_CCW", r, c + 1)]

    def _tall_bent_four_cycle(self) -> list[Action]:
        """Cycle 2, 8, 5 and 0 down a tall bent 3x2 patch.

        The transpose of :meth:`_bent_four_cycle`: two stacked blocks instead
        of two side by side, so the four cells bend down a column pair.

        Initial (r=0, c=0)::

            7 2 9 4
            1 8 3 6
            5 0 7 2
            9 4 1 8

        Target::

            7 8 9 4
            1 5 3 6
            0 2 7 2
            9 4 1 8
        """
        r, c = self._origin(3, 2)
        return [("BLOCK_CW", r, c),
                ("BLOCK_CW", r + 1, c),
                ("BLOCK_CCW", r, c)]

    def _tall_bent_flipped(self) -> list[Action]:
        """Cycle 7, 2, 1 and 5 up a mirrored tall bent 3x2 patch.

        The mirror image of :meth:`_tall_bent_four_cycle`, so the four rules
        of length three cover all four bend orientations between them.

        Initial (r=0, c=0)::

            7 2 9 4
            1 8 3 6
            5 0 7 2
            9 4 1 8

        Target::

            5 7 9 4
            2 8 3 6
            1 0 7 2
            9 4 1 8
        """
        r, c = self._origin(3, 2)
        return [("BLOCK_CW", r + 1, c),
                ("BLOCK_CW", r, c),
                ("BLOCK_CCW", r + 1, c)]

    def _horizontal_three_cycle(self) -> list[Action]:
        """Cycle the adjacent digits 7, 2 and 9 horizontally.

        Initial (row=0, centre=1)::

            7 2 9 4
            1 8 3 6
            5 0 7 2
            9 4 1 8

        Target::

            9 7 2 4
            1 8 3 6
            5 0 7 2
            9 4 1 8
        """
        if self.n < 3:
            raise ValueError("horizontal_three_cycle needs n >= 3")
        row = self.rng.randrange(self.n)
        centre = self.rng.randrange(1, self.n - 1)
        return [("ROW_LEFT", row),
                ("COL_LEFT", centre),
                ("ROW_RIGHT", row),
                ("COL_LEFT", centre)]

    def _vertical_three_cycle(self) -> list[Action]:
        """Cycle the adjacent digits 7, 1 and 5 vertically.

        Initial (top=0, col=0)::

            7 2 9 4
            1 8 3 6
            5 0 7 2
            9 4 1 8

        Target::

            5 2 9 4
            7 8 3 6
            1 0 7 2
            9 4 1 8
        """
        if self.n < 3:
            raise ValueError("vertical_three_cycle needs n >= 3")
        top = self.rng.randrange(self.n - 2)
        col = self.rng.randrange(self.n)
        return [("COL_UP", col),
                ("ROW_DOWN", top),
                ("COL_DOWN", col),
                ("ROW_DOWN", top)]

    def _l_three_cycle(self) -> list[Action]:
        """Cycle the L-shaped digits 7, 2 and 1.

        Initial (r=0, c=0)::

            7 2 9 4
            1 8 3 6
            5 0 7 2
            9 4 1 8

        Target::

            1 7 9 4
            2 8 3 6
            5 0 7 2
            9 4 1 8
        """
        r, c = self._origin(2, 2)
        return [("COL_UP", c),
                ("ROW_LEFT", r),
                ("COL_DOWN", c),
                ("ROW_RIGHT", r)]

    def _double_horizontal_swap(self) -> list[Action]:
        """Swap 7 with 2 and 8 with 3 in one four-action program.

        Initial (r=0, c=0)::

            7 2 9 4
            1 8 3 6
            5 0 7 2
            9 4 1 8

        Target::

            2 7 9 4
            1 3 8 6
            5 0 7 2
            9 4 1 8
        """
        r, c = self._origin(2, 3)
        return [("BLOCK_CW", r, c),
                ("BLOCK_CW", r, c + 1),
                ("BLOCK_CCW", r, c),
                ("BLOCK_CCW", r, c + 1)]

    def _five_cycle_2x3(self) -> list[Action]:
        """Cycle five digits 7, 2, 9, 8 and 3 in a 2x3 patch.

        Initial (r=0, c=0)::

            7 2 9 4
            1 8 3 6
            5 0 7 2
            9 4 1 8

        Target::

            9 7 3 4
            1 2 8 6
            5 0 7 2
            9 4 1 8
        """
        r, c = self._origin(2, 3)
        return [("BLOCK_CW", r, c),
                ("COL_RIGHT", c + 1),
                ("BLOCK_CCW", r, c),
                ("COL_RIGHT", c + 1)]

    def _vertical_swap_blocks(self) -> list[Action]:
        """Swap vertical neighbors 7 and 1 with five block rotations.

        Initial (r=0, c=0)::

            7 2 9 4
            1 8 3 6
            5 0 7 2
            9 4 1 8

        Target::

            1 2 9 4
            7 8 3 6
            5 0 7 2
            9 4 1 8
        """
        r, c = self._origin(3, 3)
        return [("BLOCK_CW", r, c),
                ("BLOCK_CW", r + 1, c),
                ("BLOCK_CCW", r, c + 1),
                ("BLOCK_CCW", r + 1, c),
                ("BLOCK_CW", r, c + 1)]

    def resolve_rule(self, token: str | int) -> InductionRule:
        """Resolve a rule name, 1-based number, or ruleN alias."""
        key = str(token).lower()
        number = key[4:] if key.startswith("rule") else key
        if number.isdigit():
            index = int(number)
            if 1 <= index <= len(self.rules):
                return self.rules[index - 1]
        try:
            return self.by_name[key]
        except KeyError:
            names = ", ".join(self.by_name)
            raise ValueError(
                f"unknown induction rule {token!r}; choose none, "
                f"1-{len(self.rules)}, or one of: {names}") from None

    def resolve_source(
            self, source: str | int | InductionRule | None
            ) -> InductionRule | None:
        """Resolve a source; only the public token none means no rule."""
        if source is None or isinstance(source, InductionRule):
            return source
        if source == self.NORMAL_SOURCE:
            return None
        return self.resolve_rule(source)

    @staticmethod
    def source_name(rule: InductionRule | None) -> str:
        return InductionRuleCatalog.NORMAL_SOURCE if rule is None else rule.name

    def parse_source_counts(
            self, specs: list[str], default_count: int
            ) -> list[tuple[InductionRule | None, int]]:
        """Parse SOURCE[:COUNT] selectors in the user's requested order."""
        if default_count < 0:
            raise ValueError(f"--count must be >= 0, got {default_count}")

        selected: list[tuple[InductionRule | None, int]] = []
        seen: set[str] = set()
        for spec in specs:
            name, sep, count_text = spec.partition(":")
            if sep:
                if not count_text.isdigit():
                    raise ValueError(f"invalid count in --rules {spec!r}")
                count = int(count_text)
            else:
                count = default_count

            if name.lower() == "all":
                sources: tuple[InductionRule | None, ...] = self.rules
            else:
                sources = (self.resolve_source(name),)

            for rule in sources:
                label = self.source_name(rule)
                if label in seen:
                    raise ValueError(f"source {label!r} was selected twice")
                seen.add(label)
                selected.append((rule, count))

        if not selected:
            raise ValueError("select at least one source")
        if not sum(count for _, count in selected):
            raise ValueError("the selected source counts sum to zero")
        return selected

    def print_rules(self) -> None:
        print(" -  none                        steps=1..9  "
              "moves=variable  ordinary STEP_WEIGHTS generation")
        for index, rule in enumerate(self.rules, 1):
            print(f"{index:>2}  {rule.name:<27} steps={rule.steps}     "
                  f"moves={rule.moved_positions:<3}  {rule.description}")

