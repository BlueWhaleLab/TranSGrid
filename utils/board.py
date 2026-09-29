"""Board with 10 atomic actions, plus a renderer that animates them.

Action sequence format: a list of tuples, e.g.
    [("ROW_LEFT", 0), ("COL_DOWN", 2), ("BLOCK_CW", 1, 1)]

Actions:
    ROW_LEFT(r)   : cyclic-shift row r left by one
    ROW_RIGHT(r)  : cyclic-shift row r right by one
    ROW_UP(r)     : swap row r with the row above (wraps around)
    ROW_DOWN(r)   : swap row r with the row below (wraps around)
    COL_UP(c)     : cyclic-shift column c up by one
    COL_DOWN(c)   : cyclic-shift column c down by one
    COL_LEFT(c)   : swap column c with the column to its left (wraps around)
    COL_RIGHT(c)  : swap column c with the column to its right (wraps around)
    BLOCK_CW(i,j) : rotate the 2x2 block with top-left (i,j) clockwise
    BLOCK_CCW(i,j): rotate the same 2x2 block counter-clockwise

`Board` holds the state and the rules; `BoardRenderer` turns a sequence into
an animated GIF:
    BoardRenderer(board).render(seq, "demo.gif")
"""
import random

ACTIONS = [
    "ROW_LEFT", "ROW_RIGHT", "ROW_UP", "ROW_DOWN",
    "COL_UP", "COL_DOWN", "COL_LEFT", "COL_RIGHT",
    "BLOCK_CW", "BLOCK_CCW",
]

class Board:
    def __init__(self, grid):
        self.n = len(grid)
        if any(len(row) != self.n for row in grid):
            raise ValueError("grid must be NxN")
        if any(not (0 <= v <= 9) for row in grid for v in row):
            raise ValueError("cells must be digits 0-9")
        self.grid = [list(row) for row in grid]

    def execute(self, actions, verbose=False):
        """Apply an action sequence to a copy of the board, return the final grid.

        Stops at the first invalid action and raises ValueError naming the step.
        With verbose=True, print every step as `before | action | after`.
        """
        g = [list(row) for row in self.grid]
        total = len(actions)
        for k, action in enumerate(actions, 1):
            before = [row[:] for row in g]
            try:
                name, args = self.apply(g, action)
            except ValueError as e:
                raise ValueError(f"step {k}: {e}") from None
            if verbose:
                self._print_step(k, total, name, args, before, g)
        return g

    def apply(self, g, action):
        """Validate one action and apply it to grid `g` in place.
        Returns the parsed (name, args).
        """
        name, args = self._parse_action(action)
        getattr(self, "_" + name.lower())(g, *args)
        return name, args

    def validate(self, actions):
        """Check a whole sequence up front; return it as [(name, args), ...].
        Raises ValueError naming the first bad step.
        """
        parsed = []
        for k, action in enumerate(actions, 1):
            try:
                parsed.append(self._parse_action(action))
            except ValueError as e:
                raise ValueError(f"step {k}: {e}") from None
        return parsed

    # ---- printing ----
    def print_grid(self, grid=None, title=""):
        """Print a grid as an NxN block; defaults to the board's own grid."""
        if title:
            print(title)
        for row in (self.grid if grid is None else grid):
            print(" ".join(str(v) for v in row))
        print()

    def _print_step(self, k, total, name, args, before, after):
        """Print one step as `before | action | after`, side by side."""
        arrow = f"-- {name}({','.join(map(str, args))}) -->"
        gap = " " * 4
        mid = self.n // 2
        left = [" ".join(str(v) for v in row) for row in before]
        right = [" ".join(str(v) for v in row) for row in after]
        print(f"step {k}/{total}:")
        for r in range(self.n):
            centre = arrow if r == mid else " " * len(arrow)
            print(f"{gap}{left[r]}{gap}{centre}{gap}{right[r]}")
        print()

    # ---- validation ----
    def _parse_action(self, action):
        """Validate one action, return (name, args). Raises ValueError."""
        if isinstance(action, str):
            raise ValueError(f"action must be a tuple like ('ROW_LEFT', 0), "
                             f"got the bare string {action!r}")
        try:
            name, *args = action
        except (TypeError, ValueError):
            raise ValueError(f"action must be a tuple like ('ROW_LEFT', 0), "
                             f"got {action!r}") from None
        if name not in ACTIONS:
            raise ValueError(f"unknown action {name!r}; "
                             f"expected one of {', '.join(ACTIONS)}")
        want = 2 if name.startswith("BLOCK") else 1
        if len(args) != want:
            raise ValueError(f"{name} takes {want} argument(s), "
                             f"got {len(args)}: {list(args)}")
        if want == 2:
            self._check_block(name, *args)
        else:
            self._check_line(name, "col" if name.startswith("COL") else "row",
                             args[0])
        return name, args

    def _check_line(self, name, kind, i):
        """Validate a row/column index for a full-line action."""
        if not isinstance(i, int) or isinstance(i, bool):
            raise ValueError(f"{name}: {kind} index must be an int, got {i!r}")
        if not 0 <= i < self.n:
            raise ValueError(f"{name}: {kind} {i} out of range, "
                             f"board is {self.n}x{self.n} so valid indices are "
                             f"0..{self.n - 1}")

    def _check_block(self, name, i, j):
        """Validate the top-left corner of a 2x2 block."""
        for kind, v in (("row", i), ("col", j)):
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"{name}: block {kind} index must be an int, "
                                 f"got {v!r}")
        if self.n < 2:
            raise ValueError(f"{name}: board is {self.n}x{self.n}, "
                             f"too small to hold a 2x2 block")
        if not (0 <= i <= self.n - 2 and 0 <= j <= self.n - 2):
            raise ValueError(f"{name}: 2x2 block at ({i},{j}) out of range, "
                             f"board is {self.n}x{self.n} so the top-left "
                             f"corner must be in 0..{self.n - 2}")

    # ---- row actions ----
    def _row_left(self, g, r):
        self._check_line("ROW_LEFT", "row", r)
        g[r] = g[r][1:] + g[r][:1]

    def _row_right(self, g, r):
        self._check_line("ROW_RIGHT", "row", r)
        g[r] = g[r][-1:] + g[r][:-1]

    def _row_up(self, g, r):
        self._check_line("ROW_UP", "row", r)
        g[r], g[r - 1] = g[r - 1], g[r]

    def _row_down(self, g, r):
        self._check_line("ROW_DOWN", "row", r)
        r2 = (r + 1) % self.n
        g[r], g[r2] = g[r2], g[r]

    # ---- column actions ----
    def _col_up(self, g, c):
        self._check_line("COL_UP", "col", c)
        col = [g[r][c] for r in range(self.n)]
        col = col[1:] + col[:1]
        for r in range(self.n):
            g[r][c] = col[r]

    def _col_down(self, g, c):
        self._check_line("COL_DOWN", "col", c)
        col = [g[r][c] for r in range(self.n)]
        col = col[-1:] + col[:-1]
        for r in range(self.n):
            g[r][c] = col[r]

    def _col_left(self, g, c):
        self._check_line("COL_LEFT", "col", c)
        for r in range(self.n):
            g[r][c], g[r][c - 1] = g[r][c - 1], g[r][c]

    def _col_right(self, g, c):
        self._check_line("COL_RIGHT", "col", c)
        c2 = (c + 1) % self.n
        for r in range(self.n):
            g[r][c], g[r][c2] = g[r][c2], g[r][c]

    # ---- 2x2 block actions ----
    def _block_cw(self, g, i, j):
        self._check_block("BLOCK_CW", i, j)
        g[i][j], g[i][j + 1], g[i + 1][j + 1], g[i + 1][j] = \
            g[i + 1][j], g[i][j], g[i][j + 1], g[i + 1][j + 1]

    def _block_ccw(self, g, i, j):
        self._check_block("BLOCK_CCW", i, j)
        g[i][j], g[i][j + 1], g[i + 1][j + 1], g[i + 1][j] = \
            g[i][j + 1], g[i + 1][j + 1], g[i + 1][j], g[i][j]


class BoardRenderer:
    """Render a Board's action sequence as an animated GIF (requires Pillow).

    For each action: the board *before* it is held for `duration` ms with the
    affected cells highlighted and red arrows showing where each cell will
    move, then `tween_frames` short in-between frames animate the digits
    sliding to their new positions, and the result is held again. The title
    bar shows "step k/K: ACTION(args)" plus a description. The closing frame
    is held for `final_duration` ms.

        BoardRenderer(board).render(seq, "demo.gif")
    """

    TOP = 64          # title-bar height in px
    CAPTION = 22      # per-board caption strip, between title bar and boards
    MARGIN = 16       # min margin around the boards in px
    GAP = 40          # px between the initial board and the current one
    ARROW_CHARS = {"left": "←", "right": "→",
                   "up": "↑", "down": "↓"}
    ARROW_COLOR = "#e03131"

    def __init__(self, board, cell=60, duration=500,
                 tween_frames=8, tween_ms=70, final_duration=3000):
        self.board = board
        self.n = board.n
        self.cell = cell
        self.duration = duration
        self.tween_frames = tween_frames
        self.tween_ms = tween_ms
        self.final_duration = final_duration
        self._fonts = None
        self.width = 0        # set per render, once the titles are known
        self.initial = None   # left-hand reference board, fixed per render

    def render(self, actions, path="board.gif"):
        """Write the animated GIF to `path` and return the path."""
        parsed = self.board.validate(actions)
        total = len(parsed)
        steps = [(f"step {k}/{total}: {name}({','.join(map(str, args))})",
                  self._describe(name, args), self._arrows(name, args))
                 for k, (name, args) in enumerate(parsed, 1)]

        # canvas must fit both boards and the longest title/description
        _, title, small, _ = self.fonts
        text_w = max([title.getlength(s[0]) for s in steps] +
                     [small.getlength(s[1]) for s in steps], default=0)
        boards_w = 2 * self.n * self.cell + self.GAP
        self.width = max(boards_w + 2 * self.MARGIN, int(text_w) + 16)

        self.initial = [list(row) for row in self.board.grid]
        g = [list(row) for row in self.board.grid]
        frames = [self._draw_frame(g, "start", "", {})]
        durations = [self.duration]
        for (label, desc, arrows), (name, args) in zip(steps, parsed):
            frames.append(self._draw_frame(g, label, desc, arrows,
                                           show_arrows=True))
            durations.append(self.duration)
            moves = self._moves(name, args)
            g_before = [row[:] for row in g]
            self.board.apply(g, (name, *args))
            for k in range(1, self.tween_frames + 1):
                t = k / (self.tween_frames + 1)
                frames.append(self._draw_tween(g_before, moves, t, label, desc))
                durations.append(self.tween_ms)
            frames.append(self._draw_frame(g, label, desc, arrows))
            durations.append(self.duration)
        frames.append(self._draw_frame(g, "final", "", {}))
        durations.append(self.final_duration)
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0)
        return path

    # ---- what an action looks like on screen ----
    def _moves(self, name, args):
        """Each moving cell as ((r0,c0), (r1,c1)) source -> destination."""
        n = self.n
        if name in ("ROW_LEFT", "ROW_RIGHT"):
            r, s = args[0], -1 if name == "ROW_LEFT" else 1
            return [((r, c), (r, (c + s) % n)) for c in range(n)]
        if name in ("ROW_UP", "ROW_DOWN"):
            r = args[0]
            r2 = (r - 1) % n if name == "ROW_UP" else (r + 1) % n
            return ([((r, c), (r2, c)) for c in range(n)] +
                    [((r2, c), (r, c)) for c in range(n)])
        if name in ("COL_UP", "COL_DOWN"):
            c, s = args[0], -1 if name == "COL_UP" else 1
            return [((r, c), ((r + s) % n, c)) for r in range(n)]
        if name in ("COL_LEFT", "COL_RIGHT"):
            c = args[0]
            c2 = (c - 1) % n if name == "COL_LEFT" else (c + 1) % n
            return ([((r, c), (r, c2)) for r in range(n)] +
                    [((r, c2), (r, c)) for r in range(n)])
        i, j = args
        cyc = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
        if name == "BLOCK_CCW":
            cyc.reverse()
        return [(cyc[k], cyc[(k + 1) % 4]) for k in range(4)]

    def _describe(self, name, args):
        """One-line description of an action's effect."""
        n, a = self.n, args[0]
        if name == "ROW_LEFT":
            return f"row {a} shifts left by 1 (cyclic)"
        if name == "ROW_RIGHT":
            return f"row {a} shifts right by 1 (cyclic)"
        if name == "ROW_UP":
            return f"row {a} swaps with row {(a - 1) % n}"
        if name == "ROW_DOWN":
            return f"row {a} swaps with row {(a + 1) % n}"
        if name == "COL_UP":
            return f"col {a} shifts up by 1 (cyclic)"
        if name == "COL_DOWN":
            return f"col {a} shifts down by 1 (cyclic)"
        if name == "COL_LEFT":
            return f"col {a} swaps with col {(a - 1) % n}"
        if name == "COL_RIGHT":
            return f"col {a} swaps with col {(a + 1) % n}"
        rot = "clockwise" if name == "BLOCK_CW" else "counter-clockwise"
        return f"2x2 block at ({args[0]},{args[1]}) rotates {rot}"

    def _arrows(self, name, args):
        """Cells affected by an action, as {(r, c): movement direction}."""
        n = self.n
        if name.startswith("ROW"):
            r = args[0]
            if name == "ROW_LEFT":
                return {(r, c): "left" for c in range(n)}
            if name == "ROW_RIGHT":
                return {(r, c): "right" for c in range(n)}
            r2, d1, d2 = ((r - 1) % n, "up", "down") if name == "ROW_UP" \
                else ((r + 1) % n, "down", "up")
            return {**{(r, c): d1 for c in range(n)},
                    **{(r2, c): d2 for c in range(n)}}
        if name.startswith("COL"):
            c = args[0]
            if name == "COL_UP":
                return {(r, c): "up" for r in range(n)}
            if name == "COL_DOWN":
                return {(r, c): "down" for r in range(n)}
            c2, d1, d2 = ((c - 1) % n, "left", "right") if name == "COL_LEFT" \
                else ((c + 1) % n, "right", "left")
            return {**{(r, c): d1 for r in range(n)},
                    **{(r, c2): d2 for r in range(n)}}
        i, j = args
        if name == "BLOCK_CW":
            return {(i, j): "right", (i, j + 1): "down",
                    (i + 1, j + 1): "left", (i + 1, j): "up"}
        return {(i, j): "down", (i + 1, j): "right",
                (i + 1, j + 1): "up", (i, j + 1): "left"}

    # ---- drawing ----
    @property
    def fonts(self):
        """(digit, title, small, arrow) fonts, loaded once per renderer."""
        if self._fonts is None:
            from PIL import ImageFont
            try:
                self._fonts = (
                    ImageFont.truetype("DejaVuSans-Bold.ttf", self.cell // 2),
                    ImageFont.truetype("DejaVuSans-Bold.ttf", 17),
                    ImageFont.truetype("DejaVuSans.ttf", 14),
                    ImageFont.truetype("DejaVuSans-Bold.ttf", self.cell // 3),
                )
            except OSError:
                f = ImageFont.load_default()
                self._fonts = (f, f, f, f)
        return self._fonts

    @property
    def board_top(self):
        """Top edge of both boards, below the title bar and captions."""
        return self.TOP + self.CAPTION

    @property
    def xoff(self):
        """Left edge of the initial (reference) board."""
        return (self.width - (2 * self.n * self.cell + self.GAP)) // 2

    @property
    def xoff2(self):
        """Left edge of the current (animated) board."""
        return self.xoff + self.n * self.cell + self.GAP

    def _frame_base(self, label, desc):
        """Title bar, captions and the initial board; returns (img, draw)."""
        from PIL import Image, ImageDraw
        _, title, small, _ = self.fonts
        img = Image.new("RGB", (self.width,
                                self.board_top + self.n * self.cell + self.MARGIN),
                        "white")
        d = ImageDraw.Draw(img)
        d.text((self.width / 2, 20), label, fill="black", font=title, anchor="mm")
        d.text((self.width / 2, 46), desc, fill="#555555", font=small, anchor="mm")
        half, cy = self.n * self.cell / 2, self.TOP + self.CAPTION / 2
        d.text((self.xoff + half, cy), "initial", fill="#888888",
               font=small, anchor="mm")
        d.text((self.xoff2 + half, cy), "current", fill="#888888",
               font=small, anchor="mm")
        self._draw_cells(d, self.initial, set(), self.xoff)
        return img, d

    def _draw_cells(self, d, g, touched, xoff, skip_digits=False):
        font, cell = self.fonts[0], self.cell
        for r in range(self.n):
            for c in range(self.n):
                x0, y0 = xoff + c * cell, self.board_top + r * cell
                bg = "#ffd966" if (r, c) in touched else "#f0f0f0"
                d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=bg, outline="#888888")
                if not (skip_digits and (r, c) in touched):
                    d.text((x0 + cell / 2, y0 + cell / 2), str(g[r][c]),
                           fill="black", font=font, anchor="mm")

    def _draw_frame(self, g, label, desc, arrows, show_arrows=False):
        img, d = self._frame_base(label, desc)
        cell, xoff = self.cell, self.xoff2
        self._draw_cells(d, g, set(arrows), xoff)
        if show_arrows:
            afont = self.fonts[3]
            # arrow sits at the cell edge it points toward
            pos = {"left": (0.18, 0.5), "right": (0.82, 0.5),
                   "up": (0.5, 0.18), "down": (0.5, 0.82)}
            for (r, c), direction in arrows.items():
                px, py = pos[direction]
                d.text((xoff + (c + px) * cell, self.board_top + (r + py) * cell),
                       self.ARROW_CHARS[direction], fill=self.ARROW_COLOR,
                       font=afont, anchor="mm")
        return img

    def _draw_tween(self, g, moves, t, label, desc):
        """In-between frame: moving digits drawn at interpolated positions."""
        img, d = self._frame_base(label, desc)
        font, cell, xoff = self.fonts[0], self.cell, self.xoff2
        moving = {src for src, _ in moves}
        self._draw_cells(d, g, moving, xoff, skip_digits=True)
        board_box = (xoff, self.board_top,
                     xoff + self.n * cell, self.board_top + self.n * cell)
        for (r0, c0), (r1, c1) in moves:
            dr, dc = r1 - r0, c1 - c0
            wrap = abs(dr) > 1 or abs(dc) > 1
            if wrap:  # cyclic move: slide off one edge, in from the other
                dr = -1 if dr > 1 else (1 if dr < -1 else dr)
                dc = -1 if dc > 1 else (1 if dc < -1 else dc)
            positions = [(r0 + t * dr, c0 + t * dc)]
            if wrap:
                positions.append((r1 - (1 - t) * dr, c1 - (1 - t) * dc))
            for rr, cc in positions:
                cx = xoff + (cc + 0.5) * cell
                cy = self.board_top + (rr + 0.5) * cell
                if board_box[0] <= cx <= board_box[2] and \
                        board_box[1] <= cy <= board_box[3]:
                    d.text((cx, cy), str(g[r0][c0]),
                           fill="black", font=font, anchor="mm")
        return img


if __name__ == "__main__":
    board = Board([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 0, 1, 2],
        [3, 4, 5, 6],
    ])
    board.print_grid(title="initial board:")

    # 例 1: 单个 action
    board.print_grid(board.execute([("ROW_LEFT", 0)]), "ROW_LEFT(0):")
    board.print_grid(board.execute([("COL_DOWN", 1)]), "COL_DOWN(1):")
    board.print_grid(board.execute([("BLOCK_CW", 1, 1)]), "BLOCK_CW(1,1):")

    # 例 2: action 序列, verbose 逐步打印
    seq = [("ROW_LEFT", 0), ("COL_DOWN", 1), ("ROW_UP", 2), ("BLOCK_CCW", 0, 2)]
    board.print_grid(board.execute(seq, verbose=True), f"sequence {seq}:")

    # 例 3: 随机 scramble
    random.seed(42)
    scramble = []
    for _ in range(8):
        name = random.choice(ACTIONS)
        if name.startswith("BLOCK"):
            args = (random.randrange(board.n - 1), random.randrange(board.n - 1))
        else:
            args = (random.randrange(board.n),)
        scramble.append((name, *args))
    board.print_grid(board.execute(scramble), f"random scramble {scramble}:")

    # 例 4: 可视化 action 序列, 生成 gif
    renderer = BoardRenderer(board)
    print("gif saved to:", renderer.render(seq, "board_demo.gif"))

    # 例 5: 仅用 BLOCK 旋转精确交换两个相邻元素 (0,0) <-> (0,1), 其余不动
    # (单个 2x2 块自身无法做到; 这是 BFS 搜出的最短组合, 5 步)
    swap_seq = [("BLOCK_CW", 1, 0), ("BLOCK_CCW", 0, 1), ("BLOCK_CCW", 1, 0),
                ("BLOCK_CW", 0, 1), ("BLOCK_CW", 0, 0)]
    board.print_grid(board.execute(swap_seq), "exact swap of (0,0) and (0,1):")
    print("gif saved to:", BoardRenderer(board).render(swap_seq, "swap_demo.gif"))
