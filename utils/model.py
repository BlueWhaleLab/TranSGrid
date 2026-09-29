"""Encoder-decoder Transformer for the board planning task.

The encoder reads the initial and target boards as digit tokens; the decoder
emits one token per action and stops at <EOS>.

    encoder:  <INITIAL> 4 3 4 4 <NL> ... <TARGET> 2 3 4 4 <NL> ...
    decoder:  <BOS> ROW_LEFT 0 BLOCK_CW 1 1 ... <EOS>

An action is a name token followed by its arguments, so the vocabulary holds
10 action names plus arguments 0-9 rather than one token per (action, args)
combination.

The two streams have separate vocabularies and separate embedding tables: a
4 in a board cell and a 4 as a row index are unrelated symbols, and nothing
is gained by making one vector serve both. The encoder holds 14 symbols, the
decoder 23, for every board size up to 10x10.

The decoder uses learned absolute positions; the encoder takes either those
(`src_pos="flat"`) or board/row/col embeddings that mirror the 2D layout
(`src_pos="grid"`, the default). Layers are pre-norm with GELU and a
feed-forward width of 4*d_model.

    tok = BoardTokenizer(n=6)
    model = TranSGridModel.from_level("level1", tok)
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .board import ACTIONS
except ImportError:      # running as a script: python utils/model.py
    from board import ACTIONS

# ---- two id spaces, one per stream, one embedding table each ----
#   encoder:  0 <PAD>  1 <NL>  2 <INITIAL>  3 <TARGET>  4..13 cell digits 0-9
#   decoder:  0 <PAD>  1 <BOS>  2 <EOS>  3..12 action names  13..22 args 0-9
#
# The tables are disjoint, so the two spaces reuse the low ids for unrelated
# symbols: id 1 is <NL> to the encoder and <BOS> to the decoder, id 5 is the
# cell digit 1 and the action name COL_DOWN. Only <PAD> is deliberately 0 on
# both sides, because the dataset padding, the two key padding masks and the
# loss all key off that single constant.
#
# Arguments run 0-9, which caps the board at 10x10. The encoder has no
# separate cap: its length is tokenizer.src_len, and a 10x10 board pair is
# 222 tokens either way.


PAD = 0
NL, INITIAL, TARGET = 1, 2, 3
DIGIT_BASE = 4
SRC_VOCAB_SIZE = DIGIT_BASE + 10

BOS, EOS = 1, 2
ACTION_BASE = 3
ARG_BASE = ACTION_BASE + len(ACTIONS)
MAX_ARG = 9
TGT_VOCAB_SIZE = ARG_BASE + MAX_ARG + 1

# Greedy decoding stops here, at every rung of the ladder. The cap has to be a
# constant rather than something read off the labels: sizing it from the golden
# answer lets the evaluation see the thing it is scoring, and a budget that
# moves with the batch makes two runs incomparable. 128 is far above what the
# task needs -- 9 actions of arity 2 plus <EOS> is 28 tokens -- so nothing is
# truncated and a model that rambles is scored on its rambling rather than
# rescued by the cut. generate() still stops as soon as every row has emitted
# <EOS>, so the cap costs nothing once a model has learned to stop.
MAX_DECODE_TOKENS = 128

# Six sizes, numbered by parameter count. Depth and width are both
# non-decreasing along the ladder -- each rung is deeper, or wider, or both,
# and never trades one for the other:
#
#   level1   2/2   d=128  2 heads     0.96M grid /  0.97M flat
#   level2   4/4   d=128  2 heads     1.89M grid /  1.89M flat
#   level3   4/4   d=256  4 heads     7.45M grid /  7.45M flat
#   level4   6/6   d=256  4 heads    11.13M grid / 11.14M flat
#   level5   6/6   d=512  8 heads    44.29M grid / 44.30M flat
#   level6   8/8   d=512  8 heads    59.00M grid / 59.01M flat
#   level7  12/12  d=512  8 heads    88.43M grid / 88.44M flat
#
# Every d_model is a power-of-two multiple of 128, every head count is a power
# of two, every head is 64 wide, and d_ff = 4*d_model throughout.

MODEL_LEVELS = {
    "level1": {"layers": 2, "d_model": 128, "heads": 2, "d_ff": 512},
    "level2": {"layers": 4, "d_model": 128, "heads": 2, "d_ff": 512},
    "level3": {"layers": 4, "d_model": 256, "heads": 4, "d_ff": 1024},
    "level4": {"layers": 6, "d_model": 256, "heads": 4, "d_ff": 1024},
    "level5": {"layers": 6, "d_model": 512, "heads": 8, "d_ff": 2048},
    "level6": {"layers": 8, "d_model": 512, "heads": 8, "d_ff": 2048},
    "level7": {"layers": 12, "d_model": 512, "heads": 8, "d_ff": 2048},
}


class BoardTokenizer:
    """Turns board pairs into encoder ids and action lists into decoder ids."""

    def __init__(self, n):
        if n - 1 > MAX_ARG:
            raise ValueError(f"n={n} needs argument {n - 1}, the decoder "
                             f"vocabulary only holds arguments up to {MAX_ARG}")
        self.n = n
        self.src_vocab_size = SRC_VOCAB_SIZE
        self.tgt_vocab_size = TGT_VOCAB_SIZE
        self.action_to_id = {name: ACTION_BASE + i
                             for i, name in enumerate(ACTIONS)}
        self.id_to_action = {ACTION_BASE + i: name
                             for i, name in enumerate(ACTIONS)}
        # <INITIAL> + n rows of (n cells + <NL>), twice
        self.src_len = 2 * (n * n + n) + 2

    def encode_source(self, initial, target):
        """Both boards as one flat list of ids (length self.src_len).
        """
        ids = []
        for marker, grid in ((INITIAL, initial), (TARGET, target)):
            ids.append(marker)
            for row in grid:
                for v in row:
                    if not 0 <= v <= 9:
                        raise ValueError(f"cell value {v} is not a digit 0-9")
                    ids.append(DIGIT_BASE + v)
                ids.append(NL)
        return ids

    def source_layout(self):
        """Per-position (board, row, col) indices for grid position embeddings.
        """
        n, pos = self.n, []
        for b in (0, 1):
            pos.append((b, n, n))                    # <INITIAL> / <TARGET>
            for r in range(n):
                pos += [(b, r, c) for c in range(n)]
                pos.append((b, r, n))                # <NL>
        return tuple(zip(*pos))                      # (boards, rows, cols)

    def encode_target(self, answer):
        """<BOS> + (action name, args...) per action + <EOS>."""
        ids = [BOS]
        for name, *args in answer:
            ids.append(self.action_to_id[name])
            ids += [ARG_BASE + int(a) for a in args]
        ids.append(EOS)
        return ids

    def decode_actions(self, ids):
        """Ids back to actions; stops at <EOS>, <PAD> or malformed output.

        A truncated or ill-formed tail is dropped rather than repaired, so a
        model that emits the wrong number of arguments simply scores wrong.
        """
        out, i = [], 0
        while i < len(ids):
            name = self.id_to_action.get(int(ids[i]))
            if name is None:
                break                       # <EOS>, <PAD> or not an action
            arity = 2 if name.startswith("BLOCK") else 1
            if i + arity >= len(ids):
                break
            args = []
            for j in range(1, arity + 1):
                v = int(ids[i + j]) - ARG_BASE
                if not 0 <= v <= MAX_ARG:
                    return out              # expected an argument token
                args.append(v)
            out.append((name, *args))
            i += 1 + arity
        return out


PAIR_VOCAB = 1 + 100          # <PAD> plus every (initial, target) digit pair


class PairTokenizer(BoardTokenizer):
    """One token per cell, carrying both boards' digits.

    BoardTokenizer lays the two boards out end to end, so the encoder reads
    2*(n*n+n)+2 tokens and has to spend an attention hop lining cell (r,c) of
    the initial board up with cell (r,c) of the target board before it can
    ask the only question that matters: did this cell change, and to what.
    Here each cell is a single token holding the pair:

        id = 1 + 10 * initial[r][c] + target[r][c]        (0 stays <PAD>)

    A 6x6 pair is then 36 tokens over 101 symbols instead of 86 over 14. The
    sequence is 2.4x shorter, which is 5.7x less encoder attention, and "this
    cell went from 4 to 7" is one embedding lookup rather than something to
    infer. Measured at level3 on the 8-action split, swapping the encoding
    alone moved solve 0.52 -> 0.67 and skipped the encoder-blind plateau the
    board encoding spends 2-3 epochs stuck in.

    The decoder side is untouched -- same action vocabulary, same
    encode_target -- so a checkpoint's decoder stays comparable with the
    board encoding's.
    """

    def __init__(self, n):
        super().__init__(n)
        self.src_vocab_size = PAIR_VOCAB
        self.src_len = n * n

    def encode_source(self, initial, target):
        ids = []
        for r in range(self.n):
            for c in range(self.n):
                a, b = initial[r][c], target[r][c]
                if not (0 <= a <= 9 and 0 <= b <= 9):
                    raise ValueError(f"cell ({r},{c}) holds {a}/{b}, "
                                     f"not a digit pair")
                ids.append(1 + 10 * a + b)
        return ids

    def source_layout(self):
        """(board, row, col) per position; the board axis is now constant.

        Keeping the triple means TranSGridModel needs no change -- the board
        embedding collapses to a single learned vector added everywhere,
        which the layer norm absorbs.
        """
        rows, cols = [], []
        for r in range(self.n):
            for c in range(self.n):
                rows.append(r)
                cols.append(c)
        return (tuple([0] * len(rows)), tuple(rows), tuple(cols))


TOKENIZERS = {"board": BoardTokenizer, "pair": PairTokenizer}


def parse_source(spec, root=None):
    """'path' or 'path:count' -> (Path, count or None).

    The count takes the first N lines. Splits written by split_data.py are
    already shuffled, so a prefix is a uniform random subset.
    """
    head, sep, tail = str(spec).rpartition(":")
    if sep and tail.isdigit():
        path, limit = head, int(tail)
    else:
        path, limit = str(spec), None
    return (Path(path) if root is None else Path(root) / path), limit


class BoardDataset(Dataset):
    """One or more JSONL splits, tokenized once into compact tensors.

    `sources` is a path, or a list of (path, count) pairs to mix; `count` of
    None takes the whole file. The mixture is recorded in .composition.

    Both vocabularies stay far under 256 symbols for every board size, so
    uint8 holds either side.
    """

    def __init__(self, sources, tokenizer):
        if isinstance(sources, (str, Path)):
            sources = [(sources, None)]
        src, tgt = [], []
        self.composition = []
        for path, limit in sources:
            taken = 0
            with open(path) as f:
                for line in f:
                    if limit is not None and taken >= limit:
                        break
                    item = json.loads(line)
                    src.append(tokenizer.encode_source(item["initial"],
                                                       item["target"]))
                    tgt.append(tokenizer.encode_target(item["answer"]))
                    taken += 1
            if limit is not None and taken < limit:
                raise ValueError(f"{path} holds {taken} items, "
                                 f"asked for {limit}")
            self.composition.append({"path": str(path), "items": taken})
        if not src:
            raise ValueError(f"no items read from {sources}")

        self.tokenizer = tokenizer
        self.src = torch.tensor(src, dtype=torch.uint8)
        self.tgt_len = torch.tensor([len(t) for t in tgt], dtype=torch.int16)
        width = int(self.tgt_len.max())
        self.tgt = torch.full((len(tgt), width), PAD, dtype=torch.uint8)
        for i, t in enumerate(tgt):
            self.tgt[i, :len(t)] = torch.tensor(t, dtype=torch.uint8)

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, i):
        return self.src[i], self.tgt[i], self.tgt_len[i]


def collate(batch):
    """Stack a batch: both sides at their real width, nothing padded to a constant.

    The encoder runs at tokenizer.src_len -- 36 for a 6x6 pair encoding, 86
    for the board encoding.

    The decoder is trimmed to the longest target in the batch, because nothing
    depends on its width being constant.
    """
    src = torch.stack([b[0] for b in batch]).long()
    tgt = torch.stack([b[1] for b in batch]).long()
    return src, tgt[:, :int(max(b[2] for b in batch))]


class TranSGridModel(nn.Module):
    """Standard encoder-decoder Transformer, pre-norm, learned positions.

    The encoder side takes one of two position schemes, set by `src_layout`:

    flat (src_layout=None)
        one learned vector per absolute index 0..src_len-1, the usual thing.
        Cell (r,c) of the target board then sits a fixed but *arbitrary*
        offset away from cell (r,c) of the initial board -- 43 positions for
        n=6 -- and the model has to discover that offset from scratch.

    grid (src_layout from BoardTokenizer.source_layout)
        position = board + row + col embeddings. Corresponding cells share
        their row and col vector, so "same cell, other board" is a direction
        in embedding space rather than 86 arbitrary vectors to memorise.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, heads, layers,
                 d_ff, src_len, dropout=0.1,
                 max_tgt_positions=MAX_DECODE_TOKENS, src_layout=None):
        super().__init__()
        self.d_model = d_model
        self.src_len = src_len
        self.max_tgt_positions = max_tgt_positions
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD)
        self.tgt_pos = nn.Embedding(max_tgt_positions, d_model)
        self.grid_pos = src_layout is not None
        if self.grid_pos:
            self._build_grid_pos(src_layout, d_model)
        else:
            self.src_pos = nn.Embedding(src_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=heads,
            num_encoder_layers=layers, num_decoder_layers=layers,
            dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.out = nn.Linear(d_model, tgt_vocab_size)
        self._init_embeddings()

    def _build_grid_pos(self, src_layout, d_model):
        """Embeddings and per-position index buffers for the grid scheme.

        One buffer entry per real encoder position. Nothing pads the encoder
        past src_len, so there is no tail to index; the buffers are exactly
        the layout the tokenizer handed over, which is also what binds this
        model to one board size.

        Each table carries one spare row beyond the axis it encodes. Nothing
        indexes it now; it stays in reserve.
        """
        sizes = [max(axis) + 2 for axis in src_layout]   # +1 spare, in reserve
        self.board_emb, self.row_emb, self.col_emb = (
            nn.Embedding(s, d_model) for s in sizes)
        for name, axis in zip(("board_idx", "row_idx", "col_idx"), src_layout):
            self.register_buffer(name, torch.tensor(axis, dtype=torch.long),
                                 persistent=False)

    def _init_embeddings(self):
        """Normal(0, 0.02) for every embedding table, as in BERT/GPT.

        torch's default is Normal(0, 1), which makes the position vector as
        large as the whole rest of the residual stream at step 0.
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    @classmethod
    def from_config(cls, cfg, tokenizer, dropout=0.1, src_pos="grid"):
        """Build the model from a sizes dict, e.g. MODEL_LEVELS["level6"].

        The sizes travel with the checkpoint rather than the level name:
        train.py saves this dict into it, and eval.py rebuilds from that
        saved copy. An old checkpoint therefore still loads at the sizes
        it was trained at, even if MODEL_LEVELS changes underneath it.
        """
        layout = tokenizer.source_layout() if src_pos == "grid" else None
        return cls(tokenizer.src_vocab_size, tokenizer.tgt_vocab_size,
                   cfg["d_model"], cfg["heads"], cfg["layers"], cfg["d_ff"],
                   tokenizer.src_len, dropout, src_layout=layout)

    @classmethod
    def from_level(cls, level, tokenizer, dropout=0.1, src_pos="grid"):
        return cls.from_config(MODEL_LEVELS[level], tokenizer, dropout,
                               src_pos)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def _check_len(self, ids, limit, stream):
        if ids.size(1) > limit:
            raise ValueError(f"{stream} sequence of {ids.size(1)} exceeds "
                             f"{limit} positions")
        return ids.size(1)

    def _embed_src(self, ids):
        length = self._check_len(ids, self.src_len, "encoder")
        if self.grid_pos:
            pos = (self.board_emb(self.board_idx[:length])
                   + self.row_emb(self.row_idx[:length])
                   + self.col_emb(self.col_idx[:length]))
        else:
            pos = self.src_pos(torch.arange(length, device=ids.device))
        return self.dropout(self.src_emb(ids) + pos[None])

    def _embed_tgt(self, ids):
        idx = torch.arange(self._check_len(ids, self.max_tgt_positions,
                                           "decoder"), device=ids.device)
        return self.dropout(self.tgt_emb(ids) + self.tgt_pos(idx)[None])

    def encode(self, src):
        pad = src == PAD
        memory = self.transformer.encoder(
            self._embed_src(src), src_key_padding_mask=pad)
        return memory, pad

    def decode(self, tgt_in, memory, src_pad):
        causal = torch.ones(tgt_in.size(1), tgt_in.size(1), dtype=torch.bool,
                            device=tgt_in.device).triu(1)
        hidden = self.transformer.decoder(
            self._embed_tgt(tgt_in), memory,
            tgt_mask=causal, tgt_is_causal=True,
            tgt_key_padding_mask=tgt_in == PAD,
            memory_key_padding_mask=src_pad)
        return self.out(hidden)

    def forward(self, src, tgt_in):
        memory, src_pad = self.encode(src)
        return self.decode(tgt_in, memory, src_pad)

    @torch.no_grad()
    def generate(self, src, max_new_tokens=MAX_DECODE_TOKENS,
                 return_stop=False):
        """Greedy decoding; returns an id tensor [B, T] without <BOS>.

        A row finishes when it emits <EOS> or <PAD> and is filled with <PAD>
        from there on; rows still producing tokens at `max_new_tokens` are cut
        off there.

        With `return_stop`, the second return value is a bool tensor, True for
        the rows that stopped on their own.
        """
        memory, src_pad = self.encode(src)
        b = src.size(0)
        ys = torch.full((b, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(b, dtype=torch.bool, device=src.device)
        limit = min(max_new_tokens, self.max_tgt_positions)

        for _ in range(limit):
            nxt = self.decode(ys, memory, src_pad)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt[:, None]], dim=1)
            done |= (nxt == EOS) | (nxt == PAD)
            if bool(done.all()):
                break
        return (ys[:, 1:], done) if return_stop else ys[:, 1:]

    @torch.no_grad()
    def beam_generate(self, src, beams, max_new_tokens=MAX_DECODE_TOKENS):
        """Beam search; returns ids [N, beams, T] and a done mask [N, beams].

        Same contract as generate(), one rank wider. At beams=1 this is greedy
        and agrees with generate(); above that, beam 0 is the sequence with the
        highest total log-probability, which is not the step-by-step argmax
        path and so does not reproduce generate() -- measured on level6 it
        scores a little better than greedy, and a wider beam does not
        monotonically improve it. Compare decoders on solve, not on beam 0.
        """
        memory, src_pad = self.encode(src)
        n, s, d = memory.shape
        mem = memory[:, None].expand(n, beams, s, d).reshape(n * beams, s, d)
        pad = src_pad[:, None].expand(n, beams, s).reshape(n * beams, s)

        ys = torch.full((n * beams, 1), BOS, dtype=torch.long,
                        device=src.device)
        scores = torch.full((n, beams), float("-inf"), device=src.device)
        scores[:, 0] = 0.0                      # one live beam to start
        done = torch.zeros(n * beams, dtype=torch.bool, device=src.device)

        limit = min(max_new_tokens, self.max_tgt_positions)
        for _ in range(limit):
            logp = F.log_softmax(self.decode(ys, mem, pad)[:, -1].float(), -1)
            logp[done] = float("-inf")
            logp[done, PAD] = 0.0

            vocab = logp.size(-1)
            cand = (scores.reshape(-1, 1) + logp).view(n, beams * vocab)
            scores, flat = cand.topk(beams, dim=-1)
            rows = (torch.arange(n, device=src.device)[:, None] * beams
                    + flat // vocab).reshape(-1)
            token = (flat % vocab).reshape(-1)

            ys = torch.cat([ys[rows], token[:, None]], dim=1)
            done = done[rows] | (token == EOS) | (token == PAD)
            if bool(done.all()):
                break
        return ys[:, 1:].view(n, beams, -1), done.view(n, beams)


def peek_n(path):
    """The board size recorded in the first line of a dataset file."""
    with open(path) as f:
        return json.loads(f.readline())["n"]


if __name__ == "__main__":
    for n in (4, 6):
        tok = BoardTokenizer(n)
        print(f"n={n}: src_len={tok.src_len}  "
              f"src vocab={tok.src_vocab_size} (4 specials + digits 0-9)  "
              f"tgt vocab={tok.tgt_vocab_size} (3 specials + {len(ACTIONS)} "
              f"actions + args 0..{MAX_ARG})")
        boards, rows, cols = tok.source_layout()
        pair = (0, 1 + 1 * (n + 1) + 2)   # <INITIAL> block, cell (1,2)
        twin = pair[1] + (n * n + n) + 1  # the same cell in the target board
        print(f"  grid positions: cell (1,2) sits at index {pair[1]} "
              f"(board {boards[pair[1]]}, row {rows[pair[1]]}, "
              f"col {cols[pair[1]]}) and at index {twin} "
              f"(board {boards[twin]}, row {rows[twin]}, col {cols[twin]})")
        for level in MODEL_LEVELS:
            cfg = MODEL_LEVELS[level]
            sizes = " / ".join(
                f"{TranSGridModel.from_level(level, tok, src_pos=p).n_params() / 1e6:.2f}M"
                for p in ("grid", "flat"))
            print(f"  {level}: {cfg['layers']}+{cfg['layers']} layers, "
                  f"d={cfg['d_model']}, heads={cfg['heads']} "
                  f"(dim {cfg['d_model'] // cfg['heads']}), "
                  f"ff={cfg['d_ff']} -> {sizes} params (grid / flat)")
