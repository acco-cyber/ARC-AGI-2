"""Program tokenizer and program encoder.

Tokenizer: a DSL program (an ``arcjepa.dsl.ast.Node`` or anything with ``.op`` / ``.args``, or its S-expression
string) becomes a flat pre-order id stream. Every AST node -- a primitive application or a literal argument --
contributes the triple ``(op-or-literal id, type id, depth id)``, wrapped in ``<bos>`` ... ``<eos>``. Depth ids
make the stream invertible without arity information, so ``decode`` rebuilds the tree exactly (literals inside
the DSL literal domains). The vocabulary is built from ``arcjepa.dsl.primitives.REGISTRY`` when importable and
otherwise from the 72 primitive names of ``docs/FROZEN_SPEC.md`` plus the structural ops (INPUT, RENDER,
PAINT_ON_BLANK, NEURAL_OP); the two vocabularies coincide as long as the DSL registers exactly those names
(extra registry names are appended in sorted order).

Encoder (spec §Model "primitive emb 128 + type emb, 6 tree/transformer blocks d 256"): the stream is regrouped
into one token per AST node, embedded as [primitive-or-literal emb 128 ; type emb 64 ; depth emb 64] -> 256 plus
a learned pre-order position; a learned summary token is prepended, 6 pre-norm blocks (d 256) run over
[summary, nodes] and the summary output is projected to z_p in R^256. One token per node keeps sequences 3x
shorter than the id stream (a depth-6 program is <= ~40 nodes), which matters when search scores many candidates.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from arcjepa.utils.model_blocks import TransformerStack, padding_mask_from_valid

from .config import ModelConfig

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- spec vocabulary (FROZEN_SPEC §DSL)
SPEC_PRIMITIVES: Dict[str, Tuple[str, ...]] = {
    "selection": ("SELECT_ALL", "SELECT_COLOR", "SELECT_NONZERO", "SELECT_LARGEST", "SELECT_SMALLEST",
                  "SELECT_UNIQUE", "SELECT_BORDER", "SELECT_CENTER"),
    "analysis": ("GET_COMPONENTS4", "GET_COMPONENTS8", "GET_BBOX", "GET_CENTROID", "GET_AREA", "GET_PERIMETER",
                 "GET_HOLES", "GET_SYMMETRY"),
    "spatial": ("LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "TOUCHING", "OVERLAPPING", "INSIDE", "CONTAINS",
                "NEAREST", "FARTHEST"),
    "geometric": ("ROTATE90", "ROTATE180", "ROTATE270", "REFLECT_H", "REFLECT_V", "REFLECT_D1", "REFLECT_D2",
                  "TRANSPOSE", "SHIFT", "ALIGN"),
    "manipulation": ("COPY", "MOVE", "DELETE", "DUPLICATE", "MERGE", "SPLIT", "EXTEND", "SHRINK", "GROW", "FILL",
                     "OUTLINE", "FRAME"),
    "colour": ("RECOLOR", "SWAP_COLORS", "MAP_COLOR", "MOST_COMMON_COLOR", "LEAST_COMMON_COLOR",
               "REPLACE_BACKGROUND", "COLOR_OBJECT", "COLOR_BY_POSITION"),
    "pattern": ("TILE", "REPEAT_X", "REPEAT_Y", "REPEAT_N", "MIRROR_TILE", "PATTERN_FILL", "ALTERNATE",
                "PERIODIC_REPEAT"),
    "counting": ("COUNT_OBJECTS", "COUNT_CELLS", "ARGMAX_SIZE", "ARGMIN_SIZE"),
    "conditional": ("IF", "APPLY_TO_EACH", "FILTER", "COMPOSE"),
}
SPEC_PRIMITIVE_NAMES: Tuple[str, ...] = tuple(n for names in SPEC_PRIMITIVES.values() for n in names)
STRUCTURAL_OPS: Tuple[str, ...] = ("INPUT", "OBJ", "RENDER", "RENDER_OBJ", "RENDER_BLANK", "CROP", "PAINT_ON_BLANK",
                                   "NEURAL_OP")
LEAF_OPS: Tuple[str, ...] = ("INPUT", "OBJ")
TYPE_NAMES: Tuple[str, ...] = ("GRID", "OBJECT_SET", "OBJECT", "MASK", "COLOR", "POSITION", "INTEGER", "BOOLEAN",
                               "RELATION", "PROGRAM")
POSITION_ANCHORS: Tuple[str, ...] = ("center", "top", "bottom", "left", "right")
#: RELATION string literals (the DSL's ``arcjepa.dsl.types.RELATIONS``; mirrored here so the vocab exists before it)
RELATION_LITERALS: Tuple[str, ...] = ("LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "TOUCHING", "OVERLAPPING", "INSIDE",
                                      "CONTAINS", "SAME_COLOR", "SAME_SHAPE", "SAME_SIZE", "LARGER", "SMALLER")
INT_LITERAL_RANGE = range(-9, 31)
POSITION_RANGE = range(-3, 4)

# best-effort output types used only when the DSL registry is unavailable (auxiliary tokens; decode ignores them)
_FALLBACK_OUT_TYPES: Dict[str, str] = {}
for _n in ("SELECT_ALL", "SELECT_COLOR", "SELECT_NONZERO", "SELECT_BORDER", "GET_COMPONENTS4", "GET_COMPONENTS8",
           "SPLIT", "APPLY_TO_EACH", "FILTER"):
    _FALLBACK_OUT_TYPES[_n] = "OBJECT_SET"
for _n in ("SELECT_LARGEST", "SELECT_SMALLEST", "SELECT_UNIQUE", "SELECT_CENTER", "NEAREST", "FARTHEST",
           "COPY", "MOVE", "DUPLICATE", "MERGE", "EXTEND", "SHRINK", "GROW", "FILL", "OUTLINE", "FRAME",
           "RECOLOR", "COLOR_OBJECT", "ARGMAX_SIZE", "ARGMIN_SIZE", "GET_BBOX"):
    _FALLBACK_OUT_TYPES[_n] = "OBJECT"
for _n in ("GET_AREA", "GET_PERIMETER", "GET_HOLES", "COUNT_OBJECTS", "COUNT_CELLS"):
    _FALLBACK_OUT_TYPES[_n] = "INTEGER"
for _n in ("LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "TOUCHING", "OVERLAPPING", "INSIDE", "CONTAINS", "GET_SYMMETRY"):
    _FALLBACK_OUT_TYPES[_n] = "BOOLEAN"
for _n in ("MOST_COMMON_COLOR", "LEAST_COMMON_COLOR"):
    _FALLBACK_OUT_TYPES[_n] = "COLOR"
_FALLBACK_OUT_TYPES["GET_CENTROID"] = "POSITION"
_COLOR_ARG_OPS = frozenset({"SELECT_COLOR", "RECOLOR", "SWAP_COLORS", "MAP_COLOR", "REPLACE_BACKGROUND",
                            "COLOR_OBJECT", "COLOR_BY_POSITION", "FILL", "OUTLINE", "FRAME", "PATTERN_FILL"})

KIND_SPECIAL, KIND_OP, KIND_TYPE, KIND_DEPTH, KIND_LITERAL = 0, 1, 2, 3, 4
N_KINDS = 5


def _load_registry() -> Optional[Dict[str, Any]]:
    try:
        from arcjepa.dsl.primitives import REGISTRY  # type: ignore
        return dict(REGISTRY)
    except Exception:  # noqa: BLE001 - DSL not available yet
        return None


def _load_dsl_node() -> Optional[type]:
    try:
        from arcjepa.dsl.ast import Node  # type: ignore
        return Node
    except Exception:  # noqa: BLE001
        return None


def _type_name(t: Any) -> str:
    return str(getattr(t, "name", getattr(t, "value", t)))


# --------------------------------------------------------------------------- fallback AST
@dataclass(frozen=True)
class SimpleNode:
    """Minimal stand-in for ``arcjepa.dsl.ast.Node`` (same fields and S-expression format)."""

    op: str
    args: Tuple[Any, ...] = ()

    def to_str(self) -> str:
        if not self.args:
            return self.op
        return "(" + " ".join([self.op] + [_literal_to_str(a) for a in self.args]) + ")"

    def depth(self) -> int:
        """Nested operator levels; leaves have depth 0 (same convention as ``arcjepa.dsl.ast.Node``)."""
        if not self.args:
            return 0
        return 1 + max((a.depth() for a in self.args if _is_node(a)), default=0)

    def size(self) -> int:
        return 1 + sum(a.size() for a in self.args if _is_node(a))

    def __str__(self) -> str:
        return self.to_str()

    @staticmethod
    def from_str(s: str) -> "SimpleNode":
        return parse_sexpr(s)


def _is_node(x: Any) -> bool:
    return hasattr(x, "op") and hasattr(x, "args")


def _literal_to_str(a: Any) -> str:
    if _is_node(a):
        return a.to_str()
    if isinstance(a, bool):
        return "True" if a else "False"
    if isinstance(a, tuple):
        return "(" + " ".join(str(int(v)) for v in a) + ")"  # DSL position format ``(dr dc)``
    return str(a)


_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


def parse_sexpr(s: str) -> SimpleNode:
    """Parse an S-expression such as ``(RECOLOR (SELECT_LARGEST (GET_COMPONENTS4 INPUT)) 3)``.

    Literals: ints, ``True``/``False``, ``(dr dc)`` (or ``(dr,dc)``) positions, bare strings (anchor and relation
    names). Only ``INPUT`` and ``OBJ`` are leaf nodes, as in ``arcjepa.dsl.ast``.
    """
    toks = _TOKEN_RE.findall(s)
    pos = 0

    def atom(t: str) -> Any:
        if t in ("True", "False"):
            return t == "True"
        if re.fullmatch(r"-?\d+", t):
            return int(t)
        m = re.fullmatch(r"\(?(-?\d+),(-?\d+)\)?", t)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
            return t[1:-1]
        return t

    def parse() -> Any:
        nonlocal pos
        if pos >= len(toks):
            raise ValueError("unexpected end of program string")
        t = toks[pos]
        pos += 1
        if t == "(":
            items: List[Any] = []
            while pos < len(toks) and toks[pos] != ")":
                items.append(parse())
            if pos >= len(toks):
                raise ValueError("unbalanced parentheses")
            pos += 1  # ')'
            if not items:
                raise ValueError("empty list")
            head = items[0]
            if len(items) == 1 and isinstance(head, tuple):
                return head  # "(dr,dc)" tokenised as "(", "dr,dc", ")"
            if isinstance(head, int) and not isinstance(head, bool) and len(items) == 2 and isinstance(items[1], int):
                return (head, items[1])  # (dr dc) position written with a space
            if not isinstance(head, str):
                raise ValueError(f"bad operator {head!r}")
            return SimpleNode(head, tuple(items[1:]))
        if t == ")":
            raise ValueError("unexpected ')'")
        a = atom(t)
        if isinstance(a, str) and a in LEAF_OPS:
            return SimpleNode(a)  # bare leaf op (INPUT / OBJ); other bare names are string literals
        return a

    node = parse()
    if pos != len(toks):
        raise ValueError("trailing tokens in program string")
    if not _is_node(node):
        raise ValueError("program string does not denote an operator application")
    return node


# --------------------------------------------------------------------------- tokenizer
class ProgramTokenizer:
    """Pre-order (op / literal, type, depth) triple tokenizer with an invertible ``decode``."""

    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, registry: Optional[Dict[str, Any]] = None, *, max_depth_tokens: int = 16,
                 use_dsl: bool = True, extra_ops: Iterable[str] = ()) -> None:
        self.registry = registry if registry is not None else (_load_registry() if use_dsl else None)
        self.node_cls: type = (_load_dsl_node() if use_dsl else None) or SimpleNode
        self.max_depth_tokens = int(max_depth_tokens)
        ops = list(SPEC_PRIMITIVE_NAMES) + list(STRUCTURAL_OPS)
        extra = set(extra_ops)
        if self.registry:
            extra |= set(self.registry.keys())
        ops += sorted(n for n in extra if n not in ops)
        self.op_names: Tuple[str, ...] = tuple(ops)
        self.itos: List[str] = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.kinds: List[int] = [KIND_SPECIAL] * 4
        self._add(self.op_names, KIND_OP)
        self._add([f"TYPE_{t}" for t in TYPE_NAMES] + ["TYPE_UNK"], KIND_TYPE)
        self._add([f"DEPTH_{d}" for d in range(self.max_depth_tokens)], KIND_DEPTH)
        lits = [f"LIT_INT_{v}" for v in INT_LITERAL_RANGE]
        lits += [f"LIT_POS_{dr}_{dc}" for dr in POSITION_RANGE for dc in POSITION_RANGE]
        lits += [f"LIT_ANCHOR_{a}" for a in POSITION_ANCHORS]
        lits += [f"LIT_REL_{r}" for r in RELATION_LITERALS]
        lits += ["LIT_BOOL_True", "LIT_BOOL_False", "LIT_UNK"]
        self._add(lits, KIND_LITERAL)
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}
        self.unk_lit_id = self.stoi["LIT_UNK"]
        self._depth_base = self.stoi["DEPTH_0"]
        self._type_unk = self.stoi["TYPE_UNK"]

    def _add(self, names: Iterable[str], kind: int) -> None:
        for n in names:
            self.itos.append(n)
            self.kinds.append(kind)

    # ---------------------------------------------------------------- properties
    @property
    def vocab(self) -> Dict[str, int]:
        return dict(self.stoi)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.PAD

    def kind_ids_tensor(self) -> Tensor:
        return torch.tensor(self.kinds, dtype=torch.long)

    def kind_of(self, token_id: int) -> int:
        return self.kinds[token_id]

    # ---------------------------------------------------------------- typing helpers
    def _type_id(self, name: str) -> int:
        return self.stoi.get(f"TYPE_{name}", self._type_unk)

    def _op_type_id(self, op: str) -> int:
        if self.registry and op in self.registry:
            return self._type_id(_type_name(self.registry[op].out_type))
        if op == "OBJ":
            return self._type_id("OBJECT")
        if op in STRUCTURAL_OPS:
            return self._type_id("GRID")
        for cat, names in SPEC_PRIMITIVES.items():
            if op in names:
                return self._type_id(_FALLBACK_OUT_TYPES.get(op, "GRID"))
        return self._type_unk

    def _literal_type_id(self, parent_op: str, idx: int, value: Any) -> int:
        if self.registry and parent_op in self.registry:
            arg_types = getattr(self.registry[parent_op], "arg_types", ())
            if idx < len(arg_types):
                return self._type_id(_type_name(arg_types[idx]))
        if isinstance(value, bool):
            return self._type_id("BOOLEAN")
        if isinstance(value, tuple) or (isinstance(value, str) and value in POSITION_ANCHORS):
            return self._type_id("POSITION")
        if isinstance(value, str) and value in RELATION_LITERALS:
            return self._type_id("RELATION")
        if isinstance(value, int):
            return self._type_id("COLOR" if parent_op in _COLOR_ARG_OPS else "INTEGER")
        return self._type_unk

    def _depth_id(self, depth: int) -> int:
        return self._depth_base + min(depth, self.max_depth_tokens - 1)

    # ---------------------------------------------------------------- literals
    def literal_token(self, value: Any) -> str:
        """Vocabulary token for a literal argument (``LIT_UNK`` outside the DSL literal domains)."""
        if isinstance(value, bool):
            return f"LIT_BOOL_{value}"
        if isinstance(value, int) and value in INT_LITERAL_RANGE:
            return f"LIT_INT_{value}"
        if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value) \
                and value[0] in POSITION_RANGE and value[1] in POSITION_RANGE:
            return f"LIT_POS_{value[0]}_{value[1]}"
        if isinstance(value, str) and value in POSITION_ANCHORS:
            return f"LIT_ANCHOR_{value}"
        if isinstance(value, str) and value in RELATION_LITERALS:
            return f"LIT_REL_{value}"
        return "LIT_UNK"

    @staticmethod
    def literal_value(token: str) -> Any:
        """Inverse of ``literal_token`` (``None`` for ``LIT_UNK``)."""
        if token.startswith("LIT_BOOL_"):
            return token.endswith("True")
        if token.startswith("LIT_INT_"):
            return int(token[len("LIT_INT_"):])
        if token.startswith("LIT_POS_"):
            dr, dc = token[len("LIT_POS_"):].split("_")
            return (int(dr), int(dc))
        if token.startswith("LIT_ANCHOR_"):
            return token[len("LIT_ANCHOR_"):]
        if token.startswith("LIT_REL_"):
            return token[len("LIT_REL_"):]
        return None

    # ---------------------------------------------------------------- encode / decode
    def as_node(self, program: Any) -> Any:
        """Accept a Node-like object or an S-expression string."""
        if _is_node(program):
            return program
        if isinstance(program, str):
            if self.node_cls is not SimpleNode and hasattr(self.node_cls, "from_str"):
                try:
                    return self.node_cls.from_str(program)
                except Exception:  # noqa: BLE001 - fall back to the local parser
                    log.debug("dsl Node.from_str failed on %r; using the local S-expression parser", program)
            return parse_sexpr(program)
        raise TypeError(f"cannot tokenize {type(program).__name__}")

    def encode(self, program: Any) -> List[int]:
        """Program -> flat id list ``[<bos>, (id, type, depth)*, <eos>]``."""
        node = self.as_node(program)
        out: List[int] = [self.BOS]
        self._walk(node, 0, out)
        out.append(self.EOS)
        return out

    def _walk(self, node: Any, depth: int, out: List[int]) -> None:
        op = str(node.op)
        out += [self.stoi.get(op, self.UNK), self._op_type_id(op), self._depth_id(depth)]
        for idx, arg in enumerate(node.args):
            if _is_node(arg):
                self._walk(arg, depth + 1, out)
            else:
                out += [self.stoi.get(self.literal_token(arg), self.unk_lit_id),
                        self._literal_type_id(op, idx, arg), self._depth_id(depth + 1)]

    def decode(self, ids: Sequence[int]) -> Any:
        """Flat id list -> Node (``arcjepa.dsl.ast.Node`` when importable, else ``SimpleNode``)."""
        ids = [int(i) for i in ids]
        body = [i for i in ids if i not in (self.PAD, self.BOS)]
        if self.EOS in body:
            body = body[:body.index(self.EOS)]
        if len(body) % 3 != 0 or not body:
            raise ValueError("token stream is not a sequence of (token, type, depth) triples")
        frames: List[List[Any]] = []  # [depth, op, args]
        root: Optional[Any] = None

        def close_to(depth: int) -> None:
            nonlocal root
            while frames and frames[-1][0] >= depth:
                d, op, args = frames.pop()
                node = self.node_cls(op, tuple(args))
                if frames:
                    frames[-1][2].append(node)
                else:
                    root = node

        for k in range(0, len(body), 3):
            tok, _typ, dep = body[k], body[k + 1], body[k + 2]
            if self.kinds[dep] != KIND_DEPTH:
                raise ValueError(f"expected a depth token at position {k + 2}")
            depth = dep - self._depth_base
            kind = self.kinds[tok]
            if kind == KIND_OP:
                close_to(depth)
                frames.append([depth, self.itos[tok], []])
            elif kind == KIND_LITERAL:
                close_to(depth)
                if not frames:
                    raise ValueError("literal without a parent node")
                frames[-1][2].append(self.literal_value(self.itos[tok]))
            elif tok == self.UNK:
                raise ValueError("cannot decode an <unk> operator")
            else:
                raise ValueError(f"unexpected token {self.itos[tok]!r} at position {k}")
        close_to(0)
        if root is None:
            raise ValueError("empty program")
        return root

    # ---------------------------------------------------------------- batching
    def pad_batch(self, seqs: Sequence[Sequence[int]], max_len: Optional[int] = None,
                  device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
        """Pad id lists to ``(tokens Long[B, L], mask Bool[B, L])`` (truncated to ``max_len`` when given)."""
        if len(seqs) == 0:
            return torch.zeros(0, 1, dtype=torch.long, device=device), torch.zeros(0, 1, dtype=torch.bool, device=device)
        length = max(1, max(len(s) for s in seqs))
        if max_len is not None:
            length = min(length, max_len)
        tokens = torch.full((len(seqs), length), self.PAD, dtype=torch.long, device=device)
        mask = torch.zeros(len(seqs), length, dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            s = list(s)[:length]
            if s:
                tokens[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                mask[i, :len(s)] = True
        return tokens, mask

    def encode_batch(self, programs: Sequence[Any], max_len: Optional[int] = None,
                     device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
        """Encode and pad several programs at once."""
        return self.pad_batch([self.encode(p) for p in programs], max_len=max_len, device=device)

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> Dict[str, Any]:
        return {"itos": list(self.itos), "kinds": list(self.kinds), "max_depth_tokens": self.max_depth_tokens,
                "op_names": list(self.op_names)}

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ProgramTokenizer":
        """Rebuild a tokenizer with the vocabulary saved by ``save`` (ops beyond the spec set are restored)."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        extra = [n for n in d.get("op_names", []) if n not in SPEC_PRIMITIVE_NAMES and n not in STRUCTURAL_OPS]
        tok = cls(max_depth_tokens=int(d.get("max_depth_tokens", 16)), extra_ops=extra)
        if tok.itos != d["itos"]:
            log.warning("loaded vocabulary differs from the rebuilt one; using the saved order")
            tok.itos = list(d["itos"])
            tok.kinds = list(d["kinds"])
            tok.stoi = {s: i for i, s in enumerate(tok.itos)}
            tok.unk_lit_id = tok.stoi["LIT_UNK"]
            tok._depth_base = tok.stoi["DEPTH_0"]
            tok._type_unk = tok.stoi["TYPE_UNK"]
        return tok


def max_nodes_for(max_program_len: int) -> int:
    """Number of node slots of a ``[<bos>, triples..., <eos>]`` stream capped at ``max_program_len`` ids."""
    return max(1, (int(max_program_len) + 1) // 3)


def group_triples(tokens: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Regroup padded ``[<bos>, (sym, type, depth)*, <eos>, <pad>...]`` streams into per-node id triples.

    Args:
        tokens: Long[B, L] id streams from ``ProgramTokenizer`` (``<bos>`` first).
        mask: Bool[B, L] validity (ids outside the mask are treated as ``<pad>``).
    Returns:
        sym, typ, depth Long[B, G] and node_valid Bool[B, G] with G = ceil((L - 1) / 3); the ``<eos>`` group and
        padding groups are invalid.
    """
    b, length = tokens.shape
    toks = tokens.long().masked_fill(~mask.bool(), ProgramTokenizer.PAD)
    body = toks[:, 1:]
    g = max(1, -(-(length - 1) // 3))
    extra = 3 * g - body.shape[1]
    if extra > 0:
        body = torch.cat([body, body.new_full((b, extra), ProgramTokenizer.PAD)], dim=1)
    sym, typ, dep = body.view(b, g, 3).unbind(-1)
    node_valid = (sym != ProgramTokenizer.PAD) & (sym != ProgramTokenizer.BOS) & (sym != ProgramTokenizer.EOS)
    return sym, typ, dep, node_valid


class ProgramEncoder(nn.Module):
    """forward(tokens Long[B,L], mask Bool[B,L]) -> Float[B,program_dim].

    Token streams must come from ``ProgramTokenizer.encode`` (``<bos>`` first, then (id, type, depth) triples);
    all-padding rows (missing candidates) are allowed and yield the encoding of the empty program.
    """

    def __init__(self, cfg: ModelConfig, tokenizer: Optional[ProgramTokenizer] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer if tokenizer is not None else ProgramTokenizer(max_depth_tokens=cfg.program_depth_tokens)
        d = cfg.program_dim
        v = self.tokenizer.vocab_size
        self.sym_emb = nn.Embedding(v, cfg.program_sym_dim)  # primitive / literal
        self.type_emb = nn.Embedding(v, cfg.program_aux_dim)  # indexed by the TYPE_* ids of the stream
        self.depth_emb = nn.Embedding(v, cfg.program_aux_dim)  # indexed by the DEPTH_* ids of the stream
        self.in_proj = nn.Linear(cfg.program_sym_dim + 2 * cfg.program_aux_dim, d)
        self.max_nodes = max_nodes_for(cfg.max_program_len)
        self.pos_emb = nn.Embedding(self.max_nodes, d)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.summary_token, std=0.02)
        self.blocks = TransformerStack(d, cfg.program_layers, cfg.heads, cfg.ffn_dim, cfg.dropout)
        self.out = nn.Linear(d, d)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Encode padded token streams (Long[B, L]) into program latents Float[B, program_dim]."""
        if mask is None:
            mask = tokens != ProgramTokenizer.PAD
        cap = self.cfg.max_program_len
        if tokens.shape[1] > cap:
            log.warning("program token stream of length %d truncated to %d", tokens.shape[1], cap)
            tokens, mask = tokens[:, :cap], mask[:, :cap]
        mask = mask.bool()
        if tokens.shape[0] == 0:
            return self.out_norm.weight.new_zeros(0, self.cfg.program_dim)
        if bool((mask[:, 0] & (tokens[:, 0] != ProgramTokenizer.BOS)).any()):
            raise ValueError("program token streams must start with <bos> (use ProgramTokenizer.encode)")
        sym, typ, dep, node_valid = group_triples(tokens, mask)
        b, g = sym.shape
        x = self.in_proj(torch.cat([self.sym_emb(sym), self.type_emb(typ), self.depth_emb(dep)], dim=-1))
        x = x + self.pos_emb(torch.arange(g, device=x.device)).unsqueeze(0)
        seq = torch.cat([self.summary_token.expand(b, 1, -1), x], dim=1)
        valid = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=x.device), node_valid], dim=1)
        h = self.blocks(seq, padding_mask_from_valid(valid))
        return self.out_norm(self.out(h[:, 0]))
