"""Random ARC-like input grids for synthetic tasks (INTERFACES.md §3, ``generators.py``).

:func:`random_input_grid` draws one grid in one of six styles:

* ``objects``       1..6 non-overlapping random shapes (rectangles, blobs, lines, dots) on a background
* ``lines``         full or partial horizontal / vertical lines
* ``tiles``         a small random motif tiled over the canvas (optionally with a few defects)
* ``noise_sparse``  sparse random pixels (density 3..20 %)
* ``frames``        hollow rectangles (possibly nested or with a marker pixel inside)
* ``symmetric``     a random quadrant mirrored horizontally, vertically or both

All randomness comes from the ``rng`` argument (``random.Random``), so grids are deterministic given a seed.
Every returned grid is valid (1..30 per side, colours 0..9) and contains at least one non-background cell.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from arcjepa.core.types import Grid, MAX_SIDE

__all__ = ["STYLES", "STYLE_WEIGHTS", "SHAPE_HINTS", "random_input_grid", "random_palette", "random_size",
           "shaped_input_grid"]

STYLES: Tuple[str, ...] = ("objects", "lines", "tiles", "noise_sparse", "frames", "symmetric")
#: Default style distribution used by the task sampler (object scenes dominate, as in ARC).
STYLE_WEIGHTS: Dict[str, float] = {"objects": 0.45, "lines": 0.1, "tiles": 0.1, "noise_sparse": 0.1,
                                   "frames": 0.15, "symmetric": 0.1}

Cell = Tuple[int, int]


def random_size(rng: random.Random, lo: int = 3, hi: int = MAX_SIDE) -> int:
    """Side length skewed towards typical ARC sizes (most mass in 5..15, tail up to ``hi``)."""
    lo = max(1, lo)
    hi = max(lo, min(MAX_SIDE, hi))
    u = rng.random()
    if u < 0.75:
        a, b = max(lo, 5), min(hi, 15)
    elif u < 0.93:
        a, b = max(lo, 3), min(hi, 20)
    else:
        a, b = lo, hi
    if a > b:
        a, b = lo, hi
    return rng.randint(a, b)


def random_palette(rng: random.Random, background: int = 0, k: Optional[int] = None,
                   include: Sequence[int] = ()) -> List[int]:
    """``k`` distinct foreground colours (≠ ``background``), always containing ``include`` (minus background)."""
    base = [c for c in dict.fromkeys(include) if 0 <= c <= 9 and c != background]
    if k is None:
        k = rng.choice((1, 2, 2, 3, 3, 4))
    k = max(k, len(base), 1)
    rest = [c for c in range(10) if c != background and c not in base]
    rng.shuffle(rest)
    return base + rest[: max(0, k - len(base))]


# ======================================================================================= shape helpers

def _blob(rng: random.Random, max_h: int, max_w: int) -> Set[Cell]:
    """Random 4-connected blob inside a ``max_h`` x ``max_w`` box, anchored at (0, 0)-relative coordinates."""
    target = rng.randint(2, max(2, min(max_h * max_w, 12)))
    start = (rng.randrange(max_h), rng.randrange(max_w))
    cells = {start}
    frontier = [start]
    tries = 0
    while len(cells) < target and tries < target * 8:
        tries += 1
        r, c = rng.choice(frontier)
        dr, dc = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        nr, nc = r + dr, c + dc
        if 0 <= nr < max_h and 0 <= nc < max_w and (nr, nc) not in cells:
            cells.add((nr, nc))
            frontier.append((nr, nc))
    return cells


def _random_shape(rng: random.Random, h: int, w: int) -> Set[Cell]:
    """Relative cell set of a random shape that fits in an ``h`` x ``w`` canvas."""
    kind = rng.random()
    mh = max(1, min(h, rng.randint(1, max(1, min(6, h // 2 + 1)))))
    mw = max(1, min(w, rng.randint(1, max(1, min(6, w // 2 + 1)))))
    if kind < 0.35:  # filled rectangle
        return {(r, c) for r in range(mh) for c in range(mw)}
    if kind < 0.75:  # blob
        return _blob(rng, mh, mw)
    if kind < 0.9:  # line segment
        if rng.random() < 0.5:
            return {(0, c) for c in range(mw)}
        return {(r, 0) for r in range(mh)}
    return {(0, 0)}  # dot


def _dilate(occupied: Set[Cell], gap: int) -> Set[Cell]:
    if gap <= 0 or not occupied:
        return occupied
    rng_ = range(-gap, gap + 1)
    return {(r + dr, c + dc) for r, c in occupied for dr in rng_ for dc in rng_}


def place_shape(rng: random.Random, occupied: Set[Cell], rel: Set[Cell], h: int, w: int, gap: int = 1,
                tries: int = 20) -> Optional[Set[Cell]]:
    """Translate relative cells ``rel`` to a random free location (``gap`` cells of clearance); ``None`` if none."""
    rh = max(r for r, _ in rel) + 1
    rw = max(c for _, c in rel) + 1
    if rh > h or rw > w:
        return None
    blocked = _dilate(occupied, gap)
    for _ in range(tries):
        r0 = rng.randint(0, h - rh)
        c0 = rng.randint(0, w - rw)
        cells = {(r + r0, c + c0) for r, c in rel}
        if blocked.isdisjoint(cells):
            return cells
    return None


# ======================================================================================= styles

def _style_objects(rng: random.Random, g: Grid, palette: List[int], n_objects: Optional[int]) -> None:
    h, w = len(g), len(g[0])
    n = n_objects if n_objects is not None else rng.randint(1, 6)
    occupied: Set[Cell] = set()
    gap = 1 if rng.random() < 0.85 else 0
    for _ in range(n):
        rel = _random_shape(rng, h, w)
        cells = place_shape(rng, occupied, rel, h, w, gap=gap)
        if cells is None:
            continue
        col = rng.choice(palette)
        multi = len(palette) > 1 and len(cells) > 2 and rng.random() < 0.1
        for r, c in cells:
            g[r][c] = rng.choice(palette) if multi else col
        occupied |= cells


def _style_lines(rng: random.Random, g: Grid, palette: List[int], n_objects: Optional[int]) -> None:
    h, w = len(g), len(g[0])
    n = n_objects if n_objects is not None else rng.randint(1, 4)
    for _ in range(n):
        col = rng.choice(palette)
        full = rng.random() < 0.6
        if rng.random() < 0.5:
            r = rng.randrange(h)
            a, b = (0, w - 1) if full else sorted((rng.randrange(w), rng.randrange(w)))
            for c in range(a, b + 1):
                g[r][c] = col
        else:
            c = rng.randrange(w)
            a, b = (0, h - 1) if full else sorted((rng.randrange(h), rng.randrange(h)))
            for r in range(a, b + 1):
                g[r][c] = col


def _style_tiles(rng: random.Random, g: Grid, palette: List[int], background: int) -> None:
    h, w = len(g), len(g[0])
    th = rng.randint(1, min(4, h))
    tw = rng.randint(1, min(4, w))
    motif = [[rng.choice(palette) if rng.random() < 0.6 else background for _ in range(tw)] for _ in range(th)]
    for r in range(h):
        for c in range(w):
            g[r][c] = motif[r % th][c % tw]
    if rng.random() < 0.3:  # a few defects
        for _ in range(rng.randint(1, 3)):
            g[rng.randrange(h)][rng.randrange(w)] = rng.choice(palette + [background])


def _style_noise(rng: random.Random, g: Grid, palette: List[int]) -> None:
    h, w = len(g), len(g[0])
    density = rng.uniform(0.03, 0.2)
    for r in range(h):
        row = g[r]
        for c in range(w):
            if rng.random() < density:
                row[c] = rng.choice(palette)


def _style_frames(rng: random.Random, g: Grid, palette: List[int], n_objects: Optional[int]) -> None:
    h, w = len(g), len(g[0])
    n = n_objects if n_objects is not None else rng.randint(1, 3)
    occupied: Set[Cell] = set()
    for _ in range(n):
        fh = rng.randint(3, max(3, min(h, 8)))
        fw = rng.randint(3, max(3, min(w, 8)))
        if fh > h or fw > w:
            continue
        rel = {(r, c) for r in range(fh) for c in range(fw)}
        cells = place_shape(rng, occupied, rel, h, w, gap=1)
        if cells is None:
            continue
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        col = rng.choice(palette)
        for r in range(r0, r0 + fh):
            for c in range(c0, c0 + fw):
                if r in (r0, r0 + fh - 1) or c in (c0, c0 + fw - 1):
                    g[r][c] = col
        if fh >= 5 and fw >= 5 and rng.random() < 0.3:  # nested frame
            col2 = rng.choice(palette)
            for r in range(r0 + 2, r0 + fh - 2):
                for c in range(c0 + 2, c0 + fw - 2):
                    if r in (r0 + 2, r0 + fh - 3) or c in (c0 + 2, c0 + fw - 3):
                        g[r][c] = col2
        elif rng.random() < 0.4:  # marker inside
            g[rng.randint(r0 + 1, r0 + fh - 2)][rng.randint(c0 + 1, c0 + fw - 2)] = rng.choice(palette)
        occupied |= cells


def _style_symmetric(rng: random.Random, g: Grid, palette: List[int], background: int) -> None:
    h, w = len(g), len(g[0])
    mode = rng.choice(("h", "v", "both"))
    qh = (h + 1) // 2 if mode in ("v", "both") else h
    qw = (w + 1) // 2 if mode in ("h", "both") else w
    density = rng.uniform(0.2, 0.6)
    for r in range(qh):
        for c in range(qw):
            if rng.random() < density:
                g[r][c] = rng.choice(palette)
    for r in range(h):
        for c in range(w):
            sr = h - 1 - r if (mode in ("v", "both") and r >= qh) else r
            sc = w - 1 - c if (mode in ("h", "both") and c >= qw) else c
            g[r][c] = g[sr][sc]


def random_input_grid(rng: random.Random, *, h: Optional[int] = None, w: Optional[int] = None,
                      n_objects: Optional[int] = None, palette: Optional[Sequence[int]] = None,
                      background: int = 0, style: str = "objects") -> Grid:
    """Draw one random input grid.

    Args:
        rng: source of randomness.
        h, w: grid shape (random ARC-like sizes when ``None``; clipped to 1..30).
        n_objects: number of objects / lines / frames for the object-like styles (random when ``None``).
        palette: foreground colours to draw from (random 1..4 colours when ``None``); ``background`` is removed.
        background: background colour 0..9.
        style: one of :data:`STYLES`.

    Returns:
        A valid grid with at least one non-background cell.
    """
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {STYLES}")
    min_side = 3 if style == "frames" else 1
    h = random_size(rng, lo=min_side) if h is None else max(1, min(MAX_SIDE, int(h)))
    w = random_size(rng, lo=min_side) if w is None else max(1, min(MAX_SIDE, int(w)))
    pal = [c for c in (palette if palette is not None else random_palette(rng, background)) if c != background]
    if not pal:
        pal = random_palette(rng, background, k=1)
    g: Grid = [[background] * w for _ in range(h)]
    painters: Dict[str, Callable[[], None]] = {
        "objects": lambda: _style_objects(rng, g, pal, n_objects),
        "lines": lambda: _style_lines(rng, g, pal, n_objects),
        "tiles": lambda: _style_tiles(rng, g, pal, background),
        "noise_sparse": lambda: _style_noise(rng, g, pal),
        "frames": lambda: _style_frames(rng, g, pal, n_objects),
        "symmetric": lambda: _style_symmetric(rng, g, pal, background),
    }
    painters[style]()
    if all(v == background for row in g for v in row):  # guarantee some foreground
        g[rng.randrange(h)][rng.randrange(w)] = rng.choice(pal)
    return g


# ======================================================================================= shaped inputs

#: Input shapes required by the size- / structure-constrained spec extensions (see :func:`shaped_input_grid`).
SHAPE_HINTS: Tuple[str, ...] = ("kron", "blocks", "blocks_any", "panels", "small")
_SMALL_STYLES: Tuple[str, ...] = ("objects", "noise_sparse", "symmetric", "tiles")


def _blocks(rng: random.Random, pal: List[int], background: int, k: int, any_mode: bool, style: str) -> Grid:
    """A small grid upscaled by ``k`` with per-block noise: majority-preserving flips (``any_mode`` False) or
    partially emptied non-background blocks (``any_mode`` True), so DOWNSCALE / DOWNSCALE_ANY recover it."""
    side = max(2, 30 // k)
    bh, bw = rng.randint(2, min(7, side)), rng.randint(2, min(7, side))
    base = random_input_grid(rng, h=bh, w=bw, palette=pal, background=background,
                             style=style if style in _SMALL_STYLES else "noise_sparse")
    g: Grid = [[background] * (bw * k) for _ in range(bh * k)]
    minority = (k * k - 1) // 2  # cells a block may change and keep its majority
    for r in range(bh):
        for c in range(bw):
            v = base[r][c]
            cells = [(r * k + i, c * k + j) for i in range(k) for j in range(k)]
            for rr, cc in cells:
                g[rr][cc] = v
            if any_mode:
                if v != background and rng.random() < 0.7:
                    keep = rng.randint(1, k * k - 1)
                    for rr, cc in rng.sample(cells, k * k - keep):
                        g[rr][cc] = background
            elif minority and rng.random() < 0.35:
                for rr, cc in rng.sample(cells, rng.randint(1, minority)):
                    g[rr][cc] = rng.choice([x for x in pal + [background] if x != v] or [v])
    return g


def _panels(rng: random.Random, pal: List[int], background: int, style: str, min_panels: int = 2) -> Grid:
    """2-4 (at least ``min_panels``) equally shaped random panels side by side (or stacked), joined by separator
    lines of one colour that is not used inside the panels (no separator for 2 panels, 30 % of the time: the halves
    split)."""
    n = rng.choice([x for x in (2, 2, 2, 3, 4) if x >= min_panels] or [4])
    vertical = rng.random() < 0.5  # panels side by side
    ph, pw = rng.randint(3, 6), rng.randint(3, 6)
    sep_free = [c for c in range(1, 10) if c != background and c not in pal]
    use_sep = bool(sep_free) and not (n == 2 and rng.random() < 0.3)
    panel_style = style if style in ("objects", "noise_sparse", "symmetric") else "noise_sparse"
    ps = [random_input_grid(rng, h=ph, w=pw, palette=pal, background=background, style=panel_style)
          for _ in range(n)]
    if not use_sep and n == 2:  # the halves rule splits left / right only when W >= 2H - 1
        vertical = 2 * pw >= 2 * ph - 1
    sep = rng.choice(sep_free) if use_sep else None
    if vertical:
        rows: Grid = [[] for _ in range(ph)]
        for i, p in enumerate(ps):
            for r in range(ph):
                if i and sep is not None:
                    rows[r].append(sep)
                rows[r].extend(p[r])
        return rows
    out: Grid = []
    for i, p in enumerate(ps):
        if i and sep is not None:
            out.append([sep] * pw)
        out.extend(list(row) for row in p)
    return out


def shaped_input_grid(rng: random.Random, hint: str, *, palette: Optional[Sequence[int]] = None,
                      background: int = 0, style: str = "objects", k: Optional[int] = None) -> Grid:
    """Random input grid shaped for a structure- or size-constrained spec extension.

    ``hint`` (one of :data:`SHAPE_HINTS`): ``kron`` -- sides 2..5 (KRON_SELF squares the sides); ``blocks`` /
    ``blocks_any`` -- a small grid upscaled by ``k`` (default 2 or 3) with block noise that DOWNSCALE /
    DOWNSCALE_ANY undo; ``panels`` -- 2-4 equal panels (at least ``k`` when given) joined by separator lines
    (PANEL_BOOL / PANEL_OVERLAY); ``small`` -- sides <= 30 // ``k`` (UPSCALE by ``k``, default 3).  Every result is
    a valid grid with some non-background cell.
    """
    if hint not in SHAPE_HINTS:
        raise ValueError(f"unknown shape hint {hint!r}; expected one of {SHAPE_HINTS}")
    pal = [c for c in (palette if palette is not None else random_palette(rng, background)) if c != background]
    if not pal:
        pal = random_palette(rng, background, k=1)
    small_style = style if style in _SMALL_STYLES else "objects"
    if hint == "kron":
        g = random_input_grid(rng, h=rng.randint(2, 5), w=rng.randint(2, 5), palette=pal, background=background,
                              style=small_style)
    elif hint in ("blocks", "blocks_any"):
        kk = k if k is not None and 2 <= k <= 5 else rng.choice((2, 3))
        g = _blocks(rng, pal, background, kk, hint == "blocks_any", style)
    elif hint == "panels":
        g = _panels(rng, pal, background, style, min_panels=k if k is not None else 2)
    else:
        kk = k if k is not None and k >= 1 else 3
        side = max(1, min(10, MAX_SIDE // kk))
        g = random_input_grid(rng, h=rng.randint(min(2, side), side), w=rng.randint(min(2, side), side),
                              palette=pal, background=background, style=small_style)
    if all(v == background for row in g for v in row):
        g[rng.randrange(len(g))][rng.randrange(len(g[0]))] = rng.choice(pal)
    return g
