"""Minimal ``Object`` used by the DSL when ``arcjepa.parser.objects`` is not available yet.

Field layout is exactly INTERFACES.md §2 so the real parser Object is a drop-in replacement; the DSL only relies
on the four fields (``cells``, ``color_hist``, ``primary_color``, ``bbox``) and constructs objects by keyword.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from arcjepa.core.types import Grid, PAD_ID

__all__ = ["Object"]


def _hist_from_cells(colors: Dict[Tuple[int, int], int]) -> Tuple[Tuple[int, int], ...]:
    counts: Dict[int, int] = {}
    for c in colors.values():
        counts[c] = counts.get(c, 0) + 1
    return tuple(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


@dataclass(frozen=True)
class Object:
    """A connected (or otherwise grouped) set of grid cells with a colour histogram and an inclusive bbox."""

    cells: FrozenSet[Tuple[int, int]]
    color_hist: Tuple[Tuple[int, int], ...]
    primary_color: int
    bbox: Tuple[int, int, int, int]  # r0, c0, r1, c1 inclusive

    # ------------------------------------------------------------------ constructors
    @staticmethod
    def from_cells(cells, color: int) -> "Object":
        """Build a single-colour object from an iterable of ``(r, c)`` cells."""
        fs = frozenset(cells)
        if not fs:
            raise ValueError("Object needs at least one cell")
        rs = [r for r, _ in fs]
        cs = [c for _, c in fs]
        return Object(cells=fs, color_hist=((int(color), len(fs)),), primary_color=int(color),
                      bbox=(min(rs), min(cs), max(rs), max(cs)))

    @staticmethod
    def from_colored_cells(colored: Dict[Tuple[int, int], int]) -> "Object":
        """Build an object from a ``{(r, c): colour}`` mapping (primary colour = most common)."""
        if not colored:
            raise ValueError("Object needs at least one cell")
        hist = _hist_from_cells(colored)
        rs = [r for r, _ in colored]
        cs = [c for _, c in colored]
        return Object(cells=frozenset(colored), color_hist=hist, primary_color=hist[0][0],
                      bbox=(min(rs), min(cs), max(rs), max(cs)))

    # ------------------------------------------------------------------ geometry
    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def h(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def w(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def centroid(self) -> Tuple[float, float]:
        n = len(self.cells)
        return (sum(r for r, _ in self.cells) / n, sum(c for _, c in self.cells) / n)

    @property
    def aspect(self) -> float:
        return self.h / self.w

    @property
    def density(self) -> float:
        return self.area / (self.h * self.w)

    @cached_property
    def perimeter(self) -> int:
        cells = self.cells
        return sum(1 for (r, c) in cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                   if (r + dr, c + dc) not in cells)

    @cached_property
    def holes(self) -> int:
        """Number of 4-connected background components enclosed by the object inside its bbox."""
        return len(self.hole_components())

    def hole_components(self) -> List[FrozenSet[Tuple[int, int]]]:
        """Background components inside the bbox that do not touch the bbox border."""
        r0, c0, r1, c1 = self.bbox
        cells = self.cells
        seen = set()
        comps: List[FrozenSet[Tuple[int, int]]] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) in cells or (r, c) in seen:
                    continue
                stack = [(r, c)]
                seen.add((r, c))
                comp = []
                touches = False
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    if cr in (r0, r1) or cc in (c0, c1):
                        touches = True
                    for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                        if r0 <= nr <= r1 and c0 <= nc <= c1 and (nr, nc) not in cells and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
                if not touches:
                    comps.append(frozenset(comp))
        return comps

    def _local(self) -> FrozenSet[Tuple[int, int]]:
        r0, c0, _, _ = self.bbox
        return frozenset((r - r0, c - c0) for r, c in self.cells)

    @property
    def sym_h(self) -> bool:
        """Mirror symmetry about the vertical axis (left-right)."""
        loc = self._local()
        w = self.w
        return all((r, w - 1 - c) in loc for r, c in loc)

    @property
    def sym_v(self) -> bool:
        """Mirror symmetry about the horizontal axis (top-bottom)."""
        loc = self._local()
        h = self.h
        return all((h - 1 - r, c) in loc for r, c in loc)

    @property
    def sym_d1(self) -> bool:
        if self.h != self.w:
            return False
        loc = self._local()
        return all((c, r) in loc for r, c in loc)

    @property
    def sym_d2(self) -> bool:
        if self.h != self.w:
            return False
        loc = self._local()
        n = self.h
        return all((n - 1 - c, n - 1 - r) in loc for r, c in loc)

    @property
    def orientation(self) -> int:
        """0 = square bbox, 1 = taller than wide, 2 = wider than tall."""
        if self.h == self.w:
            return 0
        return 1 if self.h > self.w else 2

    def touches(self, grid_h: int, grid_w: int) -> Tuple[bool, bool, bool, bool]:
        r0, c0, r1, c1 = self.bbox
        return (r0 <= 0, r1 >= grid_h - 1, c0 <= 0, c1 >= grid_w - 1)

    # ------------------------------------------------------------------ tensors
    def features(self, grid_h: int, grid_w: int) -> np.ndarray:
        """24 deterministic features normalised to [0, 1] where natural, zero padded to 32."""
        r0, c0, r1, c1 = self.bbox
        cr, cc = self.centroid
        t, b, l, r = self.touches(grid_h, grid_w)
        hw = max(grid_h * grid_w, 1)
        f = [
            self.primary_color / 9.0, len(self.color_hist) / 10.0, self.area / hw,
            r0 / 30.0, c0 / 30.0, r1 / 30.0, c1 / 30.0, self.h / 30.0, self.w / 30.0,
            cr / 30.0, cc / 30.0, min(self.aspect, 4.0) / 4.0, self.density, min(self.perimeter, 120) / 120.0,
            min(self.holes, 10) / 10.0, float(self.sym_h), float(self.sym_v), float(self.sym_d1),
            float(self.sym_d2), self.orientation / 2.0, float(t), float(b), float(l), float(r),
        ]
        out = np.zeros(32, dtype=np.float32)
        out[: len(f)] = np.asarray(f, dtype=np.float32)
        return out

    def crop(self, pad_to: int = 30) -> np.ndarray:
        arr = np.full((pad_to, pad_to), PAD_ID, dtype=np.int8)
        r0, c0, _, _ = self.bbox
        for r, c in self.cells:
            rr, cc = r - r0, c - c0
            if 0 <= rr < pad_to and 0 <= cc < pad_to:
                arr[rr, cc] = self.primary_color
        return arr

    # ------------------------------------------------------------------ grid ops
    def paint(self, canvas: Grid, color: Optional[int] = None) -> Grid:
        """Return a copy of ``canvas`` with the object's cells painted (clipped to the canvas)."""
        col = self.primary_color if color is None else int(color)
        out = [list(row) for row in canvas]
        h = len(out)
        w = len(out[0]) if h else 0
        for r, c in self.cells:
            if 0 <= r < h and 0 <= c < w:
                out[r][c] = col
        return out

    def translate(self, dr: int, dc: int) -> "Object":
        r0, c0, r1, c1 = self.bbox
        return Object(cells=frozenset((r + dr, c + dc) for r, c in self.cells), color_hist=self.color_hist,
                      primary_color=self.primary_color, bbox=(r0 + dr, c0 + dc, r1 + dr, c1 + dc))

    def recolor(self, c: int) -> "Object":
        return Object(cells=self.cells, color_hist=((int(c), len(self.cells)),), primary_color=int(c),
                      bbox=self.bbox)

    def mask(self, h: int, w: int) -> List[List[bool]]:
        m = [[False] * w for _ in range(h)]
        for r, c in self.cells:
            if 0 <= r < h and 0 <= c < w:
                m[r][c] = True
        return m
