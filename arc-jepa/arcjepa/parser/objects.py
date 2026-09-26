"""Frozen object representation for the ARC-JEPA parser.

An :class:`Object` is an immutable set of coloured grid cells together with a
handful of cached geometric descriptors. The 24 deterministic features listed in
``docs/FROZEN_SPEC.md`` are exposed through :meth:`Object.features` as a
32-dimensional float32 vector (24 features normalised to [0, 1], zero padded).

Design notes
------------
* Per-cell colours are kept in ``pixels`` (sorted ``(r, c, colour)`` triples) so
  that :meth:`crop` and :meth:`paint` can reproduce multi-colour objects. When
  ``pixels`` is empty (an Object built by hand with the four spec fields only)
  every cell is assumed to carry ``primary_color``.
* Coordinates are *absolute* grid coordinates and may leave the grid after
  :meth:`translate`; :meth:`paint` and :meth:`mask` simply clip such cells.
* Expensive descriptors (perimeter, holes, symmetries) are lazily cached with
  :func:`functools.cached_property`, which is compatible with a frozen dataclass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np

from arcjepa.core.types import Grid, MAX_SIDE, PAD_ID

__all__ = ["Object", "FEATURE_NAMES", "N_FEATURES", "FEATURE_DIM"]

#: Names of the 24 deterministic features, in the order produced by :meth:`Object.features`.
FEATURE_NAMES: Tuple[str, ...] = (
    "primary_color",  # 0  colour / 9
    "n_colors",  # 1  distinct colours / 10 (summary of the colour histogram)
    "area",  # 2  cells / (H*W)
    "bbox_r0",  # 3  / (H-1)
    "bbox_c0",  # 4  / (W-1)
    "bbox_r1",  # 5  / (H-1)
    "bbox_c1",  # 6  / (W-1)
    "bbox_h",  # 7  / H
    "bbox_w",  # 8  / W
    "centroid_r",  # 9  / (H-1)
    "centroid_c",  # 10 / (W-1)
    "aspect",  # 11 w / (h + w)
    "density",  # 12 cells / (h*w)
    "perimeter",  # 13 exposed 4-edges / (4*area)
    "holes",  # 14 min(holes, 8) / 8
    "sym_h",  # 15 left-right mirror symmetric
    "sym_v",  # 16 top-bottom mirror symmetric
    "sym_d1",  # 17 main-diagonal (transpose) symmetric
    "sym_d2",  # 18 anti-diagonal symmetric
    "orientation",  # 19 principal-axis angle / pi in [0, 1)
    "touch_top",  # 20
    "touch_bottom",  # 21
    "touch_left",  # 22
    "touch_right",  # 23
)
N_FEATURES: int = len(FEATURE_NAMES)  # 24
FEATURE_DIM: int = 32

_D4_TRANSFORMS = (
    lambda r, c, h, w: (r, w - 1 - c),  # flip left-right
    lambda r, c, h, w: (h - 1 - r, c),  # flip top-bottom
    lambda r, c, h, w: (h - 1 - r, w - 1 - c),  # rotate 180
    lambda r, c, h, w: (c, r),  # transpose (d1)
    lambda r, c, h, w: (w - 1 - c, h - 1 - r),  # anti-transpose (d2)
    lambda r, c, h, w: (c, h - 1 - r),  # rotate 90 clockwise
    lambda r, c, h, w: (w - 1 - c, r),  # rotate 270 clockwise
)


@dataclass(frozen=True)
class Object:
    """An immutable set of coloured cells with cached geometric descriptors.

    Attributes:
        cells: absolute ``(r, c)`` coordinates of every cell.
        color_hist: ``((colour, count), ...)`` sorted by colour.
        primary_color: most frequent colour (ties -> smallest colour value).
        bbox: ``(r0, c0, r1, c1)`` inclusive bounding box.
        pixels: sorted ``(r, c, colour)`` triples; empty means "all primary_color".
    """

    cells: FrozenSet[Tuple[int, int]]
    color_hist: Tuple[Tuple[int, int], ...]
    primary_color: int
    bbox: Tuple[int, int, int, int]
    pixels: Tuple[Tuple[int, int, int], ...] = ()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_cells(cls, cells: Iterable[Tuple[int, int, int]]) -> "Object":
        """Build an Object from ``(r, c, colour)`` triples (must be non-empty)."""
        px = tuple(sorted(set(cells)))
        if not px:
            raise ValueError("Object.from_cells needs at least one cell")
        if len(px) == 1:  # fast path: single cell
            r, c, col = px[0]
            return cls(frozenset(((r, c),)), ((col, 1),), int(col), (r, c, r, c), px)
        counts: Dict[int, int] = {}
        r0 = c0 = 1 << 30
        r1 = c1 = -(1 << 30)
        for r, c, col in px:
            counts[col] = counts.get(col, 0) + 1
            if r < r0:
                r0 = r
            if r > r1:
                r1 = r
            if c < c0:
                c0 = c
            if c > c1:
                c1 = c
        hist = tuple(sorted(counts.items()))
        primary = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        return cls(frozenset((r, c) for r, c, _ in px), hist, int(primary), (r0, c0, r1, c1), px)

    @classmethod
    def from_mask_color(cls, cells: Iterable[Tuple[int, int]], color: int) -> "Object":
        """Build a single-colour Object from ``(r, c)`` coordinates."""
        return cls.from_cells((r, c, color) for r, c in cells)

    # ------------------------------------------------------------------ colour access
    @cached_property
    def _color_map(self) -> Dict[Tuple[int, int], int]:
        if self.pixels:
            return {(r, c): col for r, c, col in self.pixels}
        return {rc: self.primary_color for rc in self.cells}

    def color_at(self, r: int, c: int) -> Optional[int]:
        """Colour of cell ``(r, c)`` or ``None`` if the cell is not part of the object."""
        return self._color_map.get((r, c))

    def iter_pixels(self) -> Tuple[Tuple[int, int, int], ...]:
        """Sorted ``(r, c, colour)`` triples (reconstructed from ``primary_color`` if needed)."""
        if self.pixels:
            return self.pixels
        return tuple(sorted((r, c, self.primary_color) for r, c in self.cells))

    # ------------------------------------------------------------------ basic geometry
    @property
    def area(self) -> int:
        """Number of cells."""
        return len(self.cells)

    @property
    def h(self) -> int:
        """Bounding-box height."""
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def w(self) -> int:
        """Bounding-box width."""
        return self.bbox[3] - self.bbox[1] + 1

    @cached_property
    def centroid(self) -> Tuple[float, float]:
        """Mean ``(row, col)`` of the cells."""
        n = len(self.cells)
        sr = sc = 0
        for r, c in self.cells:
            sr += r
            sc += c
        return (sr / n, sc / n)

    @property
    def aspect(self) -> float:
        """Width / height of the bounding box."""
        return self.w / self.h

    @property
    def density(self) -> float:
        """Cells / bounding-box area."""
        return self.area / (self.h * self.w)

    @cached_property
    def perimeter(self) -> int:
        """Number of cell edges (4-neighbourhood) facing a non-object cell."""
        cells = self.cells
        p = 0
        for r, c in cells:
            if (r - 1, c) not in cells:
                p += 1
            if (r + 1, c) not in cells:
                p += 1
            if (r, c - 1) not in cells:
                p += 1
            if (r, c + 1) not in cells:
                p += 1
        return p

    @cached_property
    def holes(self) -> int:
        """Number of enclosed 4-connected background regions inside the bounding box."""
        r0, c0, r1, c1 = self.bbox
        hh, ww = self.h, self.w
        inside = [[False] * ww for _ in range(hh)]
        for r, c in self.cells:
            inside[r - r0][c - c0] = True
        seen = [[False] * ww for _ in range(hh)]
        # flood fill the exterior from the bbox border
        stack: List[Tuple[int, int]] = []
        for r in range(hh):
            for c in (0, ww - 1):
                if not inside[r][c] and not seen[r][c]:
                    seen[r][c] = True
                    stack.append((r, c))
        for c in range(ww):
            for r in (0, hh - 1):
                if not inside[r][c] and not seen[r][c]:
                    seen[r][c] = True
                    stack.append((r, c))
        _flood(stack, inside, seen, hh, ww)
        holes = 0
        for r in range(hh):
            for c in range(ww):
                if not inside[r][c] and not seen[r][c]:
                    holes += 1
                    seen[r][c] = True
                    _flood([(r, c)], inside, seen, hh, ww)
        return holes

    # ------------------------------------------------------------------ shape / symmetry
    @cached_property
    def shape(self) -> FrozenSet[Tuple[int, int]]:
        """Cells relative to the bounding-box origin (translation-invariant shape)."""
        r0, c0 = self.bbox[0], self.bbox[1]
        return frozenset((r - r0, c - c0) for r, c in self.cells)

    @cached_property
    def colored_shape(self) -> FrozenSet[Tuple[int, int, int]]:
        """``(r, c, colour)`` relative to the bounding-box origin."""
        r0, c0 = self.bbox[0], self.bbox[1]
        return frozenset((r - r0, c - c0, col) for r, c, col in self.iter_pixels())

    def _is_symmetric(self, f) -> bool:
        h, w = self.h, self.w
        cs = self.colored_shape
        for r, c, col in cs:
            rr, cc = f(r, c, h, w)
            if (rr, cc, col) not in cs:
                return False
        return True

    @cached_property
    def sym_h(self) -> bool:
        """Left-right mirror symmetry (about a vertical axis) of the coloured crop."""
        return self._is_symmetric(_D4_TRANSFORMS[0])

    @cached_property
    def sym_v(self) -> bool:
        """Top-bottom mirror symmetry (about a horizontal axis) of the coloured crop."""
        return self._is_symmetric(_D4_TRANSFORMS[1])

    @cached_property
    def sym_d1(self) -> bool:
        """Main-diagonal (transpose) symmetry; False for non-square boxes."""
        return self.h == self.w and self._is_symmetric(_D4_TRANSFORMS[3])

    @cached_property
    def sym_d2(self) -> bool:
        """Anti-diagonal symmetry; False for non-square boxes."""
        return self.h == self.w and self._is_symmetric(_D4_TRANSFORMS[4])

    @cached_property
    def orientation(self) -> float:
        """Principal-axis angle of the cell cloud mapped to [0, 1) (0.5 = isotropic / horizontal)."""
        n = len(self.cells)
        cr, cc = self.centroid
        vr = vc = cov = 0.0
        for r, c in self.cells:
            dr, dc = r - cr, c - cc
            vr += dr * dr
            vc += dc * dc
            cov += dr * dc
        vr /= n
        vc /= n
        cov /= n
        if abs(cov) < 1e-12 and abs(vc - vr) < 1e-12:
            theta = 0.0
        else:
            theta = 0.5 * math.atan2(2.0 * cov, vc - vr)  # (-pi/2, pi/2]
        val = (theta + math.pi / 2) / math.pi
        return min(max(val, 0.0), 1.0 - 1e-7)

    def d4_shape_keys(self) -> Tuple[FrozenSet[Tuple[int, int]], ...]:
        """The 7 non-identity D4 images of :attr:`shape`, each re-normalised to origin (0, 0)."""
        h, w = self.h, self.w
        out = []
        for f in _D4_TRANSFORMS:
            pts = [f(r, c, h, w) for r, c in self.shape]
            mr = min(p[0] for p in pts)
            mc = min(p[1] for p in pts)
            out.append(frozenset((r - mr, c - mc) for r, c in pts))
        return tuple(out)

    # ------------------------------------------------------------------ grid-relative
    def touches(self, grid_h: int, grid_w: int) -> Tuple[bool, bool, bool, bool]:
        """``(top, bottom, left, right)`` - whether the bbox touches each grid border."""
        r0, c0, r1, c1 = self.bbox
        return (r0 <= 0, r1 >= grid_h - 1, c0 <= 0, c1 >= grid_w - 1)

    def features(self, grid_h: int, grid_w: int) -> np.ndarray:
        """The 24 deterministic features normalised to [0, 1], zero padded to 32 (float32)."""
        H = max(int(grid_h), 1)
        W = max(int(grid_w), 1)
        r0, c0, r1, c1 = self.bbox
        cr, cc = self.centroid
        hd = max(H - 1, 1)
        wd = max(W - 1, 1)
        h, w, area = self.h, self.w, self.area
        top, bottom, left, right = self.touches(H, W)
        f = np.zeros(FEATURE_DIM, dtype=np.float32)
        f[0] = self.primary_color / 9.0
        f[1] = len(self.color_hist) / 10.0
        f[2] = area / (H * W)
        f[3] = r0 / hd
        f[4] = c0 / wd
        f[5] = r1 / hd
        f[6] = c1 / wd
        f[7] = h / H
        f[8] = w / W
        f[9] = cr / hd
        f[10] = cc / wd
        f[11] = w / (h + w)
        f[12] = self.density
        f[13] = self.perimeter / (4.0 * area)
        f[14] = min(self.holes, 8) / 8.0
        f[15] = float(self.sym_h)
        f[16] = float(self.sym_v)
        f[17] = float(self.sym_d1)
        f[18] = float(self.sym_d2)
        f[19] = self.orientation
        f[20] = float(top)
        f[21] = float(bottom)
        f[22] = float(left)
        f[23] = float(right)
        np.clip(f, 0.0, 1.0, out=f)
        return f

    # ------------------------------------------------------------------ rendering
    def crop(self, pad_to: int = MAX_SIDE) -> np.ndarray:
        """``int8[pad_to, pad_to]`` crop anchored at the bbox origin; PAD_ID everywhere else."""
        arr = np.full((pad_to, pad_to), PAD_ID, dtype=np.int8)
        r0, c0 = self.bbox[0], self.bbox[1]
        for r, c, col in self.iter_pixels():
            rr, cc = r - r0, c - c0
            if 0 <= rr < pad_to and 0 <= cc < pad_to:
                arr[rr, cc] = col
        return arr

    def paint(self, canvas: Grid, color: Optional[int] = None) -> Grid:
        """Return a copy of ``canvas`` with the object's cells painted (clipped to the canvas)."""
        out = [list(row) for row in canvas]
        H = len(out)
        W = len(out[0]) if H else 0
        for r, c, col in self.iter_pixels():
            if 0 <= r < H and 0 <= c < W:
                out[r][c] = int(col if color is None else color)
        return out

    def translate(self, dr: int, dc: int) -> "Object":
        """Shift every cell by ``(dr, dc)`` (no clipping)."""
        return Object.from_cells((r + dr, c + dc, col) for r, c, col in self.iter_pixels())

    def recolor(self, c: int) -> "Object":
        """Return the same shape painted uniformly in colour ``c``."""
        return Object.from_cells((r, cc, int(c)) for r, cc, _ in self.iter_pixels())

    def mask(self, h: int, w: int) -> List[List[bool]]:
        """Boolean ``h x w`` mask of the object's cells (cells outside are dropped)."""
        m = [[False] * w for _ in range(h)]
        for r, c in self.cells:
            if 0 <= r < h and 0 <= c < w:
                m[r][c] = True
        return m

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Object(color={self.primary_color}, area={self.area}, bbox={self.bbox})"


def _flood(stack: List[Tuple[int, int]], inside: List[List[bool]], seen: List[List[bool]], hh: int, ww: int) -> None:
    """4-connected flood fill over non-object cells, marking ``seen`` in place."""
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < hh and 0 <= nc < ww and not inside[nr][nc] and not seen[nr][nc]:
                seen[nr][nc] = True
                stack.append((nr, nc))
