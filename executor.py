"""
executor.py — v1 geometry DSL executor

This executor takes a program AST (dict) and returns:
- a boolean mask rasterization (H x W)
- metadata computed from the mask in *unit coordinates*
- the unit-space bbox used for rendering

Design goals:
- Minimal dependencies
- Deterministic execution for ground-truth dataset evaluation
- Works with AST nodes:
  Rect, Circle, Translate, Union, Difference
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
import numpy as np


BBox = Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


@dataclass(frozen=True)
class RenderConfig:
    """Rasterization configuration in unit-space."""
    H: int = 256
    W: int = 256
    margin: float = 1.0  
    bbox_units: Optional[BBox] = None


@dataclass
class ExecResult:
    mask: np.ndarray  # bool array of shape (H, W)
    render_cfg: RenderConfig
    bbox_units: BBox  
    metadata: Dict[str, Any]


# -----------------------------
# AST utilities (bbox + raster)
# -----------------------------

def _bbox_rect(w: float, h: float) -> BBox:
    return (-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)

def _bbox_circle(r: float) -> BBox:
    return (-r, -r, r, r)

def _bbox_union(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

def _bbox_translate(bb: BBox, dx: float, dy: float) -> BBox:
    return (bb[0] + dx, bb[1] + dy, bb[2] + dx, bb[3] + dy)

def compute_bbox(ast: Dict[str, Any]) -> BBox:
    """Compute a conservative unit-space bounding box for the given AST."""
    t = ast.get("type")
    if t == "Rect":
        return _bbox_rect(float(ast["w"]), float(ast["h"]))
    if t == "Circle":
        return _bbox_circle(float(ast["r"]))
    if t == "Translate":
        bb = compute_bbox(ast["shape"])
        return _bbox_translate(bb, float(ast["dx"]), float(ast["dy"]))
    if t == "Union":
        return _bbox_union(compute_bbox(ast["a"]), compute_bbox(ast["b"]))
    if t == "Difference":
        return compute_bbox(ast["a"])  # cutouts don't expand extents
    raise ValueError(f"Unknown AST node type: {t!r}")


def _make_grid(bbox_units: BBox, H: int, W: int):
    """Create a unit-space sampling grid (pixel centers)."""
    xmin, ymin, xmax, ymax = bbox_units
    xs = np.linspace(xmin, xmax, W, endpoint=True)
    ys = np.linspace(ymin, ymax, H, endpoint=True)
    X, Y = np.meshgrid(xs, ys)
    dx = (xmax - xmin) / max(W - 1, 1)
    dy = (ymax - ymin) / max(H - 1, 1)
    return X, Y, dx, dy


def _rasterize(ast: Dict[str, Any], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Rasterize AST into a bool mask using unit-space grids."""
    t = ast.get("type")
    if t == "Rect":
        w = float(ast["w"]); h = float(ast["h"])
        return (np.abs(X) <= (w / 2.0)) & (np.abs(Y) <= (h / 2.0))
    if t == "Circle":
        r = float(ast["r"])
        return (X * X + Y * Y) <= (r * r)
    if t == "Translate":
        dx = float(ast["dx"]); dy = float(ast["dy"])
        return _rasterize(ast["shape"], X - dx, Y - dy)
    if t == "Union":
        return _rasterize(ast["a"], X, Y) | _rasterize(ast["b"], X, Y)
    if t == "Difference":
        return _rasterize(ast["a"], X, Y) & (~_rasterize(ast["b"], X, Y))
    raise ValueError(f"Unknown AST node type: {t!r}")


# -----------------------------
# Metadata extraction utilities
# -----------------------------

def _mask_bbox_units(mask: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Optional[BBox]:
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    xmin = float(X[ys, xs].min())
    xmax = float(X[ys, xs].max())
    ymin = float(Y[ys, xs].min())
    ymax = float(Y[ys, xs].max())
    return (xmin, ymin, xmax, ymax)


def _mask_centroid_units(mask: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Optional[Tuple[float, float]]:
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    cx = float(X[ys, xs].mean())
    cy = float(Y[ys, xs].mean())
    return (cx, cy)


def _count_components(mask: np.ndarray) -> int:
    """Count 4-connected components of True pixels."""
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    count = 0

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or visited[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                rr, cc = stack.pop()
                if rr > 0 and mask[rr - 1, cc] and not visited[rr - 1, cc]:
                    visited[rr - 1, cc] = True; stack.append((rr - 1, cc))
                if rr + 1 < H and mask[rr + 1, cc] and not visited[rr + 1, cc]:
                    visited[rr + 1, cc] = True; stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not visited[rr, cc - 1]:
                    visited[rr, cc - 1] = True; stack.append((rr, cc - 1))
                if cc + 1 < W and mask[rr, cc + 1] and not visited[rr, cc + 1]:
                    visited[rr, cc + 1] = True; stack.append((rr, cc + 1))
    return count


def _has_hole(mask: np.ndarray) -> bool:
    """Detect holes by flood-filling background connected to boundary."""
    H, W = mask.shape
    background = ~mask
    visited = np.zeros_like(background, dtype=bool)
    stack: List[Tuple[int, int]] = []

    # Seed from boundary background pixels
    for c in range(W):
        if background[0, c] and not visited[0, c]:
            visited[0, c] = True; stack.append((0, c))
        if background[H - 1, c] and not visited[H - 1, c]:
            visited[H - 1, c] = True; stack.append((H - 1, c))
    for r in range(H):
        if background[r, 0] and not visited[r, 0]:
            visited[r, 0] = True; stack.append((r, 0))
        if background[r, W - 1] and not visited[r, W - 1]:
            visited[r, W - 1] = True; stack.append((r, W - 1))

    while stack:
        r, c = stack.pop()
        if r > 0 and background[r - 1, c] and not visited[r - 1, c]:
            visited[r - 1, c] = True; stack.append((r - 1, c))
        if r + 1 < H and background[r + 1, c] and not visited[r + 1, c]:
            visited[r + 1, c] = True; stack.append((r + 1, c))
        if c > 0 and background[r, c - 1] and not visited[r, c - 1]:
            visited[r, c - 1] = True; stack.append((r, c - 1))
        if c + 1 < W and background[r, c + 1] and not visited[r, c + 1]:
            visited[r, c + 1] = True; stack.append((r, c + 1))

    holes = background & (~visited)
    return bool(holes.any())


def execute(program_ast: Dict[str, Any], render_cfg: Optional[RenderConfig] = None) -> ExecResult:
    """Execute AST: bbox -> raster mask -> metadata."""
    if render_cfg is None:
        render_cfg = RenderConfig()

    # Determine unit-space bbox for rendering
    if render_cfg.bbox_units is not None:
        bbox_units = render_cfg.bbox_units
    else:
        bb = compute_bbox(program_ast)
        bbox_units = (
            bb[0] - render_cfg.margin,
            bb[1] - render_cfg.margin,
            bb[2] + render_cfg.margin,
            bb[3] + render_cfg.margin,
        )

    X, Y, px_dx, px_dy = _make_grid(bbox_units, render_cfg.H, render_cfg.W)
    mask = _rasterize(program_ast, X, Y).astype(bool)

    shape_bbox = _mask_bbox_units(mask, X, Y)
    centroid = _mask_centroid_units(mask, X, Y)

    if shape_bbox is None:
        bbox_w = bbox_h = 0.0
    else:
        bbox_w = float(shape_bbox[2] - shape_bbox[0])
        bbox_h = float(shape_bbox[3] - shape_bbox[1])

    area_units = float(mask.sum()) * float(abs(px_dx * px_dy))
    num_components = _count_components(mask)
    has_hole = _has_hole(mask)

    metadata: Dict[str, Any] = {
        "area_units": area_units,
        "bbox_units": shape_bbox,
        "bbox_width_units": bbox_w,
        "bbox_height_units": bbox_h,
        "centroid_units": centroid,
        "num_components": num_components,
        "has_hole": has_hole,
        "pixel_size_units": (float(px_dx), float(px_dy)),
    }

    return ExecResult(mask=mask, render_cfg=render_cfg, bbox_units=bbox_units, metadata=metadata)
