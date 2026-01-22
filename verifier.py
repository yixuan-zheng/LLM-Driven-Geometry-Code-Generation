"""verifier.py — v1 constraint checker for the geometry dataset

This verifier checks the constraint schema emitted in geometry_dataset_v1.jsonl.

Inputs:
- exec_result: output from executor.execute() (mask + metadata in unit-space)
- constraints: list[dict] constraints (each dict includes a 'type' field)

Outputs:
- VerifyResult: per-constraint pass/fail, failures, and diagnostics that can later
  be turned into LLM feedback.

Notes:
- We intentionally keep checks deterministic and tolerance-based to account for
  rasterization discretization.
- v1 focuses on the constraint types we emitted in the dataset generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np


@dataclass
class ConstraintResult:
    constraint: Dict[str, Any]
    passed: bool
    measured: Any
    message: str


@dataclass
class VerifyResult:
    passed: bool
    per_constraint: List[ConstraintResult]
    failed_constraints: List[ConstraintResult]
    diagnostics: Dict[str, Any]


# -----------------------------
# Helper utilities
# -----------------------------

def _rel_err(measured: float, target: float) -> float:
    if target == 0:
        return abs(measured - target)
    return abs(measured - target) / abs(target)

def _within(measured: float, target: float, tol: float) -> bool:
    """Relative tolerance by default."""
    return _rel_err(measured, target) <= tol

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _tol_to_abs(tol: float, scale: float) -> float:
    """Convert relative tolerance to absolute using a scale (units)."""
    return tol * max(scale, 1e-6)


def _hole_mask(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask of hole pixels (background not connected to boundary)."""
    H, W = mask.shape
    background = ~mask
    visited = np.zeros_like(background, dtype=bool)
    stack = []

    # boundary seeds
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

    return background & (~visited)


def _largest_hole_bbox(exec_result) -> Optional[Tuple[float, float, float, float]]:
    """Compute bbox of the largest hole (by pixel area)."""
    mask = exec_result.mask
    H, W = mask.shape

    # Rebuild coordinate grids in unit-space (same mapping as executor)
    xmin, ymin, xmax, ymax = exec_result.bbox_units
    xs = np.linspace(xmin, xmax, exec_result.render_cfg.W, endpoint=True)
    ys = np.linspace(ymin, ymax, exec_result.render_cfg.H, endpoint=True)
    X, Y = np.meshgrid(xs, ys)

    holes = _hole_mask(mask)
    if not holes.any():
        return None

    # Connected components in hole mask, pick largest
    visited = np.zeros_like(holes, dtype=bool)
    best_pixels = []
    best_count = 0

    for r in range(H):
        for c in range(W):
            if not holes[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pixels = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                if rr > 0 and holes[rr - 1, cc] and not visited[rr - 1, cc]:
                    visited[rr - 1, cc] = True; stack.append((rr - 1, cc)); pixels.append((rr - 1, cc))
                if rr + 1 < H and holes[rr + 1, cc] and not visited[rr + 1, cc]:
                    visited[rr + 1, cc] = True; stack.append((rr + 1, cc)); pixels.append((rr + 1, cc))
                if cc > 0 and holes[rr, cc - 1] and not visited[rr, cc - 1]:
                    visited[rr, cc - 1] = True; stack.append((rr, cc - 1)); pixels.append((rr, cc - 1))
                if cc + 1 < W and holes[rr, cc + 1] and not visited[rr, cc + 1]:
                    visited[rr, cc + 1] = True; stack.append((rr, cc + 1)); pixels.append((rr, cc + 1))
            if len(pixels) > best_count:
                best_count = len(pixels)
                best_pixels = pixels

    ys_idx = np.array([p[0] for p in best_pixels], dtype=int)
    xs_idx = np.array([p[1] for p in best_pixels], dtype=int)
    xmin_h = float(X[ys_idx, xs_idx].min())
    xmax_h = float(X[ys_idx, xs_idx].max())
    ymin_h = float(Y[ys_idx, xs_idx].min())
    ymax_h = float(Y[ys_idx, xs_idx].max())
    return (xmin_h, ymin_h, xmax_h, ymax_h)


def _hole_centroid(exec_result) -> Optional[Tuple[float, float]]:
    hb = _largest_hole_bbox(exec_result)
    if hb is None:
        return None
    return ((hb[0] + hb[2]) / 2.0, (hb[1] + hb[3]) / 2.0)


# -----------------------------
# Constraint checks
# -----------------------------

def verify(exec_result, constraints: List[Dict[str, Any]]) -> VerifyResult:
    md = exec_result.metadata
    per: List[ConstraintResult] = []
    diag: Dict[str, Any] = {}

    # Derived quantities
    bbox = md.get("bbox_units")
    bbox_w = float(md.get("bbox_width_units") or 0.0)
    bbox_h = float(md.get("bbox_height_units") or 0.0)
    centroid = md.get("centroid_units")

    diag["bbox_units"] = bbox
    diag["centroid_units"] = centroid
    diag["area_units"] = md.get("area_units")
    diag["num_components"] = md.get("num_components")
    diag["has_hole"] = md.get("has_hole")

    # Hole info computed lazily
    hole_bbox = None
    hole_cent = None

    for c in constraints:
        ctype = c.get("type")
        passed = True
        measured = None
        msg = ""

        # -------- Basic numeric constraints --------
        if ctype == "bbox_width":
            target = float(c["value"]); tol = float(c.get("tol", 0.05))
            measured = bbox_w
            passed = _within(measured, target, tol)
            msg = f"bbox_width_units={measured:.3f}, target={target:.3f}, rel_err={_rel_err(measured,target):.3f}"

        elif ctype == "bbox_height":
            target = float(c["value"]); tol = float(c.get("tol", 0.05))
            measured = bbox_h
            passed = _within(measured, target, tol)
            msg = f"bbox_height_units={measured:.3f}, target={target:.3f}, rel_err={_rel_err(measured,target):.3f}"

        elif ctype == "area":
            target = float(c["value"]); tol = float(c.get("tol", 0.08))
            measured = float(md.get("area_units") or 0.0)
            passed = _within(measured, target, tol)
            msg = f"area_units={measured:.3f}, target={target:.3f}, rel_err={_rel_err(measured,target):.3f}"

        elif ctype == "centroid":
            target = tuple(c["value"])
            tol = float(c.get("tol", 0.07))
            measured = centroid
            if measured is None:
                passed = False
                msg = "centroid missing (empty mask?)"
            else:
                scale = max(bbox_w, bbox_h, 1.0)
                abs_tol = _tol_to_abs(tol, scale)
                passed = _dist(measured, target) <= abs_tol
                msg = f"centroid={measured}, target={target}, dist={_dist(measured,target):.3f}, abs_tol={abs_tol:.3f}"

        elif ctype == "num_components":
            target = int(c["value"])
            measured = int(md.get("num_components") or 0)
            passed = (measured == target)
            msg = f"num_components={measured}, target={target}"

        elif ctype == "has_hole":
            target = bool(c["value"])
            measured = bool(md.get("has_hole"))
            passed = (measured == target)
            msg = f"has_hole={measured}, target={target}"

        # -------- Cutout / hole constraints --------
        elif ctype == "cutout_inside_base":
            if hole_bbox is None:
                hole_bbox = _largest_hole_bbox(exec_result)
            measured = hole_bbox
            if bbox is None or hole_bbox is None:
                passed = False
                msg = "missing bbox or hole_bbox"
            else:
                passed = (hole_bbox[0] >= bbox[0] and hole_bbox[1] >= bbox[1] and
                          hole_bbox[2] <= bbox[2] and hole_bbox[3] <= bbox[3])
                msg = f"hole_bbox={hole_bbox}, shape_bbox={bbox}"

        elif ctype == "centered":
            tol = float(c.get("tol", 0.06))
            if hole_cent is None:
                hole_cent = _hole_centroid(exec_result)
            measured = hole_cent
            if measured is None:
                passed = False
                msg = "hole centroid missing"
            else:
                scale = max(bbox_w, bbox_h, 1.0)
                abs_tol = _tol_to_abs(tol, scale)
                passed = _dist(measured, (0.0, 0.0)) <= abs_tol
                msg = f"hole_centroid={measured}, target=(0,0), dist={_dist(measured,(0,0)):.3f}, abs_tol={abs_tol:.3f}"

        elif ctype == "centered_x":
            tol = float(c.get("tol", 0.06))
            if hole_cent is None:
                hole_cent = _hole_centroid(exec_result)
            measured = hole_cent
            if measured is None:
                passed = False
                msg = "hole centroid missing"
            else:
                scale = max(bbox_w, 1.0)
                abs_tol = _tol_to_abs(tol, scale)
                passed = abs(measured[0] - 0.0) <= abs_tol
                msg = f"hole_centroid_x={measured[0]:.3f}, abs_tol={abs_tol:.3f}"

        elif ctype == "circle_center":
            target = tuple(c["value"])
            tol = float(c.get("tol", 0.07))
            if hole_cent is None:
                hole_cent = _hole_centroid(exec_result)
            measured = hole_cent
            if measured is None:
                passed = False
                msg = "hole centroid missing"
            else:
                scale = max(bbox_w, bbox_h, 1.0)
                abs_tol = _tol_to_abs(tol, scale)
                passed = _dist(measured, target) <= abs_tol
                msg = f"hole_centroid={measured}, target={target}, dist={_dist(measured,target):.3f}, abs_tol={abs_tol:.3f}"

        elif ctype == "cutout_size":
            target = float(c["value"]); tol = float(c.get("tol", 0.06))
            if hole_bbox is None:
                hole_bbox = _largest_hole_bbox(exec_result)
            if hole_bbox is None:
                passed = False
                measured = None
                msg = "hole bbox missing"
            else:
                measured = float(hole_bbox[2] - hole_bbox[0])
                passed = _within(measured, target, tol)
                msg = f"hole_bbox_w={measured:.3f}, target={target:.3f}, rel_err={_rel_err(measured,target):.3f}"

        elif ctype == "cutout_radius":
            target = float(c["value"]); tol = float(c.get("tol", 0.06))
            if hole_bbox is None:
                hole_bbox = _largest_hole_bbox(exec_result)
            if hole_bbox is None:
                passed = False
                measured = None
                msg = "hole bbox missing"
            else:
                measured = float(hole_bbox[2] - hole_bbox[0]) / 2.0
                passed = _within(measured, target, tol)
                msg = f"est_r={measured:.3f}, target={target:.3f}, rel_err={_rel_err(measured,target):.3f}"

        elif ctype == "top_edge_y_equals":
            tol = float(c.get("tol", 0.06))
            if hole_bbox is None:
                hole_bbox = _largest_hole_bbox(exec_result)
            measured = hole_bbox
            if hole_bbox is None:
                passed = False
                msg = "hole bbox missing"
            else:
                top_y = hole_bbox[3]
                scale = max(bbox_h, 1.0)
                abs_tol = _tol_to_abs(tol, scale)
                passed = abs(top_y - 0.0) <= abs_tol
                msg = f"hole_top_y={top_y:.3f}, target=0.0, abs_tol={abs_tol:.3f}"

        # -------- Frame / inner rectangle constraints --------
        elif ctype == "inner_bbox_width":
            target = float(c["value"]); tol = float(c.get("tol", 0.08))
            if hole_bbox is None:
                hole_bbox = _largest_hole_bbox(exec_result)
            if hole_bbox is None:
                passed = False; measured = None; msg = "hole bbox missing"
            else:
                measured = float(hole_bbox[2] - hole_bbox[0])
                passed = _within(measured, target, tol)
                msg = f"inner_w={measured:.3f}, target={target:.3f}, rel_err={_rel_err(measured,target):.3f}"

        elif ctype == "inner_bbox_height":
            target = float(c["value"]); tol = float(c.get("tol", 0.08))
            if hole_bbox is None:
                hole_bbox = _largest_hole_bbox(exec_result)
            if hole_bbox is None:
                passed = False; measured = None; msg = "hole bbox missing"
            else:
                measured = float(hole_bbox[3] - hole_bbox[1])
                passed = _within(measured, target, tol)
                msg = f"inner_h={measured:.3f}, target={target:.3f}, rel_err={_rel_err(measured,target):.3f}"

        elif ctype == "inner_rect_centered":
            tol = float(c.get("tol", 0.06))
            if hole_cent is None:
                hole_cent = _hole_centroid(exec_result)
            measured = hole_cent
            if measured is None:
                passed = False; msg = "hole centroid missing"
            else:
                scale = max(bbox_w, bbox_h, 1.0)
                abs_tol = _tol_to_abs(tol, scale)
                passed = _dist(measured, (0.0, 0.0)) <= abs_tol
                msg = f"inner_centroid={measured}, dist={_dist(measured,(0,0)):.3f}, abs_tol={abs_tol:.3f}"

        # -------- Misc constraints (v1 heuristics) --------
        elif ctype == "union_area_greater_than":
            tol = float(c.get("tol", 0.02))
            measured = float(md.get("area_units") or 0.0)
            rect_proxy = bbox_w * bbox_h
            passed = measured >= rect_proxy * (1.0 + tol)
            msg = f"area={measured:.3f}, rect_proxy={rect_proxy:.3f}, required>={rect_proxy*(1+tol):.3f}"

        elif ctype == "union_area_greater_than_rect_area":
            rect_area = float(c["value"])
            tol = float(c.get("tol", 0.02))
            measured = float(md.get("area_units") or 0.0)
            passed = measured >= rect_area * (1.0 + tol)
            msg = f"area={measured:.3f}, rect_area={rect_area:.3f}, required>={rect_area*(1+tol):.3f}"

        elif ctype == "two_lobes_horizontal":
            mask = exec_result.mask
            mid = mask.shape[1] // 2
            left = mask[:, :mid].sum()
            right = mask[:, mid:].sum()
            measured = (int(left), int(right))
            total = left + right
            if total == 0:
                passed = False
                msg = "empty mask"
            else:
                frac_left = left / total
                frac_right = right / total
                passed = (frac_left > 0.3) and (frac_right > 0.3)
                msg = f"left_frac={frac_left:.2f}, right_frac={frac_right:.2f}"

        elif ctype == "identical_subshapes":
            mask = exec_result.mask.astype(np.uint8)
            mid = mask.shape[1] // 2
            left = mask[:, :mid]
            right = mask[:, mid:]
            right_m = np.fliplr(right)

            sig_l = left.sum(axis=0)
            sig_r = right_m.sum(axis=0)

            if sig_l.sum() == 0 or sig_r.sum() == 0:
                passed = False
                measured = None
                msg = "one side empty"
            else:
                l = sig_l / sig_l.sum()
                r = sig_r / sig_r.sum()
                sim = float(np.dot(l, r) / (np.linalg.norm(l) * np.linalg.norm(r) + 1e-9))
                measured = sim
                passed = sim > 0.90
                msg = f"cosine_sim={sim:.3f} (threshold 0.90)"

        elif ctype == "circle_centered_on_base":
            tol = float(c.get("tol", 0.06))
            measured = centroid
            if measured is None:
                passed = False
                msg = "centroid missing"
            else:
                scale = max(bbox_w, bbox_h, 1.0)
                abs_tol = _tol_to_abs(tol, scale)
                passed = _dist(measured, (0.0, 0.0)) <= abs_tol
                msg = f"centroid={measured}, dist={_dist(measured,(0,0)):.3f}, abs_tol={abs_tol:.3f}"

        elif ctype == "circle_inside_rectangle":
            if hole_bbox is None:
                hole_bbox = _largest_hole_bbox(exec_result)
            measured = hole_bbox
            if bbox is None or hole_bbox is None:
                passed = False
                msg = "missing bbox or hole_bbox"
            else:
                passed = (hole_bbox[0] >= bbox[0] and hole_bbox[1] >= bbox[1] and
                          hole_bbox[2] <= bbox[2] and hole_bbox[3] <= bbox[3])
                msg = f"hole_bbox={hole_bbox}, shape_bbox={bbox}"

        elif ctype == "cutout_shape":
            measured = c.get("value")
            passed = True
            msg = "cutout_shape is advisory in v1 (not strictly checked)"

        else:
            passed = False
            measured = None
            msg = f"Unknown constraint type: {ctype!r}"

        per.append(ConstraintResult(constraint=c, passed=passed, measured=measured, message=msg))

    failed = [r for r in per if not r.passed]
    return VerifyResult(
        passed=(len(failed) == 0),
        per_constraint=per,
        failed_constraints=failed,
        diagnostics=diag,
    )