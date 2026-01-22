# repair_utils.py
from __future__ import annotations

import copy
import hashlib
import random
import re
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Deterministic hashing / seeding
# -----------------------------

def stable_int(s: str) -> int:
    """Stable 32-bit int from a string (unlike Python's built-in hash())."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


# -----------------------------
# AST helpers
# -----------------------------

ALLOWED_TYPES = {"Rect", "Circle", "Translate", "Union", "Difference"}

def ast_type(ast: Dict[str, Any]) -> str:
    return ast.get("type", "UNKNOWN")

def is_rect(ast: Dict[str, Any]) -> bool:
    return ast_type(ast) == "Rect"

def is_circle(ast: Dict[str, Any]) -> bool:
    return ast_type(ast) == "Circle"

def is_translate(ast: Dict[str, Any]) -> bool:
    return ast_type(ast) == "Translate"

def is_union(ast: Dict[str, Any]) -> bool:
    return ast_type(ast) == "Union"

def is_diff(ast: Dict[str, Any]) -> bool:
    return ast_type(ast) == "Difference"

def get_shape(ast: Dict[str, Any]) -> Dict[str, Any]:
    return ast["shape"]

def set_shape(ast: Dict[str, Any], shape: Dict[str, Any]) -> None:
    ast["shape"] = shape

def get_a(ast: Dict[str, Any]) -> Dict[str, Any]:
    return ast["a"]

def get_b(ast: Dict[str, Any]) -> Dict[str, Any]:
    return ast["b"]

def set_a(ast: Dict[str, Any], v: Dict[str, Any]) -> None:
    ast["a"] = v

def set_b(ast: Dict[str, Any], v: Dict[str, Any]) -> None:
    ast["b"] = v

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_scale(x: float, scale: float, lo: float = 0.2, hi: float = 1e6) -> float:
    return clamp(x * scale, lo, hi)

def find_first_rect(ast: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    stack = [ast]
    while stack:
        node = stack.pop()
        t = ast_type(node)
        if t == "Rect":
            return node
        if t == "Translate":
            stack.append(node["shape"])
        elif t in ("Union", "Difference"):
            stack.append(node["a"])
            stack.append(node["b"])
    return None

def find_circle_in_ast(ast: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    stack = [ast]
    while stack:
        node = stack.pop()
        t = ast_type(node)
        if t == "Circle":
            return node
        if t == "Translate":
            stack.append(node["shape"])
        elif t in ("Union", "Difference"):
            stack.append(node["a"])
            stack.append(node["b"])
    return None

def find_frame_outer_inner(ast: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    If ast is Difference(outer_rect, inner_rect or Translate(inner_rect)), return (outer_rect, inner_rect_node).
    """
    if ast_type(ast) != "Difference":
        return None, None

    outer = ast.get("a")
    inner = ast.get("b")
    if inner and ast_type(inner) == "Translate":
        inner = inner.get("shape")

    if outer and ast_type(outer) == "Rect" and inner and ast_type(inner) == "Rect":
        return outer, inner
    return None, None


# -----------------------------
# Noisy oracle: start from GT and perturb slightly
# -----------------------------

def noisy_oracle_then_break(ex: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """
    Testing helper: start from ground-truth AST then introduce a small error.
    Deterministic across runs with stable hashing.
    """
    ex_id = str(ex.get("id", "0"))
    rng = random.Random(seed + stable_int(ex_id))

    ast = copy.deepcopy(ex["program_ast"])

    # pick a mutation type
    muts = ["scale_rect", "move_translate", "scale_circle", "swap_union_children"]
    m = muts[rng.randrange(len(muts))]

    def mutate(node: Dict[str, Any]) -> bool:
        t = ast_type(node)

        if t == "Rect" and m == "scale_rect":
            node["w"] = safe_scale(float(node["w"]), 1.0 + rng.choice([-0.15, -0.10, 0.10, 0.15]))
            node["h"] = safe_scale(float(node["h"]), 1.0 + rng.choice([-0.15, -0.10, 0.10, 0.15]))
            return True

        if t == "Circle" and m == "scale_circle":
            node["r"] = safe_scale(float(node["r"]), 1.0 + rng.choice([-0.20, -0.10, 0.10, 0.20]))
            return True

        if t == "Translate" and m == "move_translate":
            node["dx"] = float(node["dx"]) + rng.choice([-1.0, -0.5, 0.5, 1.0])
            node["dy"] = float(node["dy"]) + rng.choice([-1.0, -0.5, 0.5, 1.0])
            return True

        if t == "Union" and m == "swap_union_children":
            node["a"], node["b"] = node["b"], node["a"]
            return True

        return False

    # DFS to find first applicable node
    stack = [ast]
    while stack:
        node = stack.pop()
        if mutate(node):
            return ast

        t = ast_type(node)
        if t == "Translate":
            stack.append(node["shape"])
        elif t in ("Union", "Difference"):
            stack.append(node["a"])
            stack.append(node["b"])

    return ast


# -----------------------------
# Heuristic repair
# -----------------------------

_NUM = r"(-?\d+(?:\.\d+)?)"
NUM = _NUM  # compatibility

def heuristic_repair(ast: Dict[str, Any], failed_results: List[Any]) -> Dict[str, Any]:
    """
    Rule-based AST repair using verifier feedback messages.
    This is meant as a deterministic, non-LLM baseline and harness validator.
    """
    ast2 = copy.deepcopy(ast)

    def nudge(current: float, target: float, frac: float = 0.7) -> float:
        return current + frac * (target - current)

    for fc in failed_results:
        ctype = fc.constraint.get("type", "")
        msg = fc.message

        # -------------------------
        # bbox width/height (global shape bbox)
        # message patterns:
        # "bbox_width_units=..., target=..."
        # "bbox_height_units=..., target=..."
        # -------------------------
        if ctype == "bbox_width":
            m = re.search(r"bbox_width_units=" + NUM + r".*target=" + NUM, msg)
            if m:
                meas, tgt = float(m.group(1)), float(m.group(2))
                rect_node = find_first_rect(ast2)
                if rect_node is not None:
                    # scale rect width toward matching bbox width
                    scale = tgt / max(meas, 1e-6)
                    rect_node["w"] = max(0.2, safe_scale(float(rect_node["w"]), scale))

        if ctype == "bbox_height":
            m = re.search(r"bbox_height_units=" + NUM + r".*target=" + NUM, msg)
            if m:
                meas, tgt = float(m.group(1)), float(m.group(2))
                rect_node = find_first_rect(ast2)
                if rect_node is not None:
                    scale = tgt / max(meas, 1e-6)
                    rect_node["h"] = max(0.2, safe_scale(float(rect_node["h"]), scale))

        # -------------------------
        # centered_x: hole/shape should be centered along x
        # simplest: if there is a Translate in the cutout path, set dx=0
        # -------------------------
        if ctype == "centered_x":
            if is_diff(ast2):
                hole = get_b(ast2)
                if is_translate(hole):
                    hole["dx"] = 0.0

        # -------------------------
        # top_edge_y_equals: often for a translated square cutout
        # message: "hole_top_y=..., target=..., abs_tol=..."
        # top_y = dy + s/2  -> dy = target - s/2
        # -------------------------
        if ctype == "top_edge_y_equals":
            m = re.search(r"hole_top_y=" + NUM + r".*target=" + NUM, msg)
            if m and is_diff(ast2):
                hole = get_b(ast2)
                if is_translate(hole) and is_rect(get_shape(hole)):
                    hole_rect = get_shape(hole)
                    s = float(hole_rect["w"])
                    hole_top_target = float(m.group(2))
                    hole["dy"] = hole_top_target - s / 2.0

        # -------------------------
        # cutout_size: square/rect hole size should match target
        # message: "hole_bbox_w=..., target=..."
        # -------------------------
        if ctype == "cutout_size":
            m = re.search(r"hole_bbox_w=" + NUM + r".*target=" + NUM, msg)
            if m and is_diff(ast2):
                meas, tgt = float(m.group(1)), float(m.group(2))
                hole = get_b(ast2)

                if is_rect(hole):
                    scale = tgt / max(meas, 1e-6)
                    hole["w"] = max(0.2, safe_scale(float(hole["w"]), scale))
                    hole["h"] = float(hole["w"])
                elif is_translate(hole) and is_rect(get_shape(hole)):
                    hole_rect = get_shape(hole)
                    scale = tgt / max(meas, 1e-6)
                    hole_rect["w"] = max(0.2, safe_scale(float(hole_rect["w"]), scale))
                    hole_rect["h"] = float(hole_rect["w"])

        # -------------------------
        # cutout_radius: circular hole radius should match target
        # message formats vary; we parse floats as fallback
        # -------------------------
        if ctype == "cutout_radius":
            # Prefer explicit "target="
            m = re.search(r"target=" + NUM, msg)
            if is_diff(ast2) and m:
                # fallback: first float = measured, second float = target
                nums = re.findall(NUM, msg)
                if len(nums) >= 2:
                    meas, tgt = float(nums[0]), float(nums[1])
                    hole = get_b(ast2)

                    if is_circle(hole):
                        scale = tgt / max(meas, 1e-6)
                        hole["r"] = max(0.2, safe_scale(float(hole["r"]), scale))
                    elif is_translate(hole) and is_circle(get_shape(hole)):
                        circ = get_shape(hole)
                        scale = tgt / max(meas, 1e-6)
                        circ["r"] = max(0.2, safe_scale(float(circ["r"]), scale))

        # -------------------------
        # circle_center: ensure circle is positioned correctly (e.g., offset hole)
        # If we have Difference(rect, Circle) we can wrap with Translate; if already Translate(Circle),
        # we can try to set dx/dy based on any numeric hints in the message.
        # -------------------------
        if ctype == "circle_center":
            if is_diff(ast2):
                hole = get_b(ast2)

                # If hole is bare circle, wrap it
                if is_circle(hole):
                    # attempt to parse dx/dy from message; else default
                    dx = 1.0
                    dy = 1.0
                    mdx = re.search(r"dx=" + NUM, msg)
                    mdy = re.search(r"dy=" + NUM, msg)
                    if mdx:
                        dx = float(mdx.group(1))
                    if mdy:
                        dy = float(mdy.group(1))
                    set_b(ast2, {"type": "Translate", "dx": dx, "dy": dy, "shape": hole})

                # If already translated, maybe nudge toward hinted values
                elif is_translate(hole) and is_circle(get_shape(hole)):
                    mdx = re.search(r"dx=" + NUM, msg)
                    mdy = re.search(r"dy=" + NUM, msg)
                    if mdx:
                        hole["dx"] = float(mdx.group(1))
                    if mdy:
                        hole["dy"] = float(mdy.group(1))

        # -------------------------
        # inner_bbox_height: frame inner rectangle height
        # verifier message (current): "inner_h=..., target=..., rel_err=..."
        # -------------------------
        if ctype == "inner_bbox_height":
            m = re.search(r"inner_h=" + NUM + r".*target=" + NUM, msg)
            if m:
                meas, tgt = float(m.group(1)), float(m.group(2))
                _, inner = find_frame_outer_inner(ast2)
                if inner is not None:
                    scale = tgt / max(meas, 1e-6)
                    inner["h"] = max(0.2, safe_scale(float(inner["h"]), scale))

        # -------------------------
        # inner_bbox_width: (optional) same idea if it appears
        # message: "inner_w=..., target=..."
        # -------------------------
        if ctype == "inner_bbox_width":
            m = re.search(r"inner_w=" + NUM + r".*target=" + NUM, msg)
            if m:
                meas, tgt = float(m.group(1)), float(m.group(2))
                _, inner = find_frame_outer_inner(ast2)
                if inner is not None:
                    scale = tgt / max(meas, 1e-6)
                    inner["w"] = max(0.2, safe_scale(float(inner["w"]), scale))

        # -------------------------
        # union_area_greater_than_rect_area:
        # message: "area=..., rect_area=..., required>=..."
        # Fix: increase circle radius enough to protrude past rect,
        # and then nudge further if still short.
        # -------------------------
        if ctype == "union_area_greater_than_rect_area":
            m = re.search(r"area=" + NUM + r".*rect_area=" + NUM + r".*required>=" + NUM, msg)
            if m:
                area = float(m.group(1))
                rect_area = float(m.group(2))
                required = float(m.group(3))

                circ = find_circle_in_ast(ast2)
                rect = find_first_rect(ast2)

                if circ is not None and rect is not None:
                    w = float(rect.get("w", 0.0))
                    h = float(rect.get("h", 0.0))

                    # Force protrusion if rect and circle are centered
                    r_min = 0.5 * min(w, h) + 0.05
                    circ["r"] = max(float(circ.get("r", 1.0)), r_min)

                    # If still short, bump a bit more based on deficit
                    deficit = max(0.0, required - area)
                    if deficit > 1e-6:
                        bump = 1.0 + min(0.50, deficit / max(required, 1e-6))
                        circ["r"] = safe_scale(float(circ["r"]), bump)

    return ast2