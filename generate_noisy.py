#!/usr/bin/env python3
"""
generate_noisy.py — create a noisy-prompt variant of your geometry dataset.

Goal:
- Keep program_ast + constraints identical.
- Replace prompt with a less parseable, more natural (and sometimes underspecified) prompt.

This enables the loop:
  prompt -> model -> execute -> verify -> feedback -> revise

Usage:
  python generate_noisy.py \
    --input geometry_dataset_v1.jsonl \
    --output geometry_dataset_v1_noisy.jsonl \
    --seed 42 \
    --level medium
"""

from __future__ import annotations

import argparse
import json
import random
import re
from typing import Any, Dict, List, Tuple, Optional


# -----------------------------
# JSONL helpers
# -----------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Utilities
# -----------------------------

_NUM = r"(-?\d+(?:\.\d+)?)"

def _roundish(x: float, rng: random.Random, mode: str) -> str:
    """Return a rounded/approx string for a number depending on noise mode."""
    if mode == "low":
        val = round(x, 1)
        if rng.random() < 0.35:
            return f"about {val}"
        return f"{val}"
    if mode == "medium":
        # integer-ish
        val = int(round(x))
        if rng.random() < 0.60:
            return f"around {val}"
        return f"{val}"
    # high: very vague buckets
    ax = abs(x)
    if ax < 2.5:
        return "small"
    if ax < 6.0:
        return "medium"
    if ax < 12.0:
        return "large"
    return "very large"

def _choose(rng: random.Random, options: List[str]) -> str:
    return options[rng.randrange(len(options))]

def _maybe_add_filler(s: str, rng: random.Random, level: str) -> str:
    if level == "low":
        return s
    if level == "medium" and rng.random() < 0.20:
        return s + " Keep it simple."
    if level == "high" and rng.random() < 0.35:
        return s + " Please return just the geometry program."
    return s

def _replace_synonyms(text: str, rng: random.Random, level: str) -> str:
    """Small synonym swaps (template-agnostic)."""
    swaps = [
        ("Draw", ["Create", "Make", "Generate"]),
        ("rectangle", ["rect", "rectangle"]),
        ("circle", ["circle", "disk"]),
        ("centered", ["centered", "in the middle"]),
        ("shifted", ["shifted", "moved", "translated", "offset"]),
        ("cut out", ["cut out", "subtract", "remove"]),
        ("hole", ["hole", "cutout"]),
    ]
    out = text
    for key, opts in swaps:
        if re.search(rf"\b{re.escape(key)}\b", out, flags=re.IGNORECASE):
            repl = _choose(rng, opts)
            # keep original casing roughly
            out = re.sub(rf"\b{re.escape(key)}\b", repl, out, flags=re.IGNORECASE)
    if level == "high" and rng.random() < 0.30:
        out = out.replace("width", _choose(rng, ["width", "horizontal size"]))
        out = out.replace("height", _choose(rng, ["height", "vertical size"]))
    return out


# -----------------------------
# Template-aware prompt rewriting
# -----------------------------

def noisy_prompt_from_params(template_id: str, params: Dict[str, Any], rng: random.Random, level: str) -> str:
    """
    Build a noisy prompt from structured params when available.

    If a param is missing, we fall back to more generic language.
    """
    # Helper to format numbers
    def fnum(name: str) -> str:
        if name not in params:
            return ""
        return _roundish(float(params[name]), rng, level)

    # Some relational cues (useful for "medium/high")
    def wide_short(w: float, h: float) -> str:
        if w >= 2.5 * h:
            return "wide and short"
        if h >= 2.5 * w:
            return "tall and narrow"
        return "roughly proportional"

    tid = template_id

    if tid == "T1_single_rectangle":
        w = float(params.get("w", 10.0)); h = float(params.get("h", 5.0))
        if level in ("low", "medium"):
            s = f"Draw a rectangle with width {fnum('w')} and height {fnum('h')}."
        else:
            s = f"Draw a {wide_short(w,h)} rectangle."
        return s

    if tid == "T2_single_circle":
        if level in ("low", "medium"):
            s = f"Draw a circle with radius {fnum('r')}."
        else:
            s = "Draw a circle with a small-to-medium radius."
        return s

    if tid == "T3_translated_rectangle":
        w = float(params.get("w", 10.0)); h = float(params.get("h", 5.0))
        dx = float(params.get("dx", 0.0)); dy = float(params.get("dy", 0.0))
        if level == "low":
            s = f"Draw a rectangle of width {fnum('w')} and height {fnum('h')}, shifted by ({_roundish(dx,rng,'low')}, {_roundish(dy,rng,'low')})."
        elif level == "medium":
            s = f"Make a rectangle (about {int(round(w))} by {int(round(h))}) and move it by roughly ({int(round(dx))}, {int(round(dy))})."
        else:
            s = "Make a rectangle and shift it slightly off-center."
        return s

    if tid == "T4_centered_square_cutout":
        if level in ("low", "medium"):
            s = f"Draw a rectangle with width {fnum('w')} and height {fnum('h')}. Cut out a centered square of size {fnum('s')}."
        else:
            s = "Draw a rectangle and cut a square hole in the center."
        return s

    if tid == "T5_offset_square_cutout_top_edge_at_center":
        if level in ("low", "medium"):
            # The key semantic: top edge at y=0, centered in x
            s = f"Draw a rectangle of width {fnum('w')} and height {fnum('h')}. Cut out a square (size {fnum('s')}) so its top edge aligns with the rectangle's horizontal centerline."
        else:
            s = "Cut a square notch-like hole whose top edge sits on the rectangle’s midline (not the top)."
        return s

    if tid == "T6_centered_circular_hole":
        if level in ("low", "medium"):
            s = f"Draw a rectangle with width {fnum('w')} and height {fnum('h')}. Cut out a centered circular hole of radius {fnum('r')}."
        else:
            s = "Draw a rectangle with a centered circular hole."
        return s

    if tid == "T7_union_rect_and_circle_on_top_edge":
        if level in ("low", "medium"):
            s = f"Draw a rectangle of width {fnum('w')} and height {fnum('h')}. Add a circle of radius {fnum('r')} centered on the rectangle’s top edge."
        else:
            s = "Draw a rectangle and attach a circle on the top edge (like a lollipop head on a stick)."
        return s

    if tid == "T8_two_side_by_side_rectangles":
        if level in ("low", "medium"):
            s = f"Create two identical rectangles (width {fnum('w')}, height {fnum('h')}) placed side-by-side with some gap between them."
        else:
            s = "Make two identical rectangles side-by-side with a visible gap."
        return s

    if tid == "T9_rectangular_frame":
        if level in ("low", "medium"):
            s = f"Create a rectangular frame by subtracting a smaller centered rectangle (width {fnum('w2')}, height {fnum('h2')}) from a larger rectangle (width {fnum('w')}, height {fnum('h')})."
        else:
            s = "Create a rectangular frame: a big rectangle with a smaller centered rectangular hole."
        return s

    if tid == "T10_offset_circle_hole":
        if level in ("low", "medium"):
            s = f"Draw a rectangle with width {fnum('w')} and height {fnum('h')}. Cut out a circular hole of radius {fnum('r')} offset from the center by about ({fnum('dx')}, {fnum('dy')})."
        else:
            s = "Cut a circular hole slightly off-center inside a rectangle."
        return s

    # Fallback
    return "Generate the described 2D shape program."


def make_noisy_prompt(ex: Dict[str, Any], rng: random.Random, level: str) -> str:
    """
    Prefer template-aware prompt generation using params if present.
    Otherwise apply template-agnostic perturbations to the existing prompt.
    """
    template_id = ex.get("template_id", "UNKNOWN")
    params = ex.get("params") or {}

    if params:
        base = noisy_prompt_from_params(template_id, params, rng, level)
    else:
        # Fallback: degrade existing prompt by rounding numbers and swapping synonyms
        base = ex.get("prompt", "")
        # Round numeric literals
        def repl(m):
            x = float(m.group(0))
            return _roundish(x, rng, level)
        base = re.sub(_NUM, repl, base)

    base = _replace_synonyms(base, rng, level)
    base = _maybe_add_filler(base, rng, level)
    return base.strip()


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--level", choices=["low", "medium", "high"], default="medium",
                    help="low: mostly rounding; medium: remove precision + relational; high: vague language.")
    ap.add_argument("--keep_exact", action="store_true",
                    help="If set, keep original prompt under 'prompt_exact' and overwrite 'prompt' with noisy.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(args.input)

    out_rows = []
    for ex in rows:
        ex2 = dict(ex)  # shallow copy
        noisy = make_noisy_prompt(ex2, rng, args.level)

        if args.keep_exact:
            ex2["prompt_exact"] = ex2.get("prompt")
        ex2["prompt"] = noisy
        ex2["prompt_variant"] = f"noisy_{args.level}_seed{args.seed}"

        out_rows.append(ex2)

    write_jsonl(args.output, out_rows)
    print(f"Wrote {len(out_rows)} rows -> {args.output}")
    print(f"Level={args.level}, seed={args.seed}")


if __name__ == "__main__":
    main()