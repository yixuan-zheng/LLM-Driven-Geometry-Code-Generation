#!/usr/bin/env python3
"""
predict_eval.py — evaluate predicted programs against verifier constraints

This is the “real” scoreboard: instead of using ground-truth program_ast, it:
  prompt (+ maybe constraints) -> predictor -> pred_program_ast -> execute() -> verify()

We get:
- overall pass rate (pass@1)
- avg failed constraints per example
- pass rate by template_id / difficulty / constraint type
- Write predictions + failures to JSONL for debugging / later repair-loop

Predictor modes included:
1) oracle_regex  : parses numbers from the prompt and reconstructs AST
2) random_guess  (lower bound): guesses a plausible AST based on keyword/template cues
3) from_file     : loads precomputed predictions JSONL (for LLM outputs)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from executor import execute, RenderConfig
from verifier import verify


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# AST constructors (mirror generator.py)
# -----------------------------

def ast_rect(w: float, h: float) -> Dict[str, Any]:
    return {"type": "Rect", "w": float(w), "h": float(h)}

def ast_circle(r: float) -> Dict[str, Any]:
    return {"type": "Circle", "r": float(r)}

def ast_translate(shape: Dict[str, Any], dx: float, dy: float) -> Dict[str, Any]:
    return {"type": "Translate", "dx": float(dx), "dy": float(dy), "shape": shape}

def ast_union(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "Union", "a": a, "b": b}

def ast_diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "Difference", "a": a, "b": b}


# -----------------------------
# Prompt parsing (oracle baseline)
# -----------------------------

_NUM = r"(-?[0-9]+(?:\.[0-9]+)?)"

def _find_float(pattern: str, s: str) -> Optional[float]:
    m = re.search(pattern, s, flags=re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))

def oracle_predict_ast(prompt: str) -> Dict[str, Any]:
    """
    Parse prompt numbers and map to the right template.

    If parsing fails, raises ValueError.
    """

    p = prompt.strip()

    # T9 frame: "outer rectangle ... inner rectangular hole ..."
    if re.search(r"\bframe\b", p, flags=re.IGNORECASE):
        m = re.search(
            r"smaller\s+centered\s+rectangle\s+of\s+width\s+" + _NUM +
            r"\s+and\s+height\s+" + _NUM +
            r".*larger\s+rectangle\s+of\s+width\s+" + _NUM +
            r"\s+and\s+height\s+" + _NUM,
            p, flags=re.IGNORECASE
        )
        if not m:
            raise ValueError("Could not parse frame dims")
        w2, h2, w, h = map(float, m.groups())
        return ast_diff(ast_rect(w, h), ast_rect(w2, h2))

    # Holes / cutouts (difference)
    if re.search(r"\bhole\b|\bcutout\b|\bcut\s+out\b", p, flags=re.IGNORECASE):
        # Base rectangle
        w = _find_float(r"width\s+" + _NUM, p)
        h = _find_float(r"height\s+" + _NUM, p)
        if None in (w, h):
            raise ValueError("Could not parse base rectangle w/h")

        # Circular hole
        if re.search(r"circular|circle", p, flags=re.IGNORECASE):
            r = _find_float(r"radius\s+" + _NUM, p)
            if r is None:
                raise ValueError("Could not parse circle radius")

            # T10 wording: "offset from the center by (dx, dy)"
            m = re.search(
                r"offset\s+from\s+the\s+center\s+by\s*\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\)",
                p, flags=re.IGNORECASE
            )
            if m:
                dx, dy = float(m.group(1)), float(m.group(2))
                hole = ast_translate(ast_circle(r), dx, dy)
            else:
                # centered by default
                hole = ast_circle(r)

            base = ast_rect(w, h)
            return ast_diff(base, hole)

        # Square cutout (rect hole)
        # Look for "square cutout of size s" + optional center
        s = _find_float(r"size\s+" + _NUM, p)
        if s is None:
            # sometimes phrased as "side length"
            s = _find_float(r"side\s+length\s+" + _NUM, p)
        if s is None:
            raise ValueError("Could not parse square cutout size")

        m = re.search(r"center\s*\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\)", p, flags=re.IGNORECASE)
        if m:
            dx, dy = float(m.group(1)), float(m.group(2))
            hole = ast_translate(ast_rect(s, s), dx, dy)
        else:
            # If prompt says "top edge at center", our dataset convention was dy = -s/2
            if re.search(r"top\s+edge.*center", p, flags=re.IGNORECASE):
                hole = ast_translate(ast_rect(s, s), 0.0, -s / 2.0)
            else:
                hole = ast_rect(s, s)

        base = ast_rect(w, h)
        return ast_diff(base, hole)

    # Union: rectangle + circle on top edge
    if re.search(r"\badd a circle\b|\bunion\b", p, flags=re.IGNORECASE):
        w = _find_float(r"width\s+" + _NUM, p)
        h = _find_float(r"height\s+" + _NUM, p)
        r = _find_float(r"radius\s+" + _NUM, p)
        if None in (w, h, r):
            raise ValueError("Could not parse union rect/circle")
        base = ast_rect(w, h)
        # Our T7 convention: circle centered on rectangle top edge => y = h/2
        circle = ast_translate(ast_circle(r), 0.0, h / 2.0)
        return ast_union(base, circle)

    # T8: Two side-by-side rectangles — prompt has w,h but not d; choose a valid d from w.
    if re.search(r"two\s+identical\s+rectangles|side\s+by\s+side", p, flags=re.IGNORECASE):
        w = _find_float(r"width\s+" + _NUM, p)
        h = _find_float(r"height\s+" + _NUM, p)
        if None in (w, h):
            raise ValueError("Could not parse w/h for two rectangles")
        d = 0.35 * w  # safe choice within generator constraints
        a = ast_translate(ast_rect(w, h), -d, 0.0)
        b = ast_translate(ast_rect(w, h),  d, 0.0)
        return ast_union(a, b)

    # T3: Translated rectangle — generator says "shifted by (dx, dy)"
    if re.search(r"\boffset\b|\btranslate\b|\bshifted\b", p, flags=re.IGNORECASE):
        w = _find_float(r"width\s+" + _NUM, p)
        h = _find_float(r"height\s+" + _NUM, p)

        m = re.search(
            r"shifted\s+by\s*\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\)",
            p, flags=re.IGNORECASE
        )
        if m:
            dx, dy = float(m.group(1)), float(m.group(2))
            return ast_translate(ast_rect(w, h), dx, dy)

        # fallback: any (dx, dy) tuple if wording differs
        m2 = re.search(r"\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\)", p)
        if None in (w, h) or not m2:
            raise ValueError("Could not parse translated rectangle")
        dx, dy = float(m2.group(1)), float(m2.group(2))
        return ast_translate(ast_rect(w, h), dx, dy)

    # Single rectangle / circle
    if re.search(r"\bcircle\b", p, flags=re.IGNORECASE):
        r = _find_float(r"radius\s+" + _NUM, p)
        if r is None:
            raise ValueError("Could not parse circle radius")
        return ast_circle(r)

    if re.search(r"\brectangle\b|\brect\b", p, flags=re.IGNORECASE):
        w = _find_float(r"width\s+" + _NUM, p)
        h = _find_float(r"height\s+" + _NUM, p)
        if None in (w, h):
            raise ValueError("Could not parse rectangle w/h")
        return ast_rect(w, h)

    raise ValueError("Could not determine template from prompt")


# -----------------------------
# Random guess baseline (lower bound)
# -----------------------------

def random_guess_ast(prompt: str, seed: int = 0) -> Dict[str, Any]:
    """
    A deliberately weak baseline: tries to infer template from keywords,
    otherwise falls back to a random primitive.
    """
    rng = random.Random(seed + hash(prompt) % (2**31 - 1))
    p = prompt.lower()

    def rnd(a, b):
        return rng.uniform(a, b)

    if "frame" in p:
        ow, oh = rnd(6, 18), rnd(6, 18)
        iw, ih = rnd(2, ow - 2), rnd(2, oh - 2)
        return ast_diff(ast_rect(ow, oh), ast_rect(iw, ih))

    if "hole" in p or "cutout" in p:
        w, h = rnd(6, 18), rnd(6, 18)
        if "circle" in p:
            r = rnd(1, min(w, h) / 3)
            return ast_diff(ast_rect(w, h), ast_circle(r))
        else:
            s = rnd(1, min(w, h) / 2.5)
            return ast_diff(ast_rect(w, h), ast_rect(s, s))

    if "two" in p and ("rectangle" in p or "rect" in p):
        w, h = rnd(2, 8), rnd(2, 8)
        gap = rnd(0.5, 3.0)
        a = ast_translate(ast_rect(w, h), -(gap / 2 + w / 2), 0.0)
        b = ast_translate(ast_rect(w, h), +(gap / 2 + w / 2), 0.0)
        return ast_union(a, b)

    if "add a circle" in p or "union" in p:
        w, h = rnd(6, 18), rnd(6, 18)
        r = rnd(1, min(w, h) / 2.5)
        return ast_union(ast_rect(w, h), ast_translate(ast_circle(r), 0.0, h / 2))

    if "circle" in p:
        return ast_circle(rnd(1, 6))

    # default: rectangle
    return ast_rect(rnd(4, 12), rnd(2, 8))


# -----------------------------
# Prediction loading (from_file)
# -----------------------------

def load_predictions(pred_path: str) -> Dict[Any, Dict[str, Any]]:
    """
    Load a JSONL file where each row has:
      - id
      - pred_program_ast

    Returns a dict: id -> pred_program_ast
    """
    preds = {}
    for row in read_jsonl(pred_path):
        if "id" not in row or "pred_program_ast" not in row:
            raise ValueError("predictions file must contain 'id' and 'pred_program_ast'")
        preds[row["id"]] = row["pred_program_ast"]
    return preds


# -----------------------------
# Evaluation core
# -----------------------------

def evaluate_predictions(
    rows: List[Dict[str, Any]],
    mode: str,
    H: int = 256,
    W: int = 256,
    seed: int = 42,
    pred_file: Optional[str] = None,
    save_pred_jsonl: Optional[str] = None,
    save_fail_jsonl: Optional[str] = None,
) -> Dict[str, Any]:
    rcfg = RenderConfig(H=H, W=W)

    total = len(rows)
    passed = 0
    failed_constraints_per_ex: List[int] = []

    pass_by_template = Counter()
    total_by_template = Counter()

    pass_by_difficulty = Counter()
    total_by_difficulty = Counter()

    pass_by_ctype = Counter()
    total_by_ctype = Counter()

    pred_dump = []
    fail_dump = []

    preds_by_id = load_predictions(pred_file) if (mode == "from_file") else None

    for ex in rows:
        ex_id = ex.get("id")
        template_id = ex.get("template_id", "UNKNOWN")
        difficulty = int(ex.get("difficulty", -1))
        prompt = ex["prompt"]
        constraints = ex["constraints"]

        # ---- predict AST ----
        try:
            if mode == "oracle_regex":
                pred_ast = oracle_predict_ast(prompt)
            elif mode == "random_guess":
                pred_ast = random_guess_ast(prompt, seed=seed)
            elif mode == "from_file":
                if preds_by_id is None or ex_id not in preds_by_id:
                    raise ValueError(f"Missing prediction for id={ex_id}")
                pred_ast = preds_by_id[ex_id]
            else:
                raise ValueError(f"Unknown mode: {mode}")
        except Exception as e:
            # If prediction fails, treat as full failure without executing.
            pred_ast = None
            vr_passed = False
            per_constraint = []
            failed_constraints = [{"type": "PREDICTOR_ERROR", "message": str(e)}]
        else:
            # ---- execute + verify ----
            res = execute(pred_ast, render_cfg=rcfg)
            vr = verify(res, constraints)
            vr_passed = vr.passed
            per_constraint = vr.per_constraint
            failed_constraints = [
                {"type": r.constraint.get("type"), "message": r.message}
                for r in vr.failed_constraints
            ]

        total_by_template[template_id] += 1
        total_by_difficulty[difficulty] += 1

        if vr_passed:
            passed += 1
            pass_by_template[template_id] += 1
            pass_by_difficulty[difficulty] += 1

        # constraint-type stats
        if pred_ast is not None:
            for cr in per_constraint:
                ctype = cr.constraint.get("type", "UNKNOWN")
                total_by_ctype[ctype] += 1
                if cr.passed:
                    pass_by_ctype[ctype] += 1
            failed_constraints_per_ex.append(len(failed_constraints))
        else:
            # predictor error counts as all constraints failed
            failed_constraints_per_ex.append(len(constraints) + 1)

        if save_pred_jsonl:
            pred_dump.append({
                "id": ex_id,
                "template_id": template_id,
                "difficulty": difficulty,
                "prompt": prompt,
                "pred_program_ast": pred_ast,
            })

        if save_fail_jsonl and (not vr_passed):
            fail_dump.append({
                "id": ex_id,
                "template_id": template_id,
                "difficulty": difficulty,
                "prompt": prompt,
                "pred_program_ast": pred_ast,
                "failed_constraints": failed_constraints,
            })

    def pct(a, b) -> float:
        return 0.0 if b == 0 else 100.0 * a / b

    summary = {
        "mode": mode,
        "total_examples": total,
        "pass_rate_pct": pct(passed, total),
        "avg_failed_constraints_per_example": sum(failed_constraints_per_ex) / max(total, 1),

        "pass_rate_by_template_pct": {
            tid: pct(pass_by_template[tid], total_by_template[tid])
            for tid in sorted(total_by_template.keys())
        },
        "pass_rate_by_difficulty_pct": {
            str(k): pct(pass_by_difficulty[k], total_by_difficulty[k])
            for k in sorted(total_by_difficulty.keys())
        },
        "pass_rate_by_constraint_type_pct": {
            ctype: pct(pass_by_ctype[ctype], total_by_ctype[ctype])
            for ctype in sorted(total_by_ctype.keys())
        },
    }

    if save_pred_jsonl:
        write_jsonl(save_pred_jsonl, pred_dump)
    if save_fail_jsonl:
        write_jsonl(save_fail_jsonl, fail_dump)

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Predict Eval ===")
    print("Mode:", summary["mode"])
    print(f"Total examples: {summary['total_examples']}")
    print(f"Pass@1: {summary['pass_rate_pct']:.2f}%")
    print(f"Avg failed constraints/example: {summary['avg_failed_constraints_per_example']:.3f}")

    print("\n=== Pass rate by template_id ===")
    for tid, p in summary["pass_rate_by_template_pct"].items():
        print(f"{tid:40s} {p:6.2f}%")

    print("\n=== Pass rate by difficulty ===")
    for d, p in summary["pass_rate_by_difficulty_pct"].items():
        print(f"difficulty={d:>2s}  {p:6.2f}%")

    print("\n=== Pass rate by constraint type ===")
    for ctype, p in summary["pass_rate_by_constraint_type_pct"].items():
        print(f"{ctype:35s} {p:6.2f}%")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL split (e.g., data/test.jsonl)")
    ap.add_argument("--mode", choices=["oracle_regex", "random_guess", "from_file"], default="oracle_regex")
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--pred_file", default=None,
                    help="Predictions JSONL (required for mode=from_file). Each row: {id, pred_program_ast}.")

    ap.add_argument("--save_pred_jsonl", default=None,
                    help="Optional: write per-example predictions JSONL (id, prompt, pred_program_ast).")
    ap.add_argument("--save_fail_jsonl", default=None,
                    help="Optional: write failing examples JSONL with failed constraint messages.")

    ap.add_argument("--save_summary_json", default=None,
                    help="Optional: save summary as JSON.")

    args = ap.parse_args()

    rows = read_jsonl(args.input)
    summary = evaluate_predictions(
        rows=rows,
        mode=args.mode,
        H=args.H,
        W=args.W,
        seed=args.seed,
        pred_file=args.pred_file,
        save_pred_jsonl=args.save_pred_jsonl,
        save_fail_jsonl=args.save_fail_jsonl,
    )

    print_summary(summary)

    if args.save_summary_json:
        os.makedirs(os.path.dirname(args.save_summary_json), exist_ok=True)
        with open(args.save_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("\nSaved summary JSON to:", args.save_summary_json)


if __name__ == "__main__":
    main()
