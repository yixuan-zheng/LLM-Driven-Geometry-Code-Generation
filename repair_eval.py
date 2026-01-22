#!/usr/bin/env python3
"""
repair_eval.py — iterative repair evaluation.

Loop:
  prompt -> propose AST -> execute -> verify
  if fail:
    format verifier failures -> "feedback"
    repair AST (stub / heuristic / model)
  repeat up to K iterations

Metrics:
- pass@1, pass@K
- avg iterations-to-pass (for those that pass)
- avg failed constraints per attempt
- breakdown by template_id, difficulty, constraint type

Usage examples:
  python repair_eval.py --input data_noisy/test.jsonl --K 3 --init random_guess --repair heuristic
  python repair_eval.py --input data_noisy/test.jsonl --K 5 --init noisy_oracle_then_break --repair heuristic --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from executor import execute, RenderConfig
from verifier import verify
import hashlib
from llm_client import LLMClient, LLMConfig
from repair_utils import stable_int, noisy_oracle_then_break, heuristic_repair

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
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# AST utilities (mirror generator DSL)
# -----------------------------

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


# -----------------------------
# Baseline initial predictors
# -----------------------------

def random_guess_ast(prompt: str, seed: int = 0) -> Dict[str, Any]:
    """
    Very weak baseline: ignores constraints, guesses a plausible AST.
    Works for testing harness.
    """
    rng = random.Random(seed + stable_int(prompt))
    p = prompt.lower()

    def rnd(a, b):
        return rng.uniform(a, b)

    def rect():
        return {"type": "Rect", "w": rnd(4, 14), "h": rnd(2, 10)}

    def circle():
        return {"type": "Circle", "r": rnd(1, 6)}

    def translate(shape):
        return {"type": "Translate", "dx": rnd(-4, 4), "dy": rnd(-4, 4), "shape": shape}

    if "frame" in p:
        outer = rect()
        inner = rect()
        # force inner smaller-ish
        inner["w"] = max(1.0, 0.6 * outer["w"])
        inner["h"] = max(1.0, 0.6 * outer["h"])
        return {"type": "Difference", "a": outer, "b": inner}

    if "hole" in p or "cut out" in p or "cutout" in p:
        base = rect()
        if "circle" in p:
            hole = circle()
            if "off-center" in p or "offset" in p:
                hole = translate(hole)
        else:
            s = rnd(1, min(base["w"], base["h"]) / 2.0)
            hole = {"type": "Rect", "w": s, "h": s}
            if "off" in p or "midline" in p:
                hole = {"type": "Translate", "dx": 0.0, "dy": -s / 2.0, "shape": hole}
        return {"type": "Difference", "a": base, "b": hole}

    if "two" in p and "rectangle" in p:
        w, h = rnd(2, 7), rnd(2, 7)
        d = rnd(0.8, 3.0)
        a = {"type": "Translate", "dx": -d, "dy": 0.0, "shape": {"type": "Rect", "w": w, "h": h}}
        b = {"type": "Translate", "dx":  d, "dy": 0.0, "shape": {"type": "Rect", "w": w, "h": h}}
        return {"type": "Union", "a": a, "b": b}

    if "add a circle" in p or "attach a circle" in p or "top edge" in p:
        base = rect()
        r = rnd(1, 5)
        circ = {"type": "Translate", "dx": 0.0, "dy": base["h"] / 2.0, "shape": {"type": "Circle", "r": r}}
        return {"type": "Union", "a": base, "b": circ}

    if "circle" in p:
        return circle()

    return rect()

# -----------------------------
# Feedback formatting
# -----------------------------

def format_feedback(failed: List[Any]) -> str:
    """
    Turn failed constraint results into short, actionable text.

    We keep it compact because later this will be fed to an LLM.
    """
    lines = []
    for fc in failed:
        ctype = fc.constraint.get("type", "UNKNOWN")
        msg = fc.message
        lines.append(f"- {ctype}: {msg}")
    return "\n".join(lines)


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

def find_frame_outer_inner(ast: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    If ast is Difference(outer_rect, inner_rect or Translate(inner_rect)), return (outer_rect, inner_rect_node).
    """
    if ast.get("type") != "Difference":
        return None, None

    outer = ast.get("a")
    inner = ast.get("b")

    # unwrap translate if needed
    if inner and inner.get("type") == "Translate":
        inner = inner.get("shape")

    if outer and outer.get("type") == "Rect" and inner and inner.get("type") == "Rect":
        return outer, inner

    return None, None


# -----------------------------
# Evaluation loop
# -----------------------------

def run_repair_eval(
    rows: List[Dict[str, Any]],
    K: int,
    init_mode: str,
    repair_mode: str,
    seed: int,
    H: int,
    W: int,
    save_trace_jsonl: Optional[str] = None,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    rcfg = RenderConfig(H=H, W=W)

    total = len(rows)
    passed_at_1 = 0
    passed_at_k = 0
    iters_to_pass: List[int] = []
    avg_failed_constraints_per_attempt: List[float] = []

    pass_by_template_at_k = Counter()
    total_by_template = Counter()

    pass_by_difficulty_at_k = Counter()
    total_by_difficulty = Counter()

    # constraint-type pass stats across final attempt for those that pass
    # (keeps it simple for v1)
    pass_by_ctype_final = Counter()
    total_by_ctype_final = Counter()

    traces = []

    for ex in rows:
        ex_id = ex.get("id")
        template_id = ex.get("template_id", "UNKNOWN")
        difficulty = int(ex.get("difficulty", -1))
        prompt = ex["prompt"]
        constraints = ex["constraints"]

        total_by_template[template_id] += 1
        total_by_difficulty[difficulty] += 1

        # ---- init prediction ----
        if init_mode == "random_guess":
            cur_ast = random_guess_ast(prompt, seed=seed)
        elif init_mode == "noisy_oracle_then_break":
            cur_ast = noisy_oracle_then_break(ex, seed=seed)
        elif init_mode == "llm":
            if llm is None:
                raise ValueError("init_mode=llm requires llm client")
            cur_ast = llm.propose_ast(prompt, ex=ex, seed=seed)
        else:
            raise ValueError(f"Unknown init mode: {init_mode}")

        ex_trace = {
            "id": ex_id,
            "template_id": template_id,
            "difficulty": difficulty,
            "prompt": prompt,
            "attempts": []
        }

        success = False
        failed_counts = []

        for t in range(1, K + 1):
            res = execute(cur_ast, render_cfg=rcfg)
            vr = verify(res, constraints)

            failed_counts.append(len(vr.failed_constraints))

            ex_trace["attempts"].append({
                "t": t,
                "program_ast": cur_ast,
                "passed": vr.passed,
                "failed_constraints": [
                    {"type": r.constraint.get("type"), "message": r.message}
                    for r in vr.failed_constraints
                ],
            })

            if vr.passed:
                success = True
                if t == 1:
                    passed_at_1 += 1
                passed_at_k += 1
                iters_to_pass.append(t)
                pass_by_template_at_k[template_id] += 1
                pass_by_difficulty_at_k[difficulty] += 1

                # record final constraint pass info
                for cr in vr.per_constraint:
                    ctype = cr.constraint.get("type", "UNKNOWN")
                    total_by_ctype_final[ctype] += 1
                    if cr.passed:
                        pass_by_ctype_final[ctype] += 1
                break

            # ---- repair ----
            feedback = format_feedback(vr.failed_constraints)

            if repair_mode == "none":
                break
            elif repair_mode == "heuristic":
                cur_ast = heuristic_repair(cur_ast, vr.failed_constraints)
            elif repair_mode == "llm":
                if llm is None:
                    raise ValueError("repair_mode=llm requires llm client")
                cur_ast = llm.repair_ast(prompt, cur_ast, vr.failed_constraints)
            else:
                raise ValueError(f"Unknown repair mode: {repair_mode}")

        if not success:
            # for failures, still count final constraint-type totals to inspect
            pass

        avg_failed_constraints_per_attempt.append(sum(failed_counts) / max(len(failed_counts), 1))
        traces.append(ex_trace)

    def pct(a, b) -> float:
        return 0.0 if b == 0 else 100.0 * a / b

    summary = {
        "total_examples": total,
        "K": K,
        "init_mode": init_mode,
        "repair_mode": repair_mode,
        "pass_at_1_pct": pct(passed_at_1, total),
        "pass_at_k_pct": pct(passed_at_k, total),
        "avg_iters_to_pass": (sum(iters_to_pass) / len(iters_to_pass)) if iters_to_pass else None,
        "avg_failed_constraints_per_attempt": sum(avg_failed_constraints_per_attempt) / max(total, 1),

        "pass_at_k_by_template_pct": {
            tid: pct(pass_by_template_at_k[tid], total_by_template[tid])
            for tid in sorted(total_by_template.keys())
        },
        "pass_at_k_by_difficulty_pct": {
            str(d): pct(pass_by_difficulty_at_k[d], total_by_difficulty[d])
            for d in sorted(total_by_difficulty.keys())
        },
        "final_constraint_pass_rate_pct": {
            c: pct(pass_by_ctype_final[c], total_by_ctype_final[c])
            for c in sorted(total_by_ctype_final.keys())
        }
    }

    if save_trace_jsonl:
        write_jsonl(save_trace_jsonl, traces)

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Repair Eval ===")
    print(f"Init: {summary['init_mode']} | Repair: {summary['repair_mode']} | K={summary['K']}")
    print(f"Total examples: {summary['total_examples']}")
    print(f"Pass@1: {summary['pass_at_1_pct']:.2f}%")
    print(f"Pass@{summary['K']}: {summary['pass_at_k_pct']:.2f}%")
    print(f"Avg iters-to-pass (if passed): {summary['avg_iters_to_pass']}")
    print(f"Avg failed constraints/attempt: {summary['avg_failed_constraints_per_attempt']:.3f}")

    print("\n=== Pass@K by template_id ===")
    for tid, p in summary["pass_at_k_by_template_pct"].items():
        print(f"{tid:40s} {p:6.2f}%")

    print("\n=== Pass@K by difficulty ===")
    for d, p in summary["pass_at_k_by_difficulty_pct"].items():
        print(f"difficulty={d:>2s}  {p:6.2f}%")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL split, e.g. data_noisy/test.jsonl")
    ap.add_argument("--K", type=int, default=3, help="Max attempts per example")
    ap.add_argument(
        "--init",
        choices=["random_guess", "noisy_oracle_then_break", "llm"],
        default="random_guess"
    )
    ap.add_argument(
        "--repair",
        choices=["none", "heuristic", "llm"],
        default="heuristic"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--save_trace_jsonl", default=None, help="Optional: save per-example attempt traces")

    # LLM config (stub by default)
    ap.add_argument(
        "--llm_backend",
        choices=["stub", "noisy_oracle", "openai", "local_cmd"],
        default="stub"
    )
    ap.add_argument("--llm_model", default="gpt-4.1-mini")
    ap.add_argument("--llm_temperature", type=float, default=0.0)
    ap.add_argument("--llm_max_tokens", type=int, default=800)
    ap.add_argument("--llm_local_cmd", default=None)

    args = ap.parse_args()

    rows = read_jsonl(args.input)

    llm = LLMClient(
        LLMConfig(
            backend=args.llm_backend,
            model=args.llm_model,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            local_cmd=args.llm_local_cmd,
        )
    )

    summary = run_repair_eval(
        rows=rows,
        K=args.K,
        init_mode=args.init,
        repair_mode=args.repair,
        seed=args.seed,
        H=args.H,
        W=args.W,
        save_trace_jsonl=args.save_trace_jsonl,
        llm=llm,
    )

    print_summary(summary)


if __name__ == "__main__":
    main()