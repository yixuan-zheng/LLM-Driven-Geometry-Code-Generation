#!/usr/bin/env python3
"""
eval.py — dataset split + evaluation harness for geometry DSL

1) split:
   - creates train/val/test JSONL files (default 60/10/10 out of 80)
   - reproducible via seed
   - stratifies by template_id so each split has coverage

2) eval:
   - loads a JSONL file
   - runs execute() -> verify()
   - prints:
     - overall pass rate
     - pass rate by template_id
     - pass rate by difficulty
     - pass rate by constraint type
     - avg failed constraints per example
     - avg iterations to pass is not included here (that’s for repair loop)

3) sample_failures:
   - prints a few failing examples and their failed constraint messages

Usage examples:
  python eval.py split --input geometry_dataset_v1.jsonl --out_dir data --seed 42
  python eval.py eval --input data/test.jsonl
  python eval.py sample_failures --input data/test.jsonl --k 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict, Counter
from dataclasses import asdict
from typing import Dict, Any, List, Tuple

# Local modules
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
# Split logic (stratified by template_id)
# -----------------------------

def stratified_split(
    rows: List[Dict[str, Any]],
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    assert train_n + val_n + test_n == len(rows), "split sizes must sum to dataset size"

    rng = random.Random(seed)
    by_template: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_template[r["template_id"]].append(r)

    # Shuffle within each template bucket
    for tid in by_template:
        rng.shuffle(by_template[tid])

    # Greedy round-robin assignment by template to keep coverage balanced.
    train, val, test = [], [], []
    template_ids = sorted(by_template.keys())

    # Flatten in round-robin: take one from each template until empty
    rr = []
    still = True
    idx = 0
    while still:
        still = False
        for tid in template_ids:
            if idx < len(by_template[tid]):
                rr.append(by_template[tid][idx])
                still = True
        idx += 1

    # Now cut by requested sizes
    train = rr[:train_n]
    val = rr[train_n:train_n + val_n]
    test = rr[train_n + val_n:]
    return train, val, test


# -----------------------------
# Evaluation
# -----------------------------

def evaluate(
    rows: List[Dict[str, Any]],
    H: int = 256,
    W: int = 256,
) -> Dict[str, Any]:
    rcfg = RenderConfig(H=H, W=W)

    total = len(rows)
    passed_count = 0

    pass_by_template = Counter()
    total_by_template = Counter()

    pass_by_difficulty = Counter()
    total_by_difficulty = Counter()

    # constraint-type stats
    pass_by_ctype = Counter()
    total_by_ctype = Counter()

    failed_constraints_per_ex = []  # for avg
    failures: List[Dict[str, Any]] = []

    for ex in rows:
        res = execute(ex["program_ast"], render_cfg=rcfg)
        vr = verify(res, ex["constraints"])

        total_by_template[ex["template_id"]] += 1
        total_by_difficulty[int(ex.get("difficulty", -1))] += 1

        if vr.passed:
            passed_count += 1
            pass_by_template[ex["template_id"]] += 1
            pass_by_difficulty[int(ex.get("difficulty", -1))] += 1
        else:
            failures.append({
                "id": ex.get("id"),
                "template_id": ex.get("template_id"),
                "difficulty": ex.get("difficulty"),
                "prompt": ex.get("prompt"),
                "failed_constraints": [
                    {"type": r.constraint.get("type"), "message": r.message}
                    for r in vr.failed_constraints
                ],
            })

        # Per-constraint-type aggregation
        for cr in vr.per_constraint:
            ctype = cr.constraint.get("type", "UNKNOWN")
            total_by_ctype[ctype] += 1
            if cr.passed:
                pass_by_ctype[ctype] += 1

        failed_constraints_per_ex.append(len(vr.failed_constraints))

    def pct(a, b) -> float:
        return 0.0 if b == 0 else 100.0 * a / b

    summary = {
        "total_examples": total,
        "overall_pass_rate_pct": pct(passed_count, total),
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

        "num_failures": len(failures),
        "failures_preview": failures[:10], 
    }
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Overall ===")
    print(f"Total examples: {summary['total_examples']}")
    print(f"Overall pass rate: {summary['overall_pass_rate_pct']:.2f}%")
    print(f"Avg failed constraints/example: {summary['avg_failed_constraints_per_example']:.3f}")
    print(f"Num failures: {summary['num_failures']}")

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
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="Create train/val/test splits (JSONL)")
    sp.add_argument("--input", required=True, help="Input JSONL dataset")
    sp.add_argument("--out_dir", required=True, help="Directory to write splits")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--train_n", type=int, default=60)
    sp.add_argument("--val_n", type=int, default=10)
    sp.add_argument("--test_n", type=int, default=10)

    ev = sub.add_parser("eval", help="Evaluate a JSONL split")
    ev.add_argument("--input", required=True, help="Input JSONL split file")
    ev.add_argument("--H", type=int, default=256)
    ev.add_argument("--W", type=int, default=256)
    ev.add_argument("--save_json", default=None, help="Optional path to save summary JSON")

    sf = sub.add_parser("sample_failures", help="Print k failing examples")
    sf.add_argument("--input", required=True)
    sf.add_argument("--k", type=int, default=10)
    sf.add_argument("--H", type=int, default=256)
    sf.add_argument("--W", type=int, default=256)

    args = ap.parse_args()

    if args.cmd == "split":
        rows = read_jsonl(args.input)
        if args.train_n + args.val_n + args.test_n != len(rows):
            raise ValueError("train_n + val_n + test_n must equal dataset size for now.")

        train, val, test = stratified_split(
            rows, args.train_n, args.val_n, args.test_n, seed=args.seed
        )

        write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
        write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
        write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)

        print("Wrote splits to:", args.out_dir)
        print("train:", len(train), "val:", len(val), "test:", len(test))

    elif args.cmd == "eval":
        rows = read_jsonl(args.input)
        summary = evaluate(rows, H=args.H, W=args.W)
        print_summary(summary)

        if args.save_json:
            os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print("\nSaved summary JSON to:", args.save_json)

    elif args.cmd == "sample_failures":
        rows = read_jsonl(args.input)
        rcfg = RenderConfig(H=args.H, W=args.W)

        shown = 0
        for ex in rows:
            res = execute(ex["program_ast"], render_cfg=rcfg)
            vr = verify(res, ex["constraints"])
            if not vr.passed:
                print("\n--- Failure ---")
                print("id:", ex.get("id"))
                print("template_id:", ex.get("template_id"))
                print("difficulty:", ex.get("difficulty"))
                print("prompt:", ex.get("prompt"))
                for fc in vr.failed_constraints:
                    print("  ", fc.constraint.get("type"), "|", fc.message)
                shown += 1
                if shown >= args.k:
                    break

        if shown == 0:
            print("No failures found.")

if __name__ == "__main__":
    main()