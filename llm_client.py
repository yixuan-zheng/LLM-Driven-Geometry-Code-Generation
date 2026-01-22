# llm_client.py
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import copy
from repair_utils import noisy_oracle_then_break, heuristic_repair

# -----------------------------
# AST schema helpers
# -----------------------------

ALLOWED_TYPES = {"Rect", "Circle", "Translate", "Union", "Difference"}

def validate_ast(ast: Dict[str, Any]) -> Tuple[bool, str]:
    """Lightweight schema validation. Keep it permissive for iteration."""
    if not isinstance(ast, dict):
        return False, "AST is not a dict"
    t = ast.get("type")
    if t not in ALLOWED_TYPES:
        return False, f"Unknown type: {t}"

    if t == "Rect":
        if "w" not in ast or "h" not in ast:
            return False, "Rect missing w/h"
        return True, ""

    if t == "Circle":
        if "r" not in ast:
            return False, "Circle missing r"
        return True, ""

    if t == "Translate":
        if "dx" not in ast or "dy" not in ast or "shape" not in ast:
            return False, "Translate missing dx/dy/shape"
        ok, msg = validate_ast(ast["shape"])
        return ok, msg

    if t in ("Union", "Difference"):
        if "a" not in ast or "b" not in ast:
            return False, f"{t} missing a/b"
        ok, msg = validate_ast(ast["a"])
        if not ok:
            return ok, msg
        ok, msg = validate_ast(ast["b"])
        return ok, msg

    return False, "Unhandled type"


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from model output.
    Handles:
    - pure JSON
    - JSON fenced in ```json
    - JSON embedded in text
    """
    text = text.strip()

    # If fenced
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        return json.loads(candidate)

    # If already JSON
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Otherwise: find first {...} block by greedy brace matching
    # This is not a full parser but works for typical model outputs.
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in output")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                return json.loads(candidate)
    raise ValueError("Unbalanced braces; could not extract JSON")


# -----------------------------
# Prompts
# -----------------------------

SYSTEM_STYLE = (
    "You generate parametric 2D CAD programs as JSON ASTs.\n"
    "Return ONLY valid JSON, no commentary.\n"
    "Schema:\n"
    '{ "type": "Rect", "w": float, "h": float }\n'
    '{ "type": "Circle", "r": float }\n'
    '{ "type": "Translate", "dx": float, "dy": float, "shape": <AST> }\n'
    '{ "type": "Union", "a": <AST>, "b": <AST> }\n'
    '{ "type": "Difference", "a": <AST>, "b": <AST> }\n'
)

def build_propose_prompt(user_prompt: str) -> str:
    return (
        SYSTEM_STYLE
        + "\nTask:\n"
        + user_prompt.strip()
        + "\n\nReturn ONLY the JSON AST.\n"
    )

def build_repair_prompt(
    user_prompt: str,
    current_ast: Dict[str, Any],
    feedback: str,
    rules: str
) -> str:
    return (
        SYSTEM_STYLE
        + "\nTask:\n"
        + user_prompt.strip()
        + "\n\nCurrent AST (JSON):\n"
        + json.dumps(current_ast)
        + "\n\nVerifier failures:\n"
        + feedback.strip()
        + "\n\nRepair rules (apply these FIRST if relevant):\n"
        + rules.strip()
        + "\n\nReturn ONLY the revised JSON AST.\n"
    )

def build_repair_rules(failed_constraints: Any) -> str:
    """
    Turn verifier failures into explicit edit instructions.
    This directly targets failure modes we see in traces (e.g., has_hole=False).
    """
    types = [fc.constraint.get("type", "") for fc in failed_constraints]
    msgs = " | ".join(fc.message for fc in failed_constraints)

    rules: list[str] = []

    # 1) Structural: must introduce a hole
    if "has_hole" in types and ("has_hole=False" in msgs) and ("target=True" in msgs):
        rules.append(
            "- If has_hole failed (has_hole=False, target=True): the AST MUST be "
            '{"type":"Difference","a":<base_shape>,"b":<hole_shape>}. '
            "The hole must be in 'b' (subtracted)."
        )

    # 2) Cutout geometry and placement
    if "cutout_size" in types:
        rules.append(
            "- If cutout_size failed: adjust the HOLE size (Rect w/h) so its bbox width matches target. "
            "If hole is a square, set h=w."
        )
    if "cutout_radius" in types:
        rules.append(
            "- If cutout_radius failed: adjust the HOLE Circle radius r to match the target radius."
        )
    if "centered" in types:
        rules.append(
            "- If centered failed for a hole: ensure the hole is centered in the base. "
            "Use Translate(dx=0, dy=0) around the hole if needed."
        )
    if "centered_x" in types:
        rules.append(
            "- If centered_x failed for a hole: set the hole's Translate.dx = 0 (keep dy as needed)."
        )
    if "top_edge_y_equals" in types:
        rules.append(
            "- If top_edge_y_equals failed: for a translated square hole with side s, enforce "
            "dy = target_top_y - s/2."
        )
    if "cutout_inside_base" in types:
        rules.append(
            "- If cutout_inside_base failed: move the hole inward by reducing |dx| and/or |dy| "
            "so the hole bbox lies strictly inside the base bbox."
        )

    # 3) Connectivity: merge components
    if "num_components" in types:
        rules.append(
            "- If num_components failed (num_components=2, target=1): make shapes CONNECTED. "
            "If two rectangles are side-by-side, set the second Translate.dx so they touch/overlap "
            "(e.g., dx ≈ rect_w or slightly less)."
        )

    # 4) Frame inner bbox
    if "inner_bbox_width" in types:
        rules.append(
            "- If inner_bbox_width failed: in a frame Difference(outer, inner), adjust the INNER rect width to match target."
        )
    if "inner_bbox_height" in types:
        rules.append(
            "- If inner_bbox_height failed: adjust the INNER rect height to match target."
        )

    if not rules:
        rules.append("- Make the smallest possible change to satisfy the verifier failures.")

    return "\n".join(rules)


# -----------------------------
# Client
# -----------------------------

@dataclass
class LLMConfig:
    backend: str = "stub"  # stub | noisy_oracle | openai | local_cmd
    model: str = "gpt-4.1-mini" 
    temperature: float = 0.0
    max_tokens: int = 800
    local_cmd: Optional[str] = None  


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def propose_ast(
        self,
        prompt: str,
        ex: Optional[Dict[str, Any]] = None,
        seed: int = 0
    ) -> Dict[str, Any]:
        if self.cfg.backend == "noisy_oracle":
            if ex is None:
                raise ValueError("noisy_oracle backend requires full example")
            return noisy_oracle_then_break(ex, seed=seed)

        full = build_propose_prompt(prompt)
        text = self._complete(full)
        ast = extract_json_object(text)
        ok, msg = validate_ast(ast)
        if not ok:
            raise ValueError(f"Invalid AST from model: {msg}\nRaw={text[:400]}")
        return ast

    def repair_ast(
        self,
        prompt: str,
        current_ast: Dict[str, Any],
        failed_constraints: Any
    ) -> Dict[str, Any]:

        if self.cfg.backend == "noisy_oracle":
            return heuristic_repair(current_ast, failed_constraints)

        feedback = "\n".join(
            f"- {fc.constraint.get('type')}: {fc.message}"
            for fc in failed_constraints
        )

        rules = build_repair_rules(failed_constraints)

        full = build_repair_prompt(prompt, current_ast, feedback, rules)
        text = self._complete(full)
        ast = extract_json_object(text)
        ok, msg = validate_ast(ast)
        if not ok:
            raise ValueError(f"Invalid repaired AST: {msg}\nRaw={text[:400]}")
        return ast

    # -------- backends --------

    def _complete(self, full_prompt: str) -> str:
        if self.cfg.backend == "stub":
            return json.dumps({"type": "Rect", "w": 10.0, "h": 5.0})

        # NEW BACKEND
        if self.cfg.backend == "noisy_oracle":
            # _complete is not actually used for logic here
            # We just return a placeholder JSON so extract_json_object works
            return json.dumps({"_noisy_oracle": True})

        if self.cfg.backend == "local_cmd":
            if not self.cfg.local_cmd:
                raise ValueError("local_cmd backend requires cfg.local_cmd")
            return self._run_local_cmd(full_prompt)

        if self.cfg.backend == "openai":
            return self._run_openai(full_prompt)

        raise ValueError(f"Unknown backend: {self.cfg.backend}")

    def _run_local_cmd(self, full_prompt: str) -> str:
        # You provide a command that reads prompt from stdin and prints output to stdout
        cmd = self.cfg.local_cmd
        proc = subprocess.run(
            cmd,
            input=full_prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"local_cmd failed: {proc.stderr.decode('utf-8', errors='ignore')}")
        return proc.stdout.decode("utf-8", errors="ignore")

    def _run_openai(self, full_prompt: str) -> str:
        """
        Uses OpenAI Responses API and forces valid JSON via json_object.
        We validate/repair AST shape with our own verifier downstream.
        """
        import os
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        resp = client.responses.create(
            model=self.cfg.model,
            input=full_prompt,
            text={
                "format": {
                    "type": "json_object"
                }
            },
            temperature=self.cfg.temperature,
            max_output_tokens=self.cfg.max_tokens,
        )

        out = getattr(resp, "output_text", None)

        if out is None:
            chunks = []
            for item in resp.output:
                if getattr(item, "type", None) == "message":
                    for c in item.content:
                        if getattr(c, "type", None) == "output_text":
                            chunks.append(c.text)
            out = "".join(chunks)

        if not out or not out.strip():
            raise RuntimeError("OpenAI returned empty output")

        return out