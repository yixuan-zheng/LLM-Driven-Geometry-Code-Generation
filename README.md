## Project: LLM-Driven Geometry Code Generation with Executable Verification

### Overview

This project explores **LLM-based program synthesis for parametric 2D geometry**, where generated code is *executed, verified, and iteratively repaired* using structured, checkable feedback. Rather than relying on prompt engineering or resampling alone, the system closes the loop by treating geometry generation as an **executable and verifiable task**.

Given a natural-language prompt describing a 2D shape, the system:

1. Generates a geometry program (AST) using an LLM
2. Executes the program to produce a renderable shape
3. Verifies the output against deterministic geometric constraints
4. Uses verifier feedback to iteratively refine the program

This verifier-guided repair loop substantially improves correctness on compositional geometry tasks.

## 1) Executable Geometry Scope

### 1.1 Geometry representation

* Shapes are represented as **symbolic geometry programs** (ASTs) composed of primitives and boolean operations.
* Programs are executed into:

  * a rendered representation (binary mask)
  * structured geometric metadata (bounding boxes, components, holes, etc.)

Metadata is returned in unit-space coordinates (fields suffixed with _units in the implementation). Execution uses an automatically padded render bounding box (via a fixed rendering margin) when rasterizing shapes; this margin does not affect reported geometric measurements. This design ensures that every generated program can be **checked deterministically**.

### 1.2 Primitives (v1)

We intentionally keep the primitive set minimal and fully verifiable:

* `Rect(w, h)` — axis-aligned rectangle
* `Circle(r)` — circle

(All primitives are centered at the origin by default.)

### 1.3 Transformations

* `Translate(dx, dy, shape)` — shifts the **center** of a shape

(Optional extensions such as rotation or scaling are omitted in v1 to keep verification simple and robust.)

### 1.4 Boolean operations

* `Union(a, b)` — union of two shapes
* `Difference(a, b)` — subtracts shape `b` from `a` (supports holes, cutouts, frames)

Intersection is intentionally omitted in v1.

### 1.5 Coordinate system assumptions

* The origin `(0, 0)` denotes the **center** of each primitive.
* All transformations operate on shape centers.
* Units are arbitrary but consistent.
* All geometric comparisons use tolerances.

## 2) Verifiable Properties

Prompts are restricted to properties that can be evaluated deterministically after execution. Most constraints are strictly enforced in v1. A small subset of constraints (e.g., cutout_shape) are treated as advisory and reported diagnostically rather than causing hard failure.

### 2.1 Structural constraints

* Number of connected components
* Presence of a hole / cutout
* Correct boolean structure (e.g., `Difference` vs `Union`)

### 2.2 Dimensional constraints (with tolerance)

* Absolute size: width, height, radius
* Relative size: larger/smaller relationships
* Ratio constraints: e.g., height ≈ 2/3 width

### 2.3 Positional / relational constraints

* Centered alignment
* Containment (shape B inside shape A)
* Edge relations (touching top edge, aligned with left edge)

**Explicit non-goals (v1):**

* Free-form curves or splines
* Fine-grained aesthetics
* Pixel-perfect matching

## 3) Success Criteria

### 3.1 Program validity

A generated program is considered **valid** if:

* It parses correctly
* It executes without runtime errors
* It produces a renderable shape

### 3.2 Semantic correctness

A program is **correct** if:

* It is valid, **and**
* The executed output satisfies **all required constraints** derived from the prompt

We report:

* Pass@1 (direct generation)
* Pass@K (after up to K repair iterations)
* Average repair iterations to success

## 4) Prompt Scope

Prompts use constrained, compositional English to ensure checkability.

Examples:

* “Create a rectangle of width W and height H.”
* “Cut out a centered square hole of size S from the rectangle.”
* “Place a circle inside the rectangle, centered, with radius R.”
* “Make the rectangle height about 2/3 its width.”

Ambiguity is reduced by:

* Restricting to a fixed vocabulary of relations
* Using approximate language (“about”) only with explicit tolerances

## 5) Execution and Verification Interface

### 5.1 Program format (conceptual)

```text
Program := Expr
Expr := Rect(w, h)
     | Circle(r)
     | Translate(dx, dy, Expr)
     | Union(Expr, Expr)
     | Difference(Expr, Expr)
```

### 5.2 Executor output

```text
execute(program) -> {
  mask: 2D binary array,
  metadata: {
    bbox: (xmin, ymin, xmax, ymax),
    area: float,
    centroid: (x, y),
    num_components: int,
    has_hole: bool
  }
}
```

Field names shown here are conceptual; see executor.py for exact metadata keys (e.g., _units suffixes) and the ExecResult wrapper.

### 5.3 Constraint checker

```text
check(output, constraints) -> {
  passed: bool,
  failed_constraints: [...],
  diagnostics: { per-constraint measurements }
}
```

In the implementation, the verifier returns a structured VerifyResult object containing:
- passed: overall boolean
- per_constraint results (including measurements and pass/fail)
- failed_constraints: a filtered list of failed hard constraints
- diagnostics: auxiliary measurements for repair feedback

## 6) Verifier-Guided Repair Loop

Rather than resampling blindly, the system performs **targeted repair** based on executor feedback.

```text
y0 = LLM.propose_ast(prompt)

for t in 0..K:
  out = execute(yt)
  result = check(out, constraints_from(prompt))
  if result.passed:
    break
  feedback = format_feedback(result.failed_constraints, result.diagnostics)
  yt+1 = LLM.repair_ast(prompt, yt, feedback)
```

In practice, most successful repairs converge within **1–2 iterations**.

## 7) Experimental Results (Summary)

On a frozen test set (N=10) with multiple random seeds:

* **Direct LLM generation** achieves ~30% Pass@1
* **Verifier-guided repair** improves performance to ~80–90% Pass@3 depending on random seed (see result files)
* Increasing the repair budget beyond K=3 yields diminishing returns

These results demonstrate that **executable feedback is substantially more effective than resampling alone** for compositional geometry synthesis.

## 8) Key Takeaways

* Program synthesis becomes tractable when outputs are executable and verifiable
* Structured feedback enables rapid correction of structural errors
* Most failures are compositional (nested boolean structure), not numeric noise

## 9) Limitations and Future Work

* Extend primitive set (rounded rectangles, rotation)
* Scale evaluation to larger and more diverse datasets
* Explore learned repair policies or verifier-aware decoding

This project is intended as a **research-oriented systems prototype**, demonstrating how LLMs can be paired with executable verification to solve structured generation tasks more reliably.
