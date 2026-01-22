import json, random, math, os, textwrap, uuid
from datetime import datetime

random.seed(42)

# --- Spec: schema + template instantiation utilities ---

TOL_DEFAULT = 0.05  # relative tolerance for most numeric checks in later verifier

def make_id(i):
    return f"ex_{i:04d}"

def ast_rect(w, h):
    return {"type": "Rect", "w": w, "h": h}

def ast_circle(r):
    return {"type": "Circle", "r": r}

def ast_translate(shape, dx, dy):
    return {"type": "Translate", "shape": shape, "dx": dx, "dy": dy}

def ast_union(a, b):
    return {"type": "Union", "a": a, "b": b}

def ast_difference(a, b):
    return {"type": "Difference", "a": a, "b": b}

def quantize(x, step=0.1):
    return round(round(x/step)*step, 2)

def sample_in_range(lo, hi, step=0.1):
    return quantize(random.uniform(lo, hi), step)

def safe_positive(lo=1.0, hi=20.0):
    return sample_in_range(lo, hi)

def ensure(cond, sampler, max_tries=200):
    for _ in range(max_tries):
        v = sampler()
        if cond(v):
            return v
    raise RuntimeError("Failed to sample valid parameters")

templates = []

# T1: Single Rectangle
templates.append({
    "template_id": "T1_single_rectangle",
    "difficulty": 1,
    "instantiate": lambda: (lambda w,h: (
        ast_rect(w,h),
        f"Create a rectangle of width {w} and height {h}.",
        [
            {"type":"bbox_width", "value": w, "tol": TOL_DEFAULT},
            {"type":"bbox_height","value": h, "tol": TOL_DEFAULT},
            {"type":"num_components","value": 1},
            {"type":"has_hole","value": False},
        ],
        {"w":w,"h":h}
    ))(
        safe_positive(3.0, 18.0),
        safe_positive(3.0, 18.0)
    )
})

# T2: Single Circle
templates.append({
    "template_id": "T2_single_circle",
    "difficulty": 1,
    "instantiate": lambda: (lambda r: (
        ast_circle(r),
        f"Draw a circle with radius {r}.",
        [
            {"type":"area", "value": round(math.pi*r*r, 4), "tol": 0.08},  # looser tol for area
            {"type":"num_components","value": 1},
            {"type":"has_hole","value": False},
        ],
        {"r":r}
    ))(
        safe_positive(1.0, 10.0)
    )
})

# T3: Translated Rectangle
templates.append({
    "template_id": "T3_translated_rectangle",
    "difficulty": 2,
    "instantiate": lambda: (lambda w,h,dx,dy: (
        ast_translate(ast_rect(w,h), dx, dy),
        f"Create a rectangle of width {w} and height {h}, shifted by ({dx}, {dy}) from the center.",
        [
            {"type":"bbox_width", "value": w, "tol": TOL_DEFAULT},
            {"type":"bbox_height","value": h, "tol": TOL_DEFAULT},
            {"type":"centroid", "value": [dx, dy], "tol": 0.07},
        ],
        {"w":w,"h":h,"dx":dx,"dy":dy}
    ))(
        safe_positive(3.0, 16.0),
        safe_positive(3.0, 16.0),
        sample_in_range(-5.0, 5.0, 0.1),
        sample_in_range(-5.0, 5.0, 0.1),
    )
})

# T4: Centered Square Cutout
templates.append({
    "template_id": "T4_centered_square_cutout",
    "difficulty": 3,
    "instantiate": lambda: (lambda w,h,s: (
        ast_difference(ast_rect(w,h), ast_rect(s,s)),
        f"Create a rectangle of width {w} and height {h}. Cut out a centered square hole of size {s}.",
        [
            {"type":"has_hole","value": True},
            {"type":"cutout_shape","value":"square"},
            {"type":"cutout_size","value": s, "tol": 0.10},
            {"type":"centered","shape":"cutout","target":"base","tol": 0.06},
            {"type":"num_components","value": 1},
        ],
        {"w":w,"h":h,"s":s}
    ))(
        (lambda: (lambda w,h: (w,h))(
            safe_positive(6.0, 18.0), safe_positive(6.0, 18.0)
        ))()[0],
        (lambda: (lambda w,h: (w,h))(
            safe_positive(6.0, 18.0), safe_positive(6.0, 18.0)
        ))()[1],
        # We'll re-sample coherently below; easier: explicit sampler with closure:
        0.0
    )
})

# Fix T4 (coherent sampling)
templates[-1]["instantiate"] = lambda: (lambda w,h: (
    (lambda s: (
        ast_difference(ast_rect(w,h), ast_rect(s,s)),
        f"Create a rectangle of width {w} and height {h}. Cut out a centered square hole of size {s}.",
        [
            {"type":"has_hole","value": True},
            {"type":"cutout_shape","value":"square"},
            {"type":"cutout_size","value": s, "tol": 0.06},
            {"type":"centered","shape":"cutout","target":"base","tol": 0.06},
            {"type":"num_components","value": 1},
        ],
        {"w":w,"h":h,"s":s}
    ))(
        ensure(lambda s: s < 0.7*min(w,h), lambda: safe_positive(1.0, min(w,h)-1.0))
    )
))(
    safe_positive(6.0, 18.0),
    safe_positive(6.0, 18.0),
)

# T5: Offset Square Cutout (top edge aligns with rectangle vertical center)
templates.append({
    "template_id": "T5_offset_square_cutout_top_edge_at_center",
    "difficulty": 4,
    "instantiate": lambda: (lambda w,h: (
        (lambda s: (
            (lambda dy: (
                ast_difference(ast_rect(w,h), ast_translate(ast_rect(s,s), 0.0, dy)),
                f"Create a rectangle of width {w} and height {h}. Cut out a square of size {s} whose top edge aligns with the vertical center of the rectangle.",
                [
                    {"type":"has_hole","value": True},
                    {"type":"cutout_shape","value":"square"},
                    {"type":"cutout_size","value": s, "tol": 0.10},
                    {"type":"cutout_inside_base","value": True},
                    {"type":"top_edge_y_equals","shape":"cutout","target":"base_center_y","tol": 0.06},
                    {"type":"centered_x","shape":"cutout","target":"base","tol": 0.06},
                ],
                {"w":w,"h":h,"s":s,"dy":dy}
            ))(
                quantize(-s/2, 0.1)  
            )
        ))(
            ensure(
                lambda s: (s < 0.45*h) and (s < 0.90*w),
                lambda: safe_positive(1.0, min(w, h) - 1.0)
            )
        )
    ))(
        safe_positive(8.0, 20.0),
        safe_positive(8.0, 20.0)
    )
})

# T6: Centered Circular Hole
templates.append({
    "template_id": "T6_centered_circular_hole",
    "difficulty": 3,
    "instantiate": lambda: (lambda w,h: (
        (lambda r: (
            ast_difference(ast_rect(w,h), ast_circle(r)),
            f"Create a rectangle of width {w} and height {h} with a centered circular hole of radius {r}.",
            [
                {"type":"has_hole","value": True},
                {"type":"cutout_shape","value":"circle"},
                {"type":"cutout_radius","value": r, "tol": 0.10},
                {"type":"centered","shape":"cutout","target":"base","tol": 0.06},
                {"type":"cutout_inside_base","value": True},
            ],
            {"w":w,"h":h,"r":r}
        ))(
            ensure(lambda r: r < 0.45*min(w,h), lambda: safe_positive(0.8, min(w,h)/2 - 0.5))
        )
    ))(
        safe_positive(6.0, 18.0),
        safe_positive(6.0, 18.0)
    )
})

# T7: Rectangle with circle union (circle sits on top edge so union adds area)
templates.append({
    "template_id": "T7_union_rect_and_circle_on_top_edge",
    "difficulty": 3,
    "instantiate": lambda: (lambda w,h: (
        (lambda r: (
            ast_union(
                ast_rect(w,h),
                ast_translate(ast_circle(r), 0.0, quantize(h/2, 0.1))
            ),
            f"Draw a rectangle of width {w} and height {h}. Add a circle of radius {r} centered on the rectangle’s top edge.",
            [
                {"type":"num_components","value": 1},
                {"type":"has_hole","value": False},
                {"type":"union_area_greater_than_rect_area", "value": w*h, "tol": 0.02},
            ],
            {"w":w,"h":h,"r":r}
        ))(
            (lambda r_min: (
                ensure(
                    lambda r: (r >= r_min) and (r <= 0.45*min(w,h)),
                    lambda: safe_positive(max(0.8, r_min), 0.45*min(w,h))
                )
            ))(math.sqrt(0.04*w*h / math.pi))
        )
    ))(
        safe_positive(6.0, 18.0),
        safe_positive(6.0, 18.0)
    )
})

# T8: Two side-by-side rectangles (horizontal)
templates.append({
    "template_id": "T8_two_side_by_side_rectangles",
    "difficulty": 4,
    "instantiate": lambda: (lambda w,h: (
        (lambda d: (
            ast_union(
                ast_translate(ast_rect(w,h), -d, 0.0),
                ast_translate(ast_rect(w,h),  d, 0.0)
            ),
            f"Create two identical rectangles of width {w} and height {h}, placed side by side horizontally.",
            [
                {"type":"num_components","value": 1},  # union should connect if d small enough; we will ensure overlap/touch
                {"type":"two_lobes_horizontal","value": True},
                {"type":"identical_subshapes","value": True},
            ],
            {"w":w,"h":h,"d":d}
        ))(
            # pick d such that rectangles touch/overlap slightly: centers separated by <= w
            ensure(lambda d: 0.45*w <= 2*d <= 0.95*w, lambda: sample_in_range(0.2, w, 0.1))
        )
    ))(
        safe_positive(4.0, 14.0),
        safe_positive(4.0, 14.0)
    )
})

# T9: Rectangular frame (nested rectangles)
templates.append({
    "template_id": "T9_rectangular_frame",
    "difficulty": 4,
    "instantiate": lambda: (lambda w,h: (
        (lambda w2,h2: (
            ast_difference(ast_rect(w,h), ast_rect(w2,h2)),
            f"Create a rectangular frame by cutting a smaller centered rectangle of width {w2} and height {h2} from a larger rectangle of width {w} and height {h}.",
            [
                {"type":"has_hole","value": True},
                {"type":"inner_rect_centered","value": True, "tol": 0.06},
                {"type":"bbox_width","value": w, "tol": TOL_DEFAULT},
                {"type":"bbox_height","value": h, "tol": TOL_DEFAULT},
                {"type":"inner_bbox_width","value": w2, "tol": 0.10},
                {"type":"inner_bbox_height","value": h2, "tol": 0.10},
            ],
            {"w":w,"h":h,"w2":w2,"h2":h2}
        ))(
            ensure(lambda w2: w2 < 0.8*w, lambda: safe_positive(1.0, w-1.0)),
            ensure(lambda h2: h2 < 0.8*h, lambda: safe_positive(1.0, h-1.0)),
        )
    ))(
        safe_positive(8.0, 20.0),
        safe_positive(8.0, 20.0)
    )
})

# T10: Offset circle hole inside rectangle
templates.append({
    "template_id": "T10_offset_circle_hole",
    "difficulty": 4,
    "instantiate": lambda: (lambda w,h: (
        (lambda r: (
            (lambda dx,dy: (
                ast_difference(ast_rect(w,h), ast_translate(ast_circle(r), dx, dy)),
                f"Create a rectangle of width {w} and height {h} with a circular hole of radius {r}, offset from the center by ({dx}, {dy}).",
                [
                    {"type":"has_hole","value": True},
                    {"type":"cutout_shape","value":"circle"},
                    {"type":"cutout_radius","value": r, "tol": 0.10},
                    {"type":"circle_center","value": [dx, dy], "tol": 0.07},
                    {"type":"circle_inside_rectangle","value": True},
                ],
                {"w":w,"h":h,"r":r,"dx":dx,"dy":dy}
            ))(
                # keep offset so circle stays inside: |dx| <= (w/2 - r) * 0.8
                ensure(lambda dx: abs(dx) <= 0.8*(w/2 - r), lambda: sample_in_range(-(w/2 - r), (w/2 - r), 0.1)),
                ensure(lambda dy: abs(dy) <= 0.8*(h/2 - r), lambda: sample_in_range(-(h/2 - r), (h/2 - r), 0.1)),
            )
        ))(
            ensure(lambda r: r < 0.35*min(w,h), lambda: safe_positive(0.8, min(w,h)/2 - 0.5))
        )
    ))(
        safe_positive(8.0, 20.0),
        safe_positive(8.0, 20.0)
    )
})

# --- Generate dataset ---
N_PER_TEMPLATE = 8   # 10 templates * 8 = 80 examples
rows = []
i = 0
for t in templates:
    for _ in range(N_PER_TEMPLATE):
        ast, prompt, constraints, params = t["instantiate"]()
        row = {
            "id": make_id(i),
            "template_id": t["template_id"],
            "difficulty": t["difficulty"],
            "prompt": prompt,
            "program_ast": ast,
            "constraints": constraints,
            "params": params,
            "schema_version": "v1.0",
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z"
        }
        rows.append(row)
        i += 1

# Write JSONL
out_path = "geometry_dataset_v1.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

out_path, len(rows), rows[0]