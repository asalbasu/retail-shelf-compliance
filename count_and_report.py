import json
import re
from collections import Counter
from typing import Dict, List, Tuple, Any


# ----------------------------
# Normalization + aliases
# ----------------------------
ALIASES = {
    "coke": "coca cola",
    "coca-cola": "coca cola",
    "coca cola classic": "coca cola",
    "diet coca cola": "diet coke",
    "coke zero sugar": "coke zero",
    "7 up": "7up",
    "cheerios oat crunch cinnamon": "cheerios oat crunch",
    "multi-grain cheerios": "multi grain cheerios",
    "life original": "life cereal original",
    "life cereal": "life cereal original",
    "toasted o's": "toasted oats cereal",
    "toasted oats": "toasted oats cereal",
}

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9\s]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)

def _best_match(label: str, expected_keys: List[str]) -> str:
    """
    Map a detected label to an expected product key using:
    1) exact alias normalization
    2) substring match (best effort)
    """
    lab = _norm(label)
    if lab in expected_keys:
        return lab

    # substring match: choose the longest expected key that appears in label
    candidates = [k for k in expected_keys if k and (k in lab or lab in k)]
    if candidates:
        return max(candidates, key=len)

    return lab  # keep as-is (may become "extra")


# ----------------------------
# Planogram parsing
# ----------------------------
def load_planogram_expected(planogram_path: str) -> Dict[str, int]:
    """
    Supports both:
    A) cereal_planogram.json: {"shelves":[{"positions":[{"product","quantity"}]}]}
    B) drinks_planogram.json: {"products":[{"name","count"}]}
    """
    data = json.load(open(planogram_path, "r", encoding="utf-8"))

    expected: Dict[str, int] = {}

    # Format B: {"products":[{"name":..., "count":...}, ...]}
    if isinstance(data, dict) and "products" in data and isinstance(data["products"], list):
        for p in data["products"]:
            name = _norm(str(p.get("name", "")))
            cnt = int(p.get("count", 0) or 0)
            if name:
                expected[name] = expected.get(name, 0) + max(cnt, 0)
        return expected

    # Format A: {"shelves":[{"positions":[{"product":..., "quantity":...}]}]}
    if isinstance(data, dict) and "shelves" in data and isinstance(data["shelves"], list):
        for shelf in data["shelves"]:
            positions = shelf.get("positions", [])
            if not isinstance(positions, list):
                continue
            for slot in positions:
                prod = _norm(str(slot.get("product", "")))
                qty = int(slot.get("quantity", 0) or 0)
                if prod:
                    expected[prod] = expected.get(prod, 0) + max(qty, 0)
        return expected

    # fallback: try to find any strings under common keys
    for key in ["items", "planogram", "assortment"]:
        if key in data and isinstance(data[key], list):
            for item in data[key]:
                name = _norm(str(item.get("name", item.get("product", ""))))
                cnt = int(item.get("count", item.get("quantity", 1)) or 1)
                if name:
                    expected[name] = expected.get(name, 0) + max(cnt, 0)

    return expected


def extract_prompt_from_planogram(planogram_path: str) -> str:
    expected = load_planogram_expected(planogram_path)
    # Return dotted prompt string
    keys = sorted(expected.keys())
    if not keys:
        return ""
    return " . ".join(keys)


# ----------------------------
# Scoring
# ----------------------------
def create_compliance_report(labels: List[str], planogram_path: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    expected = load_planogram_expected(planogram_path)
    expected_keys = list(expected.keys())

    # Map detected labels to expected keys where possible
    mapped = [_best_match(l, expected_keys) for l in labels]
    detected_counts = Counter(mapped)

    # Presence score: % of SKUs expected that were detected at least once
    if expected:
        present = sum(1 for k in expected if detected_counts.get(k, 0) > 0)
        presence_score = 100.0 * present / max(len(expected), 1)
    else:
        presence_score = 0.0

    # Count score: how close counts are (but capped so it doesn't punish too hard)
    total_expected = sum(expected.values()) if expected else 0
    matched_units = 0
    missing = []
    extra = []

    for prod, exp in expected.items():
        got = detected_counts.get(prod, 0)
        matched_units += min(got, exp)
        if got < exp:
            missing.append(f"{prod} (missing {exp - got})")

    for prod, got in detected_counts.items():
        exp = expected.get(prod, 0)
        if got > exp:
            extra.append(f"{prod} (extra {got - exp})")
        elif prod not in expected:
            extra.append(f"{prod} (unexpected {got})")

    strict_count_score = 0.0 if total_expected == 0 else (100.0 * matched_units / total_expected)

    # Combined score: prioritize presence (more realistic for DINO on shelves)
    # You can tune weights later.
    compliance_score = 0.7 * presence_score + 0.3 * strict_count_score

    report = {
        "compliance_score": float(compliance_score),
        "presence_score": float(presence_score),
        "strict_count_score": float(strict_count_score),
        "missing": missing,
        "extra": extra,
        "expected": expected,
    }

    return report, dict(detected_counts)
