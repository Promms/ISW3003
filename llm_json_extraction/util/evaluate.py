"""Evaluation -- compare the model's JSON against the ground-truth labels.

Metrics:
- parse_rate    : fraction of outputs we could extract valid JSON from.
- per-field acc : for when/where/task/teacher, correct if they match after
                  normalization. (Both null also counts as correct -- knowing
                  that something is absent is part of the task.)
- items_f1      : items may differ in order/count, so we score them as a
                  set-based F1.
- overall       : mean accuracy across all fields.
"""

from util.schema import FIELDS, extract_json, items_to_set, norm_teacher, norm_text

# when/where/task use plain normalization; teacher strips titles.
_TEXT_FIELDS = ["when", "where", "task"]


def _field_correct(field, pred, gold):
    """Return whether one string field is correct (bool)."""
    if field == "teacher":
        return norm_teacher(pred) == norm_teacher(gold)
    return norm_text(pred) == norm_text(gold)


def _items_f1(pred, gold):
    """Set-based F1 for items. 1.0 if both are empty."""
    g = items_to_set(gold)
    p = items_to_set(pred)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    tp = len(g & p)
    if tp == 0:
        return 0.0
    precision = tp / len(p)
    recall = tp / len(g)
    return 2 * precision * recall / (precision + recall)


def score_record(raw_output, gold_labels):
    """Score a single raw model output; return a detailed result dict."""
    pred = extract_json(raw_output)  # None means parsing failed
    parsed = pred is not None
    pred = pred or {}

    per_field = {}
    for f in _TEXT_FIELDS + ["teacher"]:
        per_field[f] = _field_correct(f, pred.get(f), gold_labels.get(f))

    items_f1 = _items_f1(pred.get("items"), gold_labels.get("items"))
    # "items correct" (for accuracy) means a perfect F1 of 1.0.
    per_field["items"] = items_f1 == 1.0

    overall = sum(per_field[f] for f in FIELDS) / len(FIELDS)
    return {
        "parsed": parsed,
        "per_field": per_field,
        "items_f1": items_f1,
        "overall": overall,
    }


def aggregate(scores):
    """Aggregate a list of score_record results into mean metrics."""
    n = len(scores)
    if n == 0:
        return {}
    return {
        "n": n,
        "parse_rate": sum(s["parsed"] for s in scores) / n,
        "items_f1": sum(s["items_f1"] for s in scores) / n,
        "overall": sum(s["overall"] for s in scores) / n,
        "per_field": {
            f: sum(s["per_field"][f] for s in scores) / n for f in FIELDS
        },
    }


def print_report(name, agg):
    print(f"\n===== [{name}] (n={agg['n']}) =====")
    print(f"  JSON parse rate       : {agg['parse_rate']:6.1%}")
    print(f"  overall field accuracy: {agg['overall']:6.1%}")
    print(f"  items F1              : {agg['items_f1']:6.1%}")
    print("  per-field accuracy:")
    for f in FIELDS:
        print(f"    - {f:<8}: {agg['per_field'][f]:6.1%}")


def print_comparison(results):
    """Print {name: agg} side by side as a console table."""
    names = list(results.keys())
    rows = [("parse_rate", "JSON parse rate"), ("overall", "overall accuracy"),
            ("items_f1", "items F1")]
    print("\n========== comparison ==========")
    header = f"{'metric':<18}" + "".join(f"{n:>12}" for n in names)
    print(header)
    print("-" * len(header))
    for key, label in rows:
        line = f"{label:<16}" + "".join(f"{results[n][key]:>11.1%} " for n in names)
        print(line)
    print(f"\n{'per-field accuracy':<16}" + "".join(f"{n:>12}" for n in names))
    for f in FIELDS:
        line = f"  {f:<14}" + "".join(f"{results[n]['per_field'][f]:>11.1%} " for n in names)
        print(line)


def markdown_table(results):
    """Return a Markdown table students can paste into the README."""
    names = list(results.keys())
    head = "| metric | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(names) + 1)
    lines = [head, sep]
    for key, label in [("parse_rate", "JSON parse rate"),
                       ("overall", "overall field accuracy"),
                       ("items_f1", "items F1")]:
        cells = " | ".join(f"{results[n][key]:.1%}" for n in names)
        lines.append(f"| {label} | {cells} |")
    lines.append("")
    lines.append("Per-field accuracy:")
    lines.append("")
    lines.append("| field | " + " | ".join(names) + " |")
    lines.append(sep)
    for f in FIELDS:
        cells = " | ".join(f"{results[n]['per_field'][f]:.1%}" for n in names)
        lines.append(f"| {f} | {cells} |")
    return "\n".join(lines)
