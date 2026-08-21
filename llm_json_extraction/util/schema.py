"""JSON schema definition and helpers for handling model output.

Notes for students:
- The output we want from the LLM is a single JSON object with the 5 keys below.
- Small models often add extra prose or wrap the JSON in a ```json code fence,
  so we need a helper that robustly pulls just the JSON out of the raw string.
"""

import json
import re

# The fields (= JSON keys) we extract, with short English descriptions.
# Both evaluate.py and improved.py share this definition.
FIELDS = ["when", "where", "items", "task", "teacher"]

FIELD_DESC = {
    "when": "departure date/time",
    "where": "destination / place",
    "items": "list of things to bring (array of strings)",
    "task": "the activity / mission to do on site",
    "teacher": "name of the teacher in charge",
}

# Titles to strip from teacher names before comparing.
TEACHER_TITLES = {"mr", "ms", "mrs", "miss", "teacher"}


def extract_json(raw):
    """Pull a single JSON object out of the model's free-form text (raw).

    Returns a dict, or None on failure. Tries, in order:
      1) the content inside a ```json ... ``` code fence
      2) the first balanced { ... } block in the string
      3) the whole string as-is
    """
    if raw is None:
        return None
    text = str(raw).strip()

    candidates = []

    # 1) Inside a code fence.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))

    # 2) From the first '{' to its matching '}'.
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break

    # 3) Last resort: the whole string.
    candidates.append(text)

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def norm_text(value):
    """Light normalization for string comparison (whitespace/case/edge punctuation)."""
    if value is None:
        return None
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,!?")
    return s or None


def norm_teacher(value):
    """Compare teacher names after dropping titles like 'Mr.' / 'Ms.' / 'Teacher'."""
    if value is None:
        return None
    kept = []
    for token in re.split(r"\s+", str(value).strip()):
        bare = token.strip(".,").lower()
        if bare in TEACHER_TITLES:
            continue
        kept.append(token)
    return norm_text(" ".join(kept))


def items_to_set(value):
    """Turn an 'items' value into a normalized set of strings.

    Accepts None / a string / a list, so we can compare even when the model
    returns "a, b, c" as one string instead of a JSON array.
    """
    if value is None:
        return set()
    if isinstance(value, str):
        parts = re.split(r"[,/]| and ", value)
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = [value]
    out = set()
    for p in parts:
        n = norm_text(p)
        if n:
            out.add(n)
    return out
