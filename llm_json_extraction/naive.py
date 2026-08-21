"""(1) Naive prompting -- the baseline.

Run with:
    uv run main.py --method naive
"""

SYSTEM = "You are a helpful AI assistant."


def build_messages(text):
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "Extract the following information from the announcement and "
                "give it as JSON: when, where, items, task, teacher.\n\n" + text
            ),
        },
    ]


def run(text, runner, max_new_tokens=256):
    """Naive = a single model call. No schema, no checking, no repair."""
    return runner.chat(build_messages(text), max_new_tokens=max_new_tokens)
