"""(2) Improved prompting -- STUDENT VERSION.

Run with:
    uv run main.py --method improved
"""

import json

from naive import SYSTEM

def _dumps(obj):  # you may use this to check JSON validity.
    return json.dumps(obj, ensure_ascii=False)


# TODO modify the instruction to achieve high accuracy.
INSTRUCTION = """
Convert the given announcement into the required JSON format.

Instructions:
1. Return only one valid JSON object.
2. Extract only "when", "where", "items", "task", and "teacher".
3. Discard all other information.
4. If a field is missing, set its value to null.
5. The "items" field must be an array of strings.
6. Do not guess or invent information.
7. For "task", copy only the exact activity or mission phrase from the announcement. Do not paraphrase it.
8. Follow the example below.

Example:

Input:
Hello, students. The trip is on the first Tuesday of next month. Homeroom teacher: Ms. Han Seonwoo. Make sure to pack comfortable sneakers, a 1L water bottle. This time the activity is a traditional craft experience. We will visit Hahoe Folk Village in Andong. Please refer to the parent newsletter for details.

Output:
{
  "when": "the first Tuesday of next month",
  "where": "Hahoe Folk Village in Andong",
  "items": [
    "comfortable sneakers",
    "a 1L water bottle"
  ],
  "task": "a traditional craft experience",
  "teacher": "Han Seonwoo"
}

Now process the following announcement:
""".strip()

def build_messages(text):
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": INSTRUCTION + "\n\n" + text,
        },
    ]

def run(text, runner, max_new_tokens=256):
    return runner.chat(build_messages(text), max_new_tokens=max_new_tokens)
