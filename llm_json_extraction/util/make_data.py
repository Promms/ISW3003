"""Build the field-trip announcement dataset (~100 items) -> announcements.json.

Key idea:
- First pick the values that go into the ground-truth `labels`, then
- splice those values into natural sentence templates to make the `text`.
  => This guarantees the labels are 100% correct (we built the text FROM the
     labels, not the other way around).
- Each field may or may not appear (if absent, its label is null).
  => This lets us test whether the model leaves missing info as null.

This script is the INSTRUCTOR'S tool. Students are given the resulting
announcements.json (text + answers) and do not need to run this.

Run:
    uv run util/make_data.py        # writes ../data/announcements.json
"""

import json
import random
from pathlib import Path

# The extraction fields (= JSON keys). Same as schema.FIELDS, but spelled out
# here so this script under data/ runs without import-path issues.
FIELDS = ["when", "where", "items", "task", "teacher"]

# ---------------------------------------------------------------------------
# Value pools
# ---------------------------------------------------------------------------
WHENS = [
    "June 15 at 9:00 AM", "July 3 at 8:30 AM", "next Friday at 1:00 PM",
    "September 20 to 22", "May 8 at 10:00 AM", "October 5 at 7:40 AM",
    "November 12 at 2:00 PM", "April 19 at 8:00 AM", "August 1 at noon",
    "March 28 at 9:30 AM", "the first Tuesday of next month", "December 2 at 8:00 AM",
]
WHERES = [
    "Gyeongju Bulguksa Temple", "Seoraksan National Park", "Haeundae Beach in Busan",
    "Suncheon Bay Wetland", "the National Museum of Korea",
    "Seongsan Ilchulbong Peak in Jeju", "Jeonju Hanok Village",
    "Gyeongpo Beach in Gangneung", "the Royal Tomb of King Muryeong",
    "Hahoe Folk Village in Andong", "Odongdo Island in Yeosu",
    "the German Village in Namhae",
]
ITEMS_POOL = [
    "comfortable sneakers", "a 1L water bottle", "a packed lunch", "a rain poncho",
    "a hat", "sunscreen", "a notebook and pen", "a change of clothes", "snacks",
    "personal medication", "a handkerchief", "a camera",
]
TASKS = [
    "writing a report after touring the heritage site",
    "observing the tidal flat ecosystem",
    "a team history quiz",
    "a traditional craft experience",
    "a nature photography mission",
    "listening to the site guide and taking notes",
    "a survey of local specialty products",
    "a group photo and talent show",
    "an environmental cleanup activity",
    "sketching museum artifacts",
]
TEACHERS = [
    "Kim Minjun", "Lee Seoyeon", "Park Doyoon", "Choi Jiwoo", "Jung Hajun",
    "Kang Sua", "Yoon Jiho", "Lim Nayoon", "Han Seonwoo", "Oh Yerin",
]
TITLES = ["Mr.", "Ms."]  # distractor honorific; the label keeps only the name

# ---------------------------------------------------------------------------
# Sentence templates (several phrasings so naive parsing is not trivial)
# ---------------------------------------------------------------------------
WHEN_TMPL = ["The trip is on {v}.", "We depart on {v}.", "Departure: {v}.",
             "We will gather at the school gate on {v}."]
WHERE_TMPL = ["This time we are going to {v}.", "The destination is {v}.",
              "Destination: {v}.", "We will visit {v}."]
ITEMS_TMPL = ["Please bring {v}.", "Make sure to pack {v}.", "What to bring: {v}.",
              "Each student should prepare {v}."]
TASK_TMPL = ["This time the activity is {v}.", "The main schedule is {v}.",
             "On site, we will be doing {v}.", "Activity: {v}."]
TEACHER_TMPL = ["The teacher in charge is {hon} {v}.", "Homeroom teacher: {hon} {v}.",
                "Please contact {hon} {v} with any questions.",
                "{hon} {v} will be leading the group."]

# Filler sentences with no information (header/footer) to add some noise.
HEADERS = [
    "Hello, students.", "Here is the announcement for our upcoming field trip.",
    "Please read the following carefully.", "This is the notice for this month's field trip.",
]
FOOTERS = [
    "We look forward to your participation.",
    "Please refer to the parent newsletter for details.",
    "Please do not be late.", "Be sure to follow all safety rules.", "",
]


def make_one(rng):
    """Build one announcement and its ground-truth labels."""
    labels = {f: None for f in FIELDS}

    # Include each field with 75% probability (but keep at least 2).
    present = {f: rng.random() < 0.75 for f in FIELDS}
    if sum(present.values()) < 2:
        for f in rng.sample(FIELDS, 2):
            present[f] = True

    sentences = []

    if present["when"]:
        v = rng.choice(WHENS)
        labels["when"] = v
        sentences.append(rng.choice(WHEN_TMPL).format(v=v))

    if present["where"]:
        v = rng.choice(WHERES)
        labels["where"] = v
        sentences.append(rng.choice(WHERE_TMPL).format(v=v))

    if present["items"]:
        k = rng.randint(2, 4)
        v = rng.sample(ITEMS_POOL, k)
        labels["items"] = v
        sentences.append(rng.choice(ITEMS_TMPL).format(v=", ".join(v)))

    if present["task"]:
        v = rng.choice(TASKS)
        labels["task"] = v
        sentences.append(rng.choice(TASK_TMPL).format(v=v))

    if present["teacher"]:
        v = rng.choice(TEACHERS)
        labels["teacher"] = v  # label is the name only, no title
        sentences.append(rng.choice(TEACHER_TMPL).format(hon=rng.choice(TITLES), v=v))

    # Shuffle the info sentences and wrap with filler header/footer.
    rng.shuffle(sentences)
    body = [rng.choice(HEADERS)] + sentences + [rng.choice(FOOTERS)]
    text = " ".join(s for s in body if s)

    return {"text": text, "labels": labels}


def main():
    rng = random.Random(42)  # fixed seed for reproducibility
    n = 100
    records = [{"id": i + 1, **make_one(rng)} for i in range(n)]

    out_path = Path(__file__).parent.parent / "data" / "announcements.json"
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    counts = {f: sum(1 for r in records if r["labels"][f] is not None) for f in FIELDS}
    print(f"saved {n} announcements -> {out_path}")
    print("per-field presence counts:", counts)
    print("\n[example]")
    print(records[0]["text"])
    print(json.dumps(records[0]["labels"], ensure_ascii=False))


if __name__ == "__main__":
    main()
