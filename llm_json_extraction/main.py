"""Entry point -- run both prompting strategies, then evaluate and compare.

Usage:
    uv run main.py                 # naive + improved, all 100 examples
    uv run main.py --limit 10      # quick test (first 10)
    uv run main.py --method naive   # one strategy only
    uv run main.py --method improved
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

import naive
import improved
from util.evaluate import (
    aggregate,
    markdown_table,
    print_comparison,
    print_report,
    score_record,
)
from util.model_runner import ModelRunner

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "announcements.json"

# strategy name -> run(text, runner, max_new_tokens) -> raw output string.
STRATEGIES = {
    "naive": naive.run,
    "improved": improved.run,
}

def load_data(limit=None):
    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return records[:limit] if limit else records


def run_strategy(name, strategy_run, runner, records, max_new_tokens):
    """Run inference + scoring over the whole dataset for one strategy.

    Shows a live progress bar (e.g. "naive  11/100") with the running accuracy.
    """
    rows, scores = [], []
    bar = tqdm(records, desc=name, unit="ex")
    for rec in bar:
        raw = strategy_run(rec["text"], runner, max_new_tokens)
        result = score_record(raw, rec["labels"])
        scores.append(result)
        rows.append({
            "id": rec["id"],
            "text": rec["text"],
            "labels": rec["labels"],
            "raw_output": raw,
            "score": result,
        })
        # live running accuracy so students see quality as it progresses
        running = sum(s["overall"] for s in scores) / len(scores)
        bar.set_postfix(acc=f"{running:.0%}")
    return rows, aggregate(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["naive", "improved", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default="results", help="folder to save raw outputs")
    args = ap.parse_args()

    records = load_data(args.limit)
    methods = ["naive", "improved"] if args.method == "both" else [args.method]
    print(f"announcements: {len(records)} / strategies: {', '.join(methods)}")

    runner = ModelRunner()  # load the model once and share it
    out_dir = ROOT / args.out
    out_dir.mkdir(exist_ok=True)

    results = {}
    for name in methods:
        print(f"\n----- running '{name}' -----")
        rows, agg = run_strategy(name, STRATEGIES[name], runner, records, args.max_new_tokens)
        results[name] = agg
        print_report(name, agg)
        (out_dir / f"{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if len(results) > 1:
        print_comparison(results)

    print("\n----- Markdown (paste into README) -----")
    print(markdown_table(results))
    print(f"\nraw outputs saved under: {out_dir}/")


if __name__ == "__main__":
    main()
