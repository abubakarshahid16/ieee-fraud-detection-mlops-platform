from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Trigger retraining workflow from monitoring events.")
    parser.add_argument("--event", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--value", required=True, type=float)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument(
        "--comparison",
        choices=["gte", "lte"],
        default="gte",
        help="Comparison rule for triggering retraining.",
    )
    parser.add_argument("--output", default="artifacts/retraining_event.json")
    args = parser.parse_args()

    should_trigger = (
        args.value >= args.threshold if args.comparison == "gte" else args.value <= args.threshold
    )

    payload = {
        "event": args.event,
        "metric": args.metric,
        "value": args.value,
        "threshold": args.threshold,
        "comparison": args.comparison,
        "action": "trigger_retraining" if should_trigger else "no_action",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
