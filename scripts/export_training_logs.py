#!/usr/bin/env python3
"""Export verified committed training logs and separately labeled raw tails."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suction20.training_telemetry import export_training_logs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="New export snapshot directory")
    parser.add_argument("--checkpoint-role", choices=("current", "predecessor", "best"), default="current")
    parser.add_argument("--resource-log", action="append", default=[], help="Additional worker resources.jsonl (repeatable)")
    parser.add_argument("--live-attempt-id", help="Caller-designated live tail label; does not prove liveness")
    args = parser.parse_args()
    result = export_training_logs(args.run_dir, args.output_dir, checkpoint_role=args.checkpoint_role,
                                  resource_logs=args.resource_log, live_attempt_id=args.live_attempt_id)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()),
                      "checkpoint_role": result["checkpoint_role"],
                      "global_step": result["checkpoint"]["global_step"],
                      "categories": result["categories"], "gaps": result["gaps"]}, allow_nan=False))


if __name__ == "__main__":
    main()
