"""Select completed benchmark results and export or start fresh formal runs.

batch-combo chooses the highest measured throughput independently on each host;
the historical final suite retains its original effective-batch8 selection rule.
All exported runs initialize from step0 in a new training root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from suction20.final_pretraining import finalize_pretraining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-dir', required=True, help='Completed --suite batch-combo or --suite final output directory')
    parser.add_argument('--training-root', required=True, help='New per-deployment root; exports are one-shot per host')
    parser.add_argument('--resource-map', help='JSON file: {"E5":{"gpu":"0","cpus":"0-15"}}, with this host\'s overrides')
    parser.add_argument('--launch', action='store_true', help='Start newly initialized formal trainers after saving all configs; otherwise only export')
    args = parser.parse_args()
    directory = Path(args.benchmark_dir)
    plan = json.loads((directory / 'plan.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    resources = json.loads(Path(args.resource_map).read_text()) if args.resource_map else None
    report = finalize_pretraining(plan, summary, args.training_root, resource_map=resources, launch=args.launch)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
