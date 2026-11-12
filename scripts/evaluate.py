"""Evaluate a checkpoint using actual predicted R20 poses."""
import argparse
import json
from pathlib import Path
import torch

from suction20.runner import add_runtime_arguments, make_model, make_source_dataset, override_paths, require_device
from suction20.training import load_checkpoint, load_config
from suction20.validation import checkpoint_inference_config, evaluate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["val", "smoke_val", "test_seen", "test_similar", "test_novel"])
    parser.add_argument("--output")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    config = override_paths(load_config(args.config), args)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_inference_config(config, state)
    device = require_device(args.device)
    dataset = make_source_dataset(config, split=args.split)
    model = make_model(config, dataset.data_cfg, device)
    load_checkpoint(args.checkpoint, model, source_label_config=dataset.data_cfg)
    result = evaluate_model(model, config, device, split=args.split)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
