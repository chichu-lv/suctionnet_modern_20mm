"""Fetch the pinned external backbone; its source is not redistributed here."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "dda1d7586d4a9aea53da1636ffe64ae2c362e861"
REPOSITORY = "SimonHanrath/graspness_modern"
FILES = {
    "models/backbone_resunet14.py": "ad1a0eef6b2ab7d8f03d8d7a4a981dcc1e301c96",
    "models/resnet.py": "aa10c0f7810470c49a98572650f429d36a25b41f",
    "LICENSE": "5a44c2fb2095b57856cc3e5329dc8a528af14c74",
}


def fetch_backbone(destination: Path) -> None:
    for relative, expected in FILES.items():
        path = destination / relative
        if path.exists():
            content = path.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{relative}"
            with urlopen(url, timeout=60) as response:
                content = response.read()
        digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if digest != expected:
            raise ValueError(f"Pinned backbone content mismatch: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (destination / "provenance.json").write_text(json.dumps({
        "repository": REPOSITORY, "revision": REVISION, "git_blobs": FILES,
        "license": "See LICENSE: academic/noncommercial research terms",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("external/graspness_modern"))
    args = parser.parse_args()
    fetch_backbone(args.output)
    print(f"Verified backbone {REVISION} at {args.output.resolve()}")


if __name__ == "__main__":
    main()
