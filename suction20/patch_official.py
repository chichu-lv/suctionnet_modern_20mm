"""Generate 10/20 mm variants of upstream SuctionNet annotation scripts."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

def patch_seal_source(source: str, radius_m: float, selected_ids: list[int] | None) -> str:
    if radius_m <= 0 or radius_m > 0.5:
        raise ValueError("Radius must be in metres and positive")
    updated, count = re.subn(
        r"(?m)^([ \t]*radius[ \t]*=[ \t]*)0\.01([ \t]*)$",
        lambda m: f"{m.group(1)}{radius_m:.6f}{m.group(2)}",
        source,
    )
    if count != 1:
        raise RuntimeError("Cannot find exactly one official radius=0.01 assignment")
    if len(re.findall(r"(?m)^\s*num_split\s*=\s*72\s*$", source)) != 1:
        raise RuntimeError("Expected the official 72-sector Seal algorithm")
    original_ring = "(dist > radius - 0.001) & (dist < radius + 0.001)"
    if updated.count(original_ring) != 1:
        raise RuntimeError("Official ring search expression changed")
    if abs(radius_m - 0.01) > 1e-12:
        updated = updated.replace(
            original_ring,
            "(dist > radius - (0.1 * radius)) & (dist < radius + (0.1 * radius))",
        )
    updated = re.sub(r"(?<![A-Za-z0-9_])np\.bool(?![A-Za-z0-9_])", "np.bool_", updated)
    if selected_ids is not None:
        if not selected_ids or any(i not in range(88) for i in selected_ids):
            raise ValueError("Object IDs must be from 0 to 87")
        ids = sorted(set(selected_ids))
        updated, n = re.subn(r"for i in range\(88\):", f"for i in {ids!r}:", updated)
        if n != 1:
            raise RuntimeError("Cannot find official object loop")
    return updated

def create_variant(source_path: str | Path, target_path: str | Path,
                   radius_mm: float, selected_ids: list[int] | None) -> dict:
    source = Path(source_path).read_text(encoding="utf-8")
    output = patch_seal_source(source, radius_mm / 1000.0, selected_ids)
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(output, encoding="utf-8")
    return {
        "radius_mm": radius_mm,
        "ring_half_width_m": 0.001 if abs(radius_mm - 10) < 1e-9 else radius_mm / 1000 * 0.1,
        "object_ids": selected_ids,
    }

def create_wrench_g0_variant(source_path: str | Path, target_path: str | Path,
                             scene_ids: list[int]) -> dict:
    if not scene_ids or any(i not in range(190) for i in scene_ids):
        raise ValueError("Scene IDs must be from 0 to 189")
    source = Path(source_path).read_text(encoding="utf-8")
    original = "for scene_idx in range(190):"
    if source.count(original) != 1:
        raise RuntimeError("Official 190-scene loop is missing")
    if "wrench_thre = k * radius * np.pi * np.sqrt(2)" not in source:
        raise RuntimeError("Original 10mm Wrench formula not found")
    if not re.search(r"(?m)^radius\s*=\s*0\.01\s*$", source):
        raise RuntimeError("G0 requires the original 10mm radius")
    chosen = sorted(set(scene_ids))
    output = source.replace(original, f"for scene_idx in {chosen!r}:")
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(output, encoding="utf-8")
    return {"scene_ids": chosen, "radius_m": 0.01,
            "wrench_threshold": "sqrt(2) * k * r * pi"}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radius-mm", type=float, required=True)
    parser.add_argument("--objects", default="")
    args = parser.parse_args()
    ids = [int(x) for x in args.objects.split(",") if x.strip()] if args.objects else None
    print(json.dumps(create_variant(args.source, args.output, args.radius_mm, ids),
                     indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
