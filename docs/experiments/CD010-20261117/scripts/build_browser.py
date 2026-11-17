"""Build a static Chinese browser from completed baseline cases; never infer.

Serve the output with the existing result directory HTTP server so per-case
JSON fetches work. Assets remain external and relative to the generated HTML.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from urllib.parse import quote


def relative_url(path, output):
    return quote(Path(os.path.relpath(path, output.parent)).as_posix(), safe="/")


def build_index(results, output, score_key="q"):
    rows = []
    for path in sorted((results / "cases").glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("schema_version") not in ("suction20.target-baseline.case.v1", "suction20.target-finetuned.case.v1"):
            raise ValueError(f"Unsupported case schema: {path}")
        if case.get("ranking", {}).get("score_field", "q") != score_key:
            raise ValueError(f"Unexpected ranking field: {path}")
        metadata = case.get("metadata", {})
        review = metadata.get("raw_review", metadata.get("manual_outcome"))
        state = review.get("state") if isinstance(review, dict) else review
        status = "success" if state is True else "failure" if state is False else "missed" if state == "missed" else "unmarked"
        rendering = case.get("rendering", {})
        selected = sorted((c for c in case["candidates"] if c.get("scene_rank") is not None),
                          key=lambda c: c["scene_rank"])
        top_score = selected[0][score_key] if selected else None
        if top_score is not None and not math.isfinite(top_score):
            raise ValueError(f"Nonfinite top score in {path}")
        rows.append(dict(sample=case["sample_id"], day=str(metadata.get("day", case["sample_id"][:8])),
                         status=status, score=top_score, q=selected[0]["q"] if selected else None, nms_count=len(selected),
                         original_count=len(metadata.get("planned_pose_cam", [])),
                         comparison=relative_url(results / "comparison" / path.name, output),
                         json=relative_url(path, output), npz=relative_url(path.with_suffix(".npz"), output),
                         rgb=relative_url(results / "images" / rendering["rgb"], output) if rendering.get("rgb") else None,
                         overlay=relative_url(results / "images" / rendering["overlay"], output) if rendering.get("overlay") else None))
    return sorted(rows, key=lambda row: (row["day"], row["sample"]))


def build(results, output, finetuned_results=None):
    rows = build_index(results, output)
    models = [dict(id="pretrained", label="预训练 E5 best · step 2880", score_key="q",
                   score_label="物理质量 q", checkpoint_step=2880, rows=rows)]
    if finetuned_results is not None:
        tuned = build_index(finetuned_results, output, "task_success_score")
        if len(rows) != 401 or [r['sample'] for r in tuned] != [r['sample'] for r in rows]:
            raise ValueError("Publishing comparison requires identical complete 401-frame sample sets")
        for root, expected_step in ((results, 2880), (finetuned_results, 1000)):
            for path in sorted((root / "cases").glob("*.json")):
                case = json.loads(path.read_text())
                if case["identity"]["checkpoint_step"] != expected_step:
                    raise ValueError(f"Unexpected checkpoint step: {path}")
                if not path.with_suffix('.npz').is_file() or not (root/'comparison'/path.name).is_file():
                    raise ValueError(f"Incomplete case assets: {path}")
                for key in ('rgb', 'overlay'):
                    if not (root/'images'/case['rendering'][key]).is_file():
                        raise ValueError(f"Missing {key} image: {path}")
        models.append(dict(id="finetuned", label="微调 E5 best · step 1000", score_key="task_success_score",
                           score_label="任务评分", checkpoint_step=1000, rows=tuned))
    bundle = dict(default_model="finetuned" if finetuned_results is not None else "pretrained", models=models)
    # JSON is a data script, but still escape HTML script-closing characters.
    payload = json.dumps(bundle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e")):
        payload = payload.replace(character, escaped)
    static = Path(__file__).parent / "static"
    document = (static / "browser.html").read_text(encoding="utf-8").replace("__CASE_INDEX__", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = output.parent / "browser-assets"
    asset_dir.mkdir(exist_ok=True)
    for source in sorted(static.iterdir()):
        if source.suffix in {".js", ".css"}:
            target = asset_dir / source.name
            asset_tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
            shutil.copyfile(source, asset_tmp)
            os.replace(asset_tmp, target)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    temporary.write_text(document, encoding="utf-8")
    os.replace(temporary, output)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--finetuned-results", type=Path, help="Complete 401-frame finetuned result directory")
    args = parser.parse_args()
    results = args.results.resolve()
    if not (results / "cases").is_dir():
        parser.error("--results must contain a cases/ directory")
    output = args.output.resolve() if args.output else results / "browser.html"
    count = build(results, output, args.finetuned_results.resolve() if args.finetuned_results else None)
    print(f"Built {count} cases: {output}")


if __name__ == "__main__":
    main()
