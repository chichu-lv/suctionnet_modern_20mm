"""Convert official dense-point-cloud files to the expected NPZ format."""
from pathlib import Path
import numpy as np

def pack_dense(input_dir: Path, output: Path) -> dict:
    input_dir, output = Path(input_dir), Path(output)
    points = np.load(input_dir / "points.npy", allow_pickle=False)
    normals = np.load(input_dir / "normals.npy", allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("Expected points.npy and normals.npy, both [N,3]")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError("Dense geometry contains NaN or Infinity")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, points=points.astype(np.float32),
                        normals=normals.astype(np.float32))
    return {"input": str(input_dir), "output": str(output), "count": len(points)}
