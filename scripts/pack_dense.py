#!/usr/bin/env python3
"""Convert official create_dense_pcd.py outputs into NPZ for seal_annotate."""
import argparse
from pathlib import Path
from suction20.data_utils import pack_dense

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print(pack_dense(Path(args.input_dir), Path(args.output)))

if __name__ == "__main__":
    main()
