#!/usr/bin/env python3
"""Run the stdlib-only CD-009 coordinator or host worker."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suction20.run_supervisor import main

if __name__ == "__main__":
    main()
