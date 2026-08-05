#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.cli import main

raise SystemExit(main(["calibration", "import", *sys.argv[1:]]))
