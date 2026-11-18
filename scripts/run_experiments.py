"""Convenience wrapper for pcw matrix."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pcw.cli import main
main(["matrix",*sys.argv[1:]])
