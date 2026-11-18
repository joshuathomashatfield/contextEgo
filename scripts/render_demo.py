"""Convenience wrapper for pcw demo."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pcw.cli import main
main(["demo",*sys.argv[1:]])
