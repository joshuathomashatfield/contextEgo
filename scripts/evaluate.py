"""Convenience wrapper for pcw evaluate."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pcw.cli import main
main(["evaluate",*sys.argv[1:]])
