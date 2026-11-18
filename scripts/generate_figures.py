"""Convenience wrapper for pcw figures."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pcw.cli import main
main(["figures",*sys.argv[1:]])
