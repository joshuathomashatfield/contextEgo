"""Validate downloaded GTEA and create four subject-disjoint folds."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pcw.datasets.gtea import prepare

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--raw",default="data/raw/gtea"); p.add_argument("--output",default="data/processed/gtea"); p.add_argument("--frame-base",type=int,choices=[0,1],default=1)
    a=p.parse_args(); records=prepare(a.raw,a.output,a.frame_base); print(f"Prepared {len(records)} videos in four subject-disjoint folds")
if __name__=="__main__": main()
