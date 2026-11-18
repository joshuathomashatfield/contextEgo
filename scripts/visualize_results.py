"""Explain executed PCW results using seven data-analytics maxims.

1. Compare like with like: fixed splits, encoder, timestamps and decoding.
2. Show the denominator: class support and independent participant counts.
3. Expose uncertainty: cluster intervals and paired differences, not frame bootstrap.
4. Preserve honest scales: quality axes span 0..100%; differences center on zero.
5. Label the measurement: cached latency excludes CLIP and video decoding.
6. Avoid cherry-picking: timeline example follows video ID, never test score.
7. Explain limits: synthetic evidence, small N and completion semantics are visible.

Outputs high-resolution PNG/PDF figures plus a Markdown report with chart captions
and a CSV table. No hardcoded example measurements enter any numerical chart.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pcw.visualization.figures import generate
from pcw.visualization.report import report

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",default="results/gtea/matrix.json"); p.add_argument("--output",default="results/gtea")
    a=p.parse_args(); generate(a.input,Path(a.output)/"figures"); report(a.input,a.output); print(Path(a.output)/"report.md")
if __name__=="__main__": main()
