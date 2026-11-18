"""Generate a traceable descriptive report from executed matrix results."""
from pathlib import Path
import csv
from pcw.utils.io import read_json

def report(matrix_path,output):
    runs=read_json(matrix_path); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    synthetic=any(r["evidence_type"]=="synthetic" for r in runs)
    lines=["# Executed experiment report", "", "SYNTHETIC SOFTWARE VALIDATION ONLY. No empirical vision conclusions." if synthetic else "Real annotated video results. Interpret under the saved protocol and limitations.", "", "| Window (s) | Memory | Macro F1 | 95% cluster interval | Cached latency (ms) |", "|---|---|---|---|---|"]
    rows=[]
    for r in runs:
        c=r["config"]; stats=r["macro_f1"]; low,high=stats["ci95"]
        row=dict(run_id=r["run_id"],evidence_type=r["evidence_type"],window=c["window"],memory=c["memory"],macro_f1=stats["mean"],ci_low=low,ci_high=high,cached_latency_ms=r["mean_latency_ms"])
        rows.append(row); lines.append(f"| {c['window']} | {c['memory']} | {stats['mean']:.4f} | [{low:.4f}, {high:.4f}] | {r['mean_latency_ms']:.3f} |")
    lines.extend(["", "## Reading the figures", "", "![Recognition by context window](figures/accuracy_window.png)", "Shaded intervals resample independent participants/videos. Read small-sample uncertainty before comparing curves.", "", "![Paired changes](figures/paired_differences.png)", "The zero line means no observed difference from single frame. These exploratory contrasts do not establish equivalence or causation.", "", "![Quality and cost](figures/accuracy_cost.png)", "Higher and further left is preferable. Cost includes cached sequential inference only; markers are measured conditions.", "", "![Timeline](figures/timeline.png)", "This deterministically selected example exposes transitions and errors; it is not selected for a favorable score.", "", "![Confusion](figures/confusion.png)", "Read each row as the distribution of predictions for one true action. Row labels include observation support.", "", "Intervals resample independent participant clusters or videos, never observations. Cached latency excludes visual encoding. No end-to-end or real-time benchmark was executed.", "", "Remaining work: authorized dataset conversion, pretrained-encoder/GPU validation, repeated seeds, validation-selected decoding ablations, and predeclared real-data hypothesis tests."])
    (output/"report.md").write_text("\n".join(lines),encoding="utf-8")
    with (output/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
