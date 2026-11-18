"""Run the full GTEA pipeline and automatically explain its results.

Downloads -> audited four-fold conversion -> shared CLIP extraction -> 13 conditions
per fold -> held-out subject aggregation -> descriptive figures and report.
"""
import argparse
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",default="data/raw/gtea"); p.add_argument("--data",default="data/processed/gtea"); p.add_argument("--output",default="results/gtea")
    p.add_argument("--device",default="auto"); p.add_argument("--epochs",type=int,default=5); p.add_argument("--seed",type=int,default=7); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--frame-base",type=int,choices=[0,1],default=1)
    p.add_argument("--skip-download",action="store_true"); p.add_argument("--skip-prepare",action="store_true"); p.add_argument("--skip-extract",action="store_true"); p.add_argument("--fold",choices=["all","1","2","3","4"],default="all")
    a=p.parse_args(); repository=Path(__file__).resolve().parents[1]
    if not a.skip_download: subprocess.run([sys.executable,str(repository/"scripts/download_gtea.py"),"--root",a.raw],check=True)
    if not a.skip_prepare:
        from pcw.datasets.gtea import prepare
        prepare(a.raw,a.data,a.frame_base)
    if not a.skip_extract:
        from pcw.cli import main as cli
        cli(["extract","--data",str(Path(a.data)/"fold_1"),"--device",a.device,"--batch-size",str(a.batch_size)])
    import torch
    torch.set_num_threads(1)  # Tiny per-observation heads avoid CPU thread overhead.
    from pcw.utils.io import config,write_json
    from pcw.training.runner import matrix
    from pcw.training.cross_validation import combine
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    matrices=[]
    for fold in ([1,2,3,4] if a.fold=="all" else [int(a.fold)]):
        print(f"Running GTEA test subject S{fold}",flush=True)
        result=matrix(Path(a.data)/f"fold_{fold}",output/f"fold_{fold}",config(epochs=a.epochs,seed=a.seed,device=a.device))
        matrices.append(result)
        write_json(output/"completed_folds.json",[dict(fold=i+1,run_ids=[r["run_id"] for r in runs]) for i,runs in enumerate(matrices)])
    if len(matrices)==4: combine(matrices,output)
    print("Results:",output.resolve(),flush=True)
if __name__=="__main__": main()
