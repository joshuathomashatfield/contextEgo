"""Offline research commands. See README for normalized data requirements."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from pcw.utils.io import config,read_json,write_json

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["synthetic","prepare","inspect","extract","train","matrix","figures","demo","evaluate","report"])
    parser.add_argument("--data",default="data/processed/synthetic"); parser.add_argument("--output",default="results"); parser.add_argument("--config"); parser.add_argument("--input"); parser.add_argument("--procedure"); parser.add_argument("--seed",type=int); parser.add_argument("--epochs",type=int); parser.add_argument("--window",type=float); parser.add_argument("--memory",choices=["none","previous","history"]); parser.add_argument("--aggregation",choices=["mean","gru"]); parser.add_argument("--smoothing",choices=["none","majority","confirmation"]); parser.add_argument("--device"); parser.add_argument("--batch-size",type=int,default=32); parser.add_argument("--show-truth",action="store_true")
    args=parser.parse_args(argv); c=config(args.config,**{k:getattr(args,k) for k in ["seed","epochs","window","memory","aggregation","smoothing","device"]})
    root=Path(args.data)
    if args.command=="synthetic":
        from pcw.datasets.synthetic import create
        create(root,c["seed"]); print(root)
    elif args.command=="inspect":
        path=Path(args.input or args.data)
        if path.is_dir(): print("\n".join(str(p.relative_to(path)) for p in sorted(path.rglob("*")) if p.is_file())[:20000])
        else:
            value=read_json(path); print(json.dumps(value,indent=2)[:20000])
    elif args.command=="prepare":
        from pcw.datasets.schema import validate_procedure,load_manifest,make_splits
        if not args.input or not args.procedure: parser.error("prepare requires --input normalized.json --procedure procedure.json")
        procedure=validate_procedure(read_json(args.procedure)); records=load_manifest(args.input,procedure)
        root.mkdir(parents=True,exist_ok=True); write_json(root/"manifest.json",records); write_json(root/"procedure.json",procedure); write_json(root/"splits.json",make_splits(records,c["seed"]))
    elif args.command=="extract":
        from pcw.datasets.schema import load_manifest,validate_procedure
        from pcw.video.reader import frames
        from pcw.encoders.clip import ClipEncoder
        from pcw.encoders.cache import fingerprint,save_cache,load_cache,feature_directory
        if args.batch_size<1: parser.error("batch-size must be positive")
        records=load_manifest(root/"manifest.json",validate_procedure(read_json(root/"procedure.json"))); encoder=None
        for r in records:
            meta=dict(source_sha256=fingerprint(r["video_path"]),encoder=c["encoder"],sample_hz=c["sample_hz"],kind="real",preprocessing="CLIPProcessor-default-rgb-v1")
            dest=feature_directory(root)/f"{r['video_id']}.npz"
            if dest.exists():
                _,_,old=load_cache(dest)
                if all(old.get(k)==v for k,v in meta.items()): print("cache hit",r["video_id"]); continue
            start=time.perf_counter()
            if encoder is None: encoder=ClipEncoder(c["encoder"],c["device"])
            times=[]; batches=[]; images=[]
            for t,image in frames(r["video_path"],c["sample_hz"]):
                if t>=r["duration_seconds"]: break
                times.append(t); images.append(image)
                if len(images)==args.batch_size: batches.append(encoder.encode(images)); images=[]
            if images: batches.append(encoder.encode(images))
            if not batches: raise ValueError("No video observations")
            meta["extraction_seconds"]=time.perf_counter()-start; save_cache(dest,np.array(times),np.concatenate(batches),meta); print(dest)
    elif args.command in ["train","matrix"]:
        from pcw.training.runner import run,matrix
        if args.command=="train": print(run(root,args.output,c)[0])
        else: print(f"Completed {len(matrix(root,args.output,c))} conditions")
    elif args.command=="figures":
        from pcw.visualization.figures import generate
        generate(args.input or str(Path(args.output)/"matrix.json"),Path(args.output)/"figures")
    elif args.command=="report":
        from pcw.visualization.report import report
        report(args.input or str(Path(args.output)/"matrix.json"),args.output)
    elif args.command=="demo":
        from pcw.video.demo import render
        if not args.input: parser.error("demo requires --input prediction.json")
        render(args.input,args.output,show_truth=args.show_truth)
    elif args.command=="evaluate":
        from pcw.evaluation.metrics import classification,temporal_metrics
        if not args.input: parser.error("evaluate requires --input saved_prediction.json")
        pred=read_json(args.input); y=np.array(pred["truth"]); p=np.array(pred["prediction"]); t=np.array(pred["timestamps"])
        k=pred.get("classes",max(max(p),max(y))+1)
        result=classification(y,p,k); result.update(temporal_metrics(t,y,p,pred["duration"],c["boundary_tolerance"],c["segment_thresholds"])); write_json(Path(args.output)/"recomputed_metrics.json",result)

if __name__=="__main__": main()
