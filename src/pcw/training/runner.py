"""Sequential training and evaluation with model-generated memory only."""
import copy
import platform
import time
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4
import numpy as np
import torch
from pcw.datasets.schema import load_manifest, validate_procedure, validate_splits, labels_at
from pcw.encoders.cache import load_cache, fingerprint, feature_directory
from pcw.temporal.windows import window_indices
from pcw.procedural.memory import Memory, Decoder
from pcw.models.classifier import StepModel
from pcw.evaluation.metrics import classification,temporal_metrics,bootstrap
from pcw.utils.io import read_json,write_json,seed_all

def sync(device):
    if device.type=="cuda": torch.cuda.synchronize(device)

def sequence(model,t,x,y,c,k,device,optimizer=None):
    memory=Memory(k,c["memory"],c["history_size"]); decoder=Decoder(c["smoothing"],c["smooth_size"])
    predictions=[]; raw=[]; confidence=[]; latencies=[]; counts=[]; losses=[]
    for i,now in enumerate(t):
        sync(device); start=time.perf_counter()
        w=window_indices(t,now,c["window"]); counts.append(int(w.stop-w.start))
        seq=torch.as_tensor(x[w],dtype=torch.float32,device=device); mem=torch.as_tensor(memory.vector(),device=device)
        if optimizer is not None: optimizer.zero_grad()
        with torch.set_grad_enabled(optimizer is not None):
            logits=model(seq,mem); probabilities=logits.detach().softmax(-1); r=int(probabilities.argmax())
            p=decoder.update(r)
            if optimizer is not None and y[i]>=0:
                loss=torch.nn.functional.cross_entropy(logits.unsqueeze(0),torch.tensor([int(y[i])],device=device)); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        memory.update(p); sync(device)
        latencies.append(time.perf_counter()-start); predictions.append(p); raw.append(r); confidence.append(float(probabilities[p]))
    return dict(prediction=predictions,raw_prediction=raw,confidence=confidence,latency_seconds=latencies,frames_used=counts,loss=float(np.mean(losses)) if losses else None)

def run(root,output,c):
    seed_all(c["seed"]); root=Path(root); procedure=validate_procedure(read_json(root/"procedure.json")); records=load_manifest(root/"manifest.json",procedure); splits=read_json(root/"splits.json"); validate_splits(records,splits)
    k=len(procedure["steps"]); device=torch.device(c["device"] if c["device"]!="auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data={}; metadata={}; cache_hashes={}
    for r in records:
        path=feature_directory(root)/f"{r['video_id']}.npz"; t,x,meta=load_cache(path)
        if t[-1]>=r["duration_seconds"]: raise ValueError("Observation outside annotated duration")
        data[r["video_id"]]=(r,t,x,labels_at(r,t)); metadata[r["video_id"]]=meta; cache_hashes[r["video_id"]]=fingerprint(path)
    signatures={(v.get("encoder"),v.get("sample_hz"),v.get("kind")) for v in metadata.values()}
    if len(signatures)!=1: raise ValueError("Mixed feature encoders, sampling rates or evidence types")
    signature=next(iter(signatures))
    if signature[1]!=c["sample_hz"]: raise ValueError("Configured sampling rate differs from cache")
    if signature[2]!="synthetic" and signature[0]!=c["encoder"]: raise ValueError("Configured encoder differs from cache")
    train_x=np.concatenate([data[v][2] for v in splits["train"]]); mean=train_x.mean(0); std=np.maximum(train_x.std(0),1e-6)
    for v,(r,t,x,y) in data.items(): data[v]=(r,t,(x-mean)/std,y)
    dimensions={x.shape[1] for _,_,x,_ in data.values()}
    if len(dimensions)!=1: raise ValueError("Feature dimensions differ")
    mem=Memory(k,c["memory"],c["history_size"]); model=StepModel(len(mean),k,mem.dimension,c["aggregation"],c["hidden"]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=c["learning_rate"])
    start=time.perf_counter(); best=-1.; best_state=None; training=[]
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(c["epochs"]):
        model.train(); losses=[]
        for vid in splits["train"]:
            _,t,x,y=data[vid]; losses.append(sequence(model,t,x,y,c,k,device,optimizer)["loss"])
        model.eval(); scores=[]
        for vid in splits["validation"]:
            _,t,x,y=data[vid]; pred=sequence(model,t,x,y,c,k,device); scores.append(classification(y,pred["prediction"],k)["macro_f1"])
        score=float(np.mean(scores)); training.append(dict(epoch=epoch,validation_macro_f1=score,train_loss=losses))
        if score>best: best=score; best_state=copy.deepcopy(model.state_dict())
    train_seconds=time.perf_counter()-start; model.load_state_dict(best_state); model.eval()
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid4().hex[:8]; dest=Path(output)/run_id; dest.mkdir(parents=True)
    torch.save(dict(state=best_state,mean=mean.tolist(),std=std.tolist(),config=c,classes=k,features=len(mean)),dest/"model.pt")
    write_json(dest/"config.json",c); write_json(dest/"splits.json",splits); write_json(dest/"procedure.json",procedure)
    per_video=[]; pred_dir=dest/"predictions"; total_start=time.perf_counter()
    for vid in splits["test"]:
        r,t,x,y=data[vid]; pred=sequence(model,t,x,y,c,k,device)
        metrics=classification(y,pred["prediction"],k); metrics.update(temporal_metrics(t,y,np.array(pred["prediction"]),r["duration_seconds"],c["boundary_tolerance"],c["segment_thresholds"]))
        raw_metrics=temporal_metrics(t,y,np.array(pred["raw_prediction"]),r["duration_seconds"],c["boundary_tolerance"],c["segment_thresholds"])
        metrics["raw_temporal"]=raw_metrics
        metrics["decoded_fraction_changed"]=float(np.mean(np.array(pred["prediction"])!=np.array(pred["raw_prediction"])))
        metrics.update(video_id=vid,participant_id=r["participant_id"],classifier_seconds=sum(pred["latency_seconds"]),mean_frames_used=float(np.mean(pred["frames_used"])))
        per_video.append(metrics)
        write_json(pred_dir/f"{vid}.json",dict(video_id=vid,video_path=r["video_path"],classes=k,duration=r["duration_seconds"],timestamps=t.tolist(),truth=y.tolist(),config=c,**pred))
    elapsed=time.perf_counter()-total_start
    # Cluster by participant if available, otherwise independent video.
    clusters={}
    for v in per_video: clusters.setdefault(v["participant_id"] or v["video_id"],[]).append(v["macro_f1"])
    stats=bootstrap([np.mean(v) for v in clusters.values()],c["seed"],c["bootstrap_samples"])
    n=sum(len(data[v][1]) for v in splits["test"]); infer=sum(v["classifier_seconds"] for v in per_video)
    result=dict(run_id=run_id,evidence_type=signature[2],config=c,encoder=signature[0],per_video=per_video,macro_f1=stats,training=training,training_seconds=train_seconds,cache_evaluation_seconds=elapsed,classifier_seconds=infer,observations_per_second=n/max(infer,1e-12),mean_latency_ms=infer/max(1,n)*1000,peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor(),torch_version=torch.__version__,python=platform.python_version(),evaluation_observations=n,parameter_count=sum(p.numel() for p in model.parameters()),cache_hashes=cache_hashes,feature_extraction_seconds=sum(m.get("extraction_seconds",0.) for m in metadata.values()) if signature[2]!="synthetic" else None,end_to_end_seconds=None)
    write_json(dest/"metrics.json",result)
    return dest,result

def matrix(root,output,c):
    results=[]
    for window,memory in [(0.,"none")]+[(w,m) for w in [1.,2.,5.,10.] for m in ["none","previous","history"]]:
        condition=dict(c,window=window,memory=memory,smoothing="none")
        dest,result=run(root,output,condition); results.append(dict(path=str(dest.resolve()),**result))
    write_json(Path(output)/"matrix.json",results)
    # Paired differences aligned by video; resample participant clusters.
    paired=[]
    for reference in results:
        for candidate in results:
            if reference["config"]["memory"]!="none" or candidate["config"]["memory"]=="none": continue
            ref={v["video_id"]:v for v in reference["per_video"]}; groups={}
            for v in candidate["per_video"]:
                groups.setdefault(v["participant_id"] or v["video_id"],[]).append(v["macro_f1"]-ref[v["video_id"]]["macro_f1"])
            paired.append(dict(reference=reference["run_id"],candidate=candidate["run_id"],difference_macro_f1=bootstrap([np.mean(v) for v in groups.values()],c["seed"],c["bootstrap_samples"])))
    write_json(Path(output)/"paired_comparisons.json",paired)
    from pcw.visualization.figures import generate
    from pcw.visualization.report import report
    generate(Path(output)/"matrix.json",Path(output)/"figures")
    report(Path(output)/"matrix.json",output)
    return results

