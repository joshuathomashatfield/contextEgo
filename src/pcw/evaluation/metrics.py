"""One-to-one segment and boundary matching with explicit censoring."""
import numpy as np

def classification(y,p,k):
    cm=np.zeros((k,k),dtype=int)
    for a,b in zip(y,p):
        if a>=0: cm[int(a),int(b)]+=1
    tp=cm.diagonal(); precision=tp/np.maximum(1,cm.sum(0)); recall=tp/np.maximum(1,cm.sum(1))
    f1=2*precision*recall/np.maximum(1e-12,precision+recall)
    return dict(accuracy=float(tp.sum()/max(1,cm.sum())),macro_f1=float(f1.mean()),precision=precision.tolist(),recall=recall.tolist(),confusion_matrix=cm.tolist(),labeled_observations=int(cm.sum()))

def segments(t,y,end):
    out=[]
    for i,label in enumerate(y):
        right=float(t[i+1]) if i+1<len(t) else float(end)
        if right<=t[i]: continue
        if out and out[-1][2]==int(label) and out[-1][1]==float(t[i]): out[-1][1]=right
        else: out.append([float(t[i]),right,int(label)])
    return out

def boundaries(t,y):
    return [(float(t[i]),int(y[i-1]),int(y[i])) for i in range(1,len(t)) if y[i]!=y[i-1] and y[i]>=0 and y[i-1]>=0]

def match_boundaries(truth,pred,tolerance):
    # Nearest globally available pair first; require both transition labels.
    candidates=sorted((abs(a[0]-b[0]),i,j) for i,a in enumerate(truth) for j,b in enumerate(pred) if a[1:]==b[1:] and abs(a[0]-b[0])<=tolerance)
    used_t=set(); used_p=set(); pairs=[]
    for _,i,j in candidates:
        if i not in used_t and j not in used_p: used_t.add(i); used_p.add(j); pairs.append((i,j))
    return pairs

def prf(tp,n_pred,n_true):
    p=tp/max(1,n_pred); r=tp/max(1,n_true)
    return dict(precision=p,recall=r,f1=2*p*r/max(1e-12,p+r))

def temporal_metrics(t,y,p,end,tolerance=1.,thresholds=(.1,.25,.5)):
    # Unknown intervals break runs and censor predictions in those intervals.
    masked=np.where(y<0,-1,p)
    truth=[s for s in segments(t,y,end) if s[2]>=0]
    predicted=[s for s in segments(t,masked,end) if s[2]>=0]
    result={}
    for threshold in thresholds:
        candidates=[]
        for i,a in enumerate(truth):
            for j,b in enumerate(predicted):
                intersection=max(0,min(a[1],b[1])-max(a[0],b[0])); union=max(a[1],b[1])-min(a[0],b[0])
                iou=intersection/max(1e-12,union)
                if a[2]==b[2] and iou>=threshold: candidates.append((-iou,i,j))
        used_t=set(); used_p=set()
        for _,i,j in sorted(candidates):
            if i not in used_t and j not in used_p: used_t.add(i); used_p.add(j)
        result[f"segment_f1@{threshold:g}"]=prf(len(used_t),len(predicted),len(truth))["f1"]
    tb=boundaries(t,y); pb=boundaries(t,masked); pairs=match_boundaries(tb,pb,tolerance)
    result["boundary"]=prf(len(pairs),len(pb),len(tb))
    delays=[pb[j][0]-tb[i][0] for i,j in pairs]
    # Delays are signed and conditional on tolerance-matched transitions.
    result.update(mean_delay=float(np.mean(delays)) if delays else None,median_delay=float(np.median(delays)) if delays else None,matched_transitions=len(pairs),missed_transitions=len(tb)-len(pairs),false_transitions=len(pb)-len(pairs))
    labeled_duration=sum(s[1]-s[0] for s in truth)
    result["false_transitions_per_minute"]=(len(pb)-len(pairs))*60/max(1e-12,labeled_duration)
    result["prediction_changes"]=len(pb)
    return result

def bootstrap(values,seed=7,samples=1000):
    values=np.asarray(values,dtype=float)
    if not len(values): raise ValueError("No independent units")
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(values,size=len(values),replace=True).mean() for _ in range(samples)])
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),units=len(values))
