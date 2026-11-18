"""Adapter for inspected GTEA rectified videos and cleaned 71-action labels.

Collapse verb/object actions to ten verbs plus background; do not infer completion.
"""
import re
from pathlib import Path
import cv2
from pcw.datasets.schema import validate_record,validate_splits
from pcw.encoders.cache import fingerprint
from pcw.utils.io import write_json

VERBS=["background","close","fold","open","pour","put","scoop","shake","spread","stir","take"]
ACTION=re.compile(r"<([^>]+)><([^>]+)>\s*\((\d+)-(\d+)\)\s*\[([01])\]")
OBJECT=re.compile(r"<[^>]+>\s*(?:\(\d+-\d+\)|<\d+-\d+>)")

def parse_actions(path):
    actions=[]
    for number,line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(),1):
        line=line.strip()
        if not line: continue
        match=ACTION.fullmatch(line)
        if match:
            verb,objects,start,end,flag=match.groups()
            if verb not in VERBS[1:] or int(end)<int(start): raise ValueError(f"Invalid action at {path}:{number}")
            actions.append(dict(verb=verb,objects=objects,start_frame=int(start),end_frame=int(end),source_flag=int(flag)))
        elif not OBJECT.fullmatch(line): raise ValueError(f"Unrecognized annotation at {path}:{number}: {line}")
    if not actions: raise ValueError(f"No actions in {path}")
    return actions

def video_timing(path):
    cap=cv2.VideoCapture(str(path)); times=[]
    if not cap.isOpened(): raise ValueError(f"Cannot open {path}")
    fps=cap.get(cv2.CAP_PROP_FPS)
    if abs(fps-15.)>.01: cap.release(); raise ValueError("This adapter requires the published rectified 15-Hz edition")
    try:
        while True:
            ok,_=cap.read()
            if not ok: break
            t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
            if t<0 or (times and t<=times[-1]): raise ValueError("Video lacks strictly increasing PTS")
            times.append(t)
    finally: cap.release()
    if len(times)<2: raise ValueError("Video needs at least two frames")
    if abs(times[0])>1e-3 or any(abs((b-a)-1/15)>.002 for a,b in zip(times,times[1:])): raise ValueError("Rectified video PTS do not match annotation clock")
    return times[-1]+(times[-1]-times[-2]),len(times)

def to_segments(actions,duration,frame_base=1):
    """Inclusive source end -> half-open seconds; conflicts become unknown (-1)."""
    events={0.:[],duration:[]}
    for index,a in enumerate(actions):
        start=(a["start_frame"]-frame_base)/15.; end=(a["end_frame"]-frame_base+1)/15.
        if start<0 or end>duration+1/15+.001: raise ValueError("Annotation outside video clock")
        end=min(end,duration)
        events.setdefault(start,[]).append((index,VERBS.index(a["verb"]),True))
        events.setdefault(end,[]).append((index,VERBS.index(a["verb"]),False))
    times=sorted(events); active={}; out=[]; conflicts=0.
    for start,end in zip(times,times[1:]):
        for index,label,enter in events[start]:
            if enter: active[index]=label
            else: active.pop(index,None)
        labels=set(active.values()); label=0 if not labels else (next(iter(labels)) if len(labels)==1 else -1)
        if label==-1: conflicts+=end-start
        if out and out[-1]["step_id"]==label and out[-1]["end"]==start: out[-1]["end"]=end
        else: out.append(dict(start=start,end=end,step_id=label))
    return out,conflicts

def subject_split(records,test_subject):
    people=["S1","S2","S3","S4"]
    if test_subject not in people: raise ValueError("Test subject must be S1..S4")
    validation=people[(people.index(test_subject)+1)%4]
    split={"train":[],"validation":[],"test":[]}
    for r in records:
        name="test" if r["participant_id"]==test_subject else ("validation" if r["participant_id"]==validation else "train")
        split[name].append(r["video_id"])
    validate_splits(records,split); return split

def prepare(raw,output,frame_base=1):
    raw=Path(raw); output=Path(output)
    if frame_base not in [0,1]: raise ValueError("Frame base must be 0 or 1")
    videos={p.stem:p for p in (raw/"extracted").rglob("*") if p.suffix.lower() in [".mp4",".avi"] and re.fullmatch(r"S[1-4]_.+_C1",p.stem)}
    labels={p.stem:p for p in (raw/"extracted").rglob("*.txt") if re.fullmatch(r"S[1-4]_.+_C1",p.stem)}
    if len(videos)!=28 or set(videos)!=set(labels): raise ValueError("Require all 28 matching videos and cleaned label files; run download_gtea.py")
    procedure=dict(procedure_id="gtea_action_units",steps=[dict(id=i,name=name) for i,name in enumerate(VERBS)],semantic_scope="Verb action units within meal preparation, not unique recipe steps")
    records=[]; audits=[]
    for vid in sorted(videos):
        duration,count=video_timing(videos[vid]); actions=parse_actions(labels[vid]); segments,conflicts=to_segments(actions,duration,frame_base)
        r=dict(video_id=vid,participant_id=vid.split("_")[0],procedure_id=procedure["procedure_id"],recipe_id=vid.split("_")[1],video_path=str(videos[vid].resolve()),duration_seconds=duration,segments=segments)
        validate_record(r,procedure); records.append(r)
        audits.append(dict(video_id=vid,video_sha256=fingerprint(videos[vid]),annotation_sha256=fingerprint(labels[vid]),duration_seconds=duration,decoded_frames=count,source_actions=len(actions),conflicting_seconds=conflicts))
    for fold in range(1,5):
        dest=output/f"fold_{fold}"; write_json(dest/"manifest.json",records); write_json(dest/"procedure.json",procedure); write_json(dest/"splits.json",subject_split(records,f"S{fold}")); write_json(dest/"feature_store.json",dict(path="../features"))
        write_json(dest/"dataset_audit.json",dict(dataset="GTEA",edition="Rectified 15Hz video + cleaned 71 action labels, collapsed to verbs",frame_base=frame_base,end_inclusive=True,alignment_note="README specifies 15Hz but not index origin; chosen origin is an explicit assumption, configurable for sensitivity checks",overlap_policy="Distinct simultaneous verbs censored to -1; same-verb overlap merged; gaps background",source_flag_policy="Preserved in parser, unused; never interpreted as successful completion",protocol="Rotating held-out test subject; next subject validation; remaining two subjects training. Custom validation protocol, not published benchmark scores.",videos=audits))
    return records
