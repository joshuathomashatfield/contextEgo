"""Normalized half-open annotations; gaps and unknown labels map to -1."""
from pathlib import Path
import math
import re
import numpy as np
from pcw.utils.io import read_json

def validate_procedure(p):
    ids = [s["id"] for s in p["steps"]]
    if not p.get("procedure_id") or ids != list(range(len(ids))) or not ids:
        raise ValueError("Step IDs must be contiguous from zero")
    for a,b in p.get("transitions", []):
        if a not in ids or b not in ids: raise ValueError("Unknown graph endpoint")
    return p

def validate_record(r, procedure):
    for k in ["video_id","participant_id","procedure_id","video_path","duration_seconds","segments"]:
        if k not in r: raise ValueError(f"Missing {k}")
    if not isinstance(r["video_id"],str) or not re.fullmatch(r"[A-Za-z0-9_.-]+",r["video_id"]) or r["video_id"] in [".",".."]: raise ValueError("Use a safe nonempty video ID")
    if r["procedure_id"] != procedure["procedure_id"] or not math.isfinite(r["duration_seconds"]) or r["duration_seconds"] <= 0: raise ValueError("Invalid procedure or duration")
    last = 0.
    for s in sorted(r["segments"], key=lambda s:s["start"]):
        if not 0 <= s["start"] < s["end"] <= r["duration_seconds"] or s["start"] < last: raise ValueError("Invalid or overlapping segments")
        if s["step_id"] not in [-1] + list(range(len(procedure["steps"]))): raise ValueError("Unknown step")
        last = s["end"]
    return r

def labels_at(record, timestamps):
    y = np.full(len(timestamps), -1, dtype=np.int64)
    for s in record["segments"]:
        y[(timestamps >= s["start"]) & (timestamps < s["end"])] = s["step_id"]
    return y

def load_manifest(path, procedure):
    records = read_json(path)
    if len({r["video_id"] for r in records}) != len(records): raise ValueError("Duplicate video IDs")
    for r in records:
        validate_record(r, procedure)
        r["video_path"] = str((Path(path).parent / r["video_path"]).resolve())
    return records

def make_splits(records, seed=7):
    rng = np.random.default_rng(seed)
    participants = all(r["participant_id"] is not None for r in records)
    key = "participant_id" if participants else "video_id"
    groups = sorted({r[key] for r in records}); rng.shuffle(groups)
    if len(groups) < 3: raise ValueError("At least three independent groups required")
    n = len(groups); a = max(1, int(.6*n)); b = max(a+1, int(.8*n)); b = min(b,n-1)
    out = {name:[r["video_id"] for r in records if r[key] in chunk] for name,chunk in zip(["train","validation","test"],[groups[:a],groups[a:b],groups[b:]])}
    validate_splits(records,out)
    return out

def validate_splits(records, splits):
    if set(splits) != {"train","validation","test"}: raise ValueError("Three splits required")
    lookup = {r["video_id"]:r for r in records}; seen=set(); people=set()
    for ids in splits.values():
        if not ids or len(ids)!=len(set(ids)) or seen.intersection(ids) or set(ids)-lookup.keys(): raise ValueError("Invalid video split")
        current = {lookup[i]["participant_id"] for i in ids} - {None}
        if current & people: raise ValueError("Participant leakage")
        people |= current; seen.update(ids)
    if seen != lookup.keys(): raise ValueError("Split must cover all videos")
