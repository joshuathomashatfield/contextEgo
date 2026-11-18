"""Content-addressed, validated feature archives without pickle."""
import hashlib
import json
from pathlib import Path
import numpy as np
from pcw.temporal.windows import validate_features

def fingerprint(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def cache_key(metadata): return hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()

def save_cache(path,t,x,metadata):
    validate_features(t,x)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    with temp.open("wb") as f: np.savez_compressed(f,timestamps=t,features=x,metadata=json.dumps(metadata,sort_keys=True))
    temp.replace(path)

def load_cache(path,expected=None):
    with np.load(path,allow_pickle=False) as z:
        t=z["timestamps"]; x=z["features"]; meta=json.loads(str(z["metadata"]))
    validate_features(t,x)
    if expected is not None and meta!=expected: raise ValueError("Stale feature cache")
    return t,x,meta


def feature_directory(root):
    """Resolve an explicitly configured shared feature store across folds."""
    root=Path(root); pointer=root/"feature_store.json"
    if pointer.exists(): return (root/json.loads(pointer.read_text(encoding="utf-8"))["path"]).resolve()
    return root/"features"
