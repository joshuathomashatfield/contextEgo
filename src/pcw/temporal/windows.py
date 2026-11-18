"""Causal timestamp windows independent of observation count."""
import numpy as np

def validate_features(t,x):
    if t.ndim!=1 or x.ndim!=2 or len(t)!=len(x) or not len(t) or not np.isfinite(t).all() or not np.isfinite(x).all() or np.any(np.diff(t)<=0) or t[0]<0:
        raise ValueError("Expected finite embeddings and strictly increasing nonnegative timestamps")

def window_indices(t, now, seconds):
    if seconds < 0: raise ValueError("Negative window")
    right = np.searchsorted(t,now,side="right")
    if seconds == 0: return slice(max(0,right-1),right)
    return slice(np.searchsorted(t,now-seconds,side="left"),right)
