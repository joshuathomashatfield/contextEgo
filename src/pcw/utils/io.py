"""Reproducible configuration and artifact utilities."""
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import yaml

DEFAULT = dict(seed=7, sample_hz=2., window=2., memory="none", aggregation="mean", hidden=32, epochs=5, learning_rate=.003, smoothing="none", smooth_size=3, history_size=16, boundary_tolerance=1., segment_thresholds=[.1,.25,.5], bootstrap_samples=1000, encoder="openai/clip-vit-base-patch32", device="cpu")
def config(path=None, **overrides):
    c = DEFAULT.copy()
    if path:
        supplied = yaml.safe_load(Path(path).read_text()) or {}
        if set(supplied) - set(c):
            raise ValueError("Unknown configuration keys")
        c.update(supplied)
    if set(overrides)-set(c): raise ValueError("Unknown configuration override")
    c.update({k:v for k,v in overrides.items() if v is not None})
    if not all(isinstance(c[k],(int,float)) and math.isfinite(c[k]) for k in ["window","sample_hz","learning_rate","boundary_tolerance"]): raise ValueError("Finite numeric settings required")
    if c["boundary_tolerance"] < 0 or c["bootstrap_samples"] < 1 or not c["segment_thresholds"] or any(not 0 < v <= 1 for v in c["segment_thresholds"]): raise ValueError("Invalid metric settings")
    if c["window"] < 0 or c["sample_hz"] <= 0 or c["epochs"] < 1 or c["hidden"] < 1 or c["learning_rate"] <= 0 or c["history_size"] < 1 or c["smooth_size"] < 1:
        raise ValueError("Invalid numeric configuration")
    for key, values in [("memory",["none","previous","history"]),("aggregation",["mean","gru"]),("smoothing",["none","majority","confirmation"])]:
        if c[key] not in values: raise ValueError(f"Invalid {key}")
    return c

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
