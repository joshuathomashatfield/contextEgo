"""Features only: not a simulated empirical computer vision benchmark."""
from pathlib import Path
import numpy as np
from pcw.encoders.cache import save_cache
from pcw.utils.io import write_json
from pcw.datasets.schema import make_splits

def create(root,seed=7):
    root=Path(root); rng=np.random.default_rng(seed); records=[]
    procedure=dict(procedure_id="fixture",steps=[dict(id=i,name=f"fixture_step_{i}") for i in range(3)],transitions=[[0,0],[0,1],[1,1],[1,2],[2,2],[2,0],[1,0]])
    centers=rng.normal(size=(3,8)).astype(np.float32)
    for i in range(9):
        t=np.arange(0,18,.5); order=[0,1,2] if i%3 else [0,1,0]
        y=np.array([order[min(2,int(v//6))] for v in t]); x=centers[y]+rng.normal(0,.7,(len(t),8)).astype(np.float32)
        vid=f"synthetic_{i}"; r=dict(video_id=vid,participant_id=f"fixture_person_{i}",procedure_id="fixture",video_path=f"unavailable/{vid}.mp4",duration_seconds=18.,segments=[dict(start=j*6.,end=(j+1)*6.,step_id=s) for j,s in enumerate(order)])
        records.append(r); save_cache(root/"features"/f"{vid}.npz",t,x,dict(kind="synthetic",encoder="synthetic-fixture",sample_hz=2.,seed=seed))
    write_json(root/"manifest.json",records); write_json(root/"procedure.json",procedure); write_json(root/"splits.json",make_splits(records,seed))
