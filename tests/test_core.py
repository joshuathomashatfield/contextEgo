import numpy as np
import pytest
import torch
from pcw.utils.io import config
from pcw.datasets.schema import labels_at,validate_procedure,validate_record,make_splits,validate_splits
from pcw.temporal.windows import window_indices,validate_features
from pcw.procedural.memory import Memory,Decoder
from pcw.encoders.cache import save_cache,load_cache,cache_key
from pcw.evaluation.metrics import classification,temporal_metrics,match_boundaries,bootstrap
from pcw.models.classifier import StepModel
from pcw.datasets.synthetic import create
from pcw.training.runner import run,sequence

def test_alignment():
    r={"segments":[{"start":1.,"end":2.,"step_id":0}]}
    assert labels_at(r,np.array([0.,1.,1.9,2.])).tolist()==[-1,0,0,-1]

def test_windows():
    t=np.array([0.,1.,2.,5.,10.])
    assert t[window_indices(t,5.,3.)].tolist()==[2.,5.]
    assert t[window_indices(t,4.,10.)].tolist()==[0.,1.,2.]
    assert t[window_indices(t,4.,0.)].tolist()==[2.]
    assert t[window_indices(t,0.,10.)].tolist()==[0.]
    with pytest.raises(ValueError): validate_features(np.array([1.,1.]),np.ones((2,3)))

def test_procedure_and_annotations():
    p=validate_procedure(dict(procedure_id="p",steps=[dict(id=0,name="a"),dict(id=1,name="b")],transitions=[[0,0],[0,1],[1,0]]))
    with pytest.raises(ValueError): validate_procedure(dict(p,transitions=[[0,3]]))
    r=dict(video_id="v",participant_id=None,procedure_id="p",video_path="v.mp4",duration_seconds=2.,segments=[dict(start=0.,end=1.5,step_id=0),dict(start=1.,end=2.,step_id=1)])
    with pytest.raises(ValueError): validate_record(r,p)

@pytest.mark.parametrize("mode",["none","previous","history"])
def test_memory(mode):
    m=Memory(3,mode,3)
    for label in [0,1,0,0]: m.update(label)
    assert len(m.history)==3 and len(m.vector())==m.dimension
    if mode=="history": assert np.isclose(m.vector()[3:6].sum(),1.)
    m.reset(); assert np.all(m.vector()==0)
    with pytest.raises(ValueError): m.update(3)

def test_decoders():
    d=Decoder("confirmation",2)
    assert [d.update(p) for p in [0,1,0,1,1]]==[0,0,0,0,1]
    d=Decoder("majority",3)
    assert [d.update(p) for p in [0,1,0]]==[0,1,0]

def test_cache(tmp_path):
    t=np.arange(3.); x=np.ones((3,4),np.float32); path=tmp_path/"f.npz"; meta={"encoder":"test"}
    save_cache(path,t,x,meta); a,b,m=load_cache(path,meta)
    assert np.array_equal(a,t) and np.array_equal(b,x)
    with pytest.raises(ValueError): load_cache(path,{"encoder":"other"})
    assert cache_key(meta)==cache_key(dict(meta))
    path.write_bytes(b"broken")
    with pytest.raises(Exception): load_cache(path)

def test_splits():
    records=[dict(video_id=str(i),participant_id=str(i//2)) for i in range(12)]
    split=make_splits(records); validate_splits(records,split)
    split["test"].append(split["train"][0])
    with pytest.raises(ValueError): validate_splits(records,split)

def test_metrics():
    y=np.array([0,0,1,1]); t=np.arange(4.)
    c=classification(y,y,2); assert c["accuracy"]==1 and c["macro_f1"]==1
    m=temporal_metrics(t,y,y,4.); assert m["segment_f1@0.5"]==1 and m["boundary"]["f1"]==1 and m["mean_delay"]==0
    m=temporal_metrics(t,y,np.array([0,0,0,1]),4.,1.)
    assert m["mean_delay"]==1 and m["false_transitions"]==0
    assert len(match_boundaries([(1.,0,1)],[(1.,0,1),(1.1,0,1)],1.))==1
    assert match_boundaries([(1.,0,1)],[(1.,1,0)],1.)==[]
    assert bootstrap([1.,1.],samples=10)["ci95"]==[1.,1.]

def test_unknown_censoring():
    m=temporal_metrics(np.arange(5.),np.array([0,0,-1,1,1]),np.array([0,0,2,1,1]),5.)
    assert m["false_transitions"]==0

@pytest.mark.parametrize("aggregation",["mean","gru"])
def test_causality_and_session_reset(aggregation):
    torch.manual_seed(1); c=config(memory="history",aggregation=aggregation); m=Memory(3,"history")
    model=StepModel(4,3,m.dimension,aggregation,8).eval(); t=np.arange(6.); x=np.ones((6,4),np.float32); y=np.zeros(6,dtype=int)
    first=sequence(model,t,x,y,c,3,torch.device("cpu"))
    changed=x.copy(); changed[-1]=100
    second=sequence(model,t,changed,y+2,c,3,torch.device("cpu"))
    assert first["prediction"][:-1]==second["prediction"][:-1]
    assert first["confidence"][:-1]==second["confidence"][:-1]
    assert first["prediction"]==sequence(model,t,x,y,c,3,torch.device("cpu"))["prediction"]
    assert model(torch.ones((1,4)),torch.zeros(m.dimension)).shape==(3,)

def test_config(tmp_path):
    p=tmp_path/"c.yaml"; p.write_text("window: 5\nmemory: history\n")
    assert config(p)["window"]==5
    p.write_text("typo: true")
    with pytest.raises(ValueError): config(p)
    with pytest.raises(ValueError): config(window=-1)

def test_end_to_end(tmp_path):
    create(tmp_path/"data")
    destination,result=run(tmp_path/"data",tmp_path/"results",config(epochs=1,memory="history",bootstrap_samples=20))
    assert result["evidence_type"]=="synthetic"
    assert (destination/"model.pt").exists() and len(result["per_video"])==2


def test_video_timestamps_and_export(tmp_path):
    import cv2
    from pcw.video.reader import frames
    from pcw.video.demo import render
    from pcw.utils.io import write_json
    video=tmp_path/"source.avi"
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),10.,(160,120))
    if not writer.isOpened(): pytest.skip("MJPG writer unavailable")
    for i in range(20): writer.write(np.full((120,160,3),i*10,dtype=np.uint8))
    writer.release()
    observations=list(frames(video,2.))
    times=np.array([t for t,x in observations]); assert np.all(np.diff(times)>0) and len(times)==4
    assert np.allclose(times,[0,.5,1.,1.5],atol=.05)
    prediction=tmp_path/"prediction.json"
    write_json(prediction,dict(video_path=str(video),timestamps=times.tolist(),prediction=[0,0,1,1],truth=[0,0,1,1],confidence=[.8]*4,duration=2.,config=config()))
    render(prediction,tmp_path/"demo.mp4",show_truth=True)
    capture=cv2.VideoCapture(str(tmp_path/"demo.mp4")); assert capture.isOpened(); assert capture.read()[0]; capture.release()

def test_nonfinite_config_and_annotations():
    with pytest.raises(ValueError): config(window=float("nan"))
    with pytest.raises(ValueError): config(bootstrap_samples=0)
    with pytest.raises(ValueError): config(unknown_setting=1)


def test_graph_allows_repeats_and_reports_deviation():
    from pcw.procedural.graph import ProcedureGraph
    graph=ProcedureGraph(dict(procedure_id="p",steps=[dict(id=i,name=str(i)) for i in range(3)],transitions=[[0,0],[0,1],[1,0]]))
    assert graph.expected(0,0) and graph.expected(1,0)
    assert not graph.expected(1,2)
