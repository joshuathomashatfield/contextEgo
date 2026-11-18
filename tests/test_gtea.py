"""GTEA conversion contracts derived from the inspected cleaned annotation format."""
from pathlib import Path
import sys
import zipfile
import pytest
from pcw.datasets.gtea import parse_actions,to_segments,subject_split,VERBS
from pcw.datasets.schema import labels_at
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from download_gtea import extract

def test_action_rows_do_not_include_object_presence_or_completion(tmp_path):
    path=tmp_path/"labels.txt"
    path.write_text("<take><cup> (1-15) [0]\n<pour><water,cup> (16-30) [1]\n<cup> (1-30)\n<spoon> <1-30>\n")
    actions=parse_actions(path)
    assert len(actions)==2 and actions[1]["source_flag"]==1
    assert "completion" not in actions[1]
    segments,conflicts=to_segments(actions,3.,frame_base=1)
    record={"segments":segments}
    assert labels_at(record,np.array([0.,.99,1.,1.99,2.,3.])).tolist()==[VERBS.index("take"),VERBS.index("take"),VERBS.index("pour"),VERBS.index("pour"),0,-1]
    assert conflicts==0

def test_conflicting_overlaps_are_censored():
    actions=[dict(verb="take",start_frame=1,end_frame=15),dict(verb="pour",start_frame=10,end_frame=20)]
    segments,conflicts=to_segments(actions,2.)
    assert conflicts==pytest.approx(6/15)
    assert labels_at({"segments":segments},np.array([.2,.7,1.1,1.9])).tolist()==[VERBS.index("take"),-1,VERBS.index("pour"),0]

def test_four_fold_subject_isolation():
    records=[dict(video_id=f"S{s}_task{t}",participant_id=f"S{s}") for s in range(1,5) for t in range(7)]
    tests=[]
    for s in range(1,5):
        split=subject_split(records,f"S{s}")
        assert [len(split[k]) for k in ["train","validation","test"]]==[14,7,7]
        assert all(v.startswith(f"S{s}_") for v in split["test"])
        tests.extend(split["test"])
    assert len(tests)==len(set(tests))==28

def test_archive_path_traversal_rejected(tmp_path):
    path=tmp_path/"bad.zip"
    with zipfile.ZipFile(path,"w") as z: z.writestr("../outside.txt","bad")
    with pytest.raises(ValueError,match="Unsafe"): extract(path,tmp_path/"output")
    assert not (tmp_path/"outside.txt").exists()

def test_unknown_action_rejected(tmp_path):
    path=tmp_path/"labels.txt"; path.write_text("<invented><cup> (1-15) [0]")
    with pytest.raises(ValueError): parse_actions(path)
