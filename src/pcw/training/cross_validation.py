"""Combine disjoint held-out GTEA subjects without treating frames as replicates."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from pcw.evaluation.metrics import bootstrap
from pcw.utils.io import write_json

def combine(fold_matrices,output):
    if len(fold_matrices)!=4: raise ValueError("Require all four GTEA test-subject folds")
    combined=[]
    for index in range(13):
        group=[fold[index] for fold in fold_matrices]
        configurations=[r["config"] for r in group]
        if any(c!=configurations[0] for c in configurations): raise ValueError("Fold configurations differ")
        evidence={r["evidence_type"] for r in group}
        if len(evidence)!=1: raise ValueError("Mixed evidence")
        videos=[v for r in group for v in r["per_video"]]
        if len({v["video_id"] for v in videos})!=len(videos): raise ValueError("Duplicate out-of-fold test video")
        subjects={}
        for fold_index,r in enumerate(group):
            for v in r["per_video"]:
                person=v["participant_id"]
                if person in subjects and subjects[person]!=fold_index: raise ValueError("Test participant occurs in multiple folds")
                subjects[person]=fold_index
        clusters=defaultdict(list)
        for v in videos: clusters[v["participant_id"]].append(v["macro_f1"])
        stats=bootstrap([np.mean(v) for v in clusters.values()],configurations[0]["seed"],configurations[0]["bootstrap_samples"])
        observations=sum(r["evaluation_observations"] for r in group); seconds=sum(r["classifier_seconds"] for r in group)
        merged=dict(group[0]); merged.update(run_id=f"out_of_fold_{index}",per_video=videos,macro_f1=stats,fold_run_ids=[r["run_id"] for r in group],evaluation_observations=observations,classifier_seconds=seconds,mean_latency_ms=1000*seconds/observations,training_seconds=sum(r["training_seconds"] for r in group),prediction_paths=[str(p.resolve()) for r in group for p in sorted((Path(r["path"])/"predictions").glob("*.json"))],protocol="Four rotating test subjects; one separate validation subject; two training subjects. Shared fitted models within each fold induce additional dependence not captured by the descriptive subject bootstrap.")
        combined.append(merged)
    output=Path(output); write_json(output/"matrix.json",combined)
    from pcw.visualization.figures import generate
    from pcw.visualization.report import report
    generate(output/"matrix.json",output/"figures"); report(output/"matrix.json",output)
    return combined
