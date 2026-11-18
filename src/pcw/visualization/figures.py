"""Descriptive analytics: comparable scales, explicit uncertainty, honest denominators.

Charts expose independent-unit counts and cached timing scope. Selection of timeline
examples is deterministic, never based on favorable test performance.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
from pcw.utils.io import read_json
from pcw.evaluation.metrics import segments,boundaries,match_boundaries

COLORS={"none":"#0072B2","previous":"#D55E00","history":"#009E73"}
NAMES={"none":"Visual only","previous":"Previous predicted step","history":"Predicted history"}

def save(fig,out,name):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f"{name}.png",dpi=300,bbox_inches="tight",facecolor="white")
    fig.savefig(out/f"{name}.pdf",bbox_inches="tight",facecolor="white"); plt.close(fig)

def finish(fig,ax,title,note):
    ax.set_title(title,loc="left",fontweight="bold",pad=16)
    ax.grid(axis="y",alpha=.18); ax.set_axisbelow(True)
    fig.text(.01,.01,note,ha="left",va="bottom",fontsize=8,color="#444444",wrap=True)
    fig.tight_layout(rect=(0,.12,1,1))

def prediction_files(run):
    if run.get("prediction_paths"): return [Path(p) for p in run["prediction_paths"]]
    return sorted((Path(run["path"])/"predictions").glob("*.json"))

def timeline(ax,pred,names=None):
    names=names or [f"Step {i}" for i in range(pred.get("classes",max(pred["prediction"])+1))]
    colors=["#cccccc" if name=="background" else plt.get_cmap("tab20")(i%20) for i,name in enumerate(names)]
    t=np.asarray(pred["timestamps"]); y=np.asarray(pred["truth"]); p=np.asarray(pred["prediction"])
    for row,values in enumerate([y,p]):
        for a,b,label in segments(t,values,pred["duration"]):
            ax.broken_barh([(a,b-a)],(row-.28,.56),facecolors="#ffffff" if label<0 else colors[label],edgecolors="white",linewidth=.2)
    masked=np.where(y<0,-1,p)
    tb=boundaries(t,y); pb=boundaries(t,masked); pairs=match_boundaries(tb,pb,pred["config"]["boundary_tolerance"])
    matched_t={i for i,j in pairs}; matched_p={j for i,j in pairs}
    for i,b in enumerate(tb):
        if i not in matched_t: ax.plot(b[0],-.4,marker="v",color="#CC3311",markersize=5)
    for j,b in enumerate(pb):
        if j not in matched_p: ax.plot(b[0],1.4,"x",color="#CC3311",markersize=5)
    for i,j in pairs:
        if pb[j][0]>tb[i][0]: ax.annotate("",xy=(pb[j][0],1.65),xytext=(tb[i][0],1.65),arrowprops=dict(arrowstyle="->",color="#D55E00"))
    ax.set_yticks([0,1],["Ground truth","Prediction"]); ax.set_xlabel("Video time (seconds)"); ax.set_xlim(0,pred["duration"]); ax.set_ylim(-.65,1.9)
    handles=[Patch(facecolor=color,label=name) for color,name in zip(colors,names)]+[Patch(facecolor="white",edgecolor="gray",label="Unknown / censored")]
    ax.legend(handles=handles,ncol=min(6,len(handles)),loc="upper center",bbox_to_anchor=(.5,-.22),fontsize=8,frameon=False)

def generate(matrix_path,out):
    runs=read_json(matrix_path)
    if not runs: raise ValueError("No executed runs to visualize")
    kinds={r["evidence_type"] for r in runs}
    if len(kinds)!=1: raise ValueError("Never combine synthetic and real evidence")
    prefix="SYNTHETIC CHECK | " if kinds=={"synthetic"} else ""
    units=runs[0]["macro_f1"]["units"]
    note=f"95% percentile bootstrap intervals; n={units} independent participant/video units. "
    note+=("One unit: uncertainty cannot be estimated meaningfully." if units<2 else "Small samples limit generalization; intervals are not equivalence tests.")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,ax=plt.subplots(figsize=(12,3)); ax.axis("off")
    boxes=["Video + timestamps","Frozen CLIP + cache","Past-only window","Visual + memory head","Optional decoder","Prediction + metrics"]
    for i,label in enumerate(boxes):
        ax.text((i+.5)/6,.6,label,ha="center",va="center",bbox=dict(boxstyle="round,pad=.6",fc="#e6edf7",ec="#7799bb"),transform=ax.transAxes,fontsize=8)
        if i<5: ax.annotate("",xy=((i+1)/6+.01,.6),xytext=((i+1)/6-.01,.6),xycoords="axes fraction",arrowprops=dict(arrowstyle="->"))
    ax.annotate("Decoded predictions update bounded memory; reset between videos",xy=(.57,.38),xytext=(.7,.12),xycoords="axes fraction",ha="center",arrowprops=dict(arrowstyle="->"),fontsize=9)
    save(fig,out,"architecture")
    fig,ax=plt.subplots(figsize=(9,5.5))
    for memory in COLORS:
        group=sorted([r for r in runs if r["config"]["memory"]==memory and r["config"]["window"]>0],key=lambda r:r["config"]["window"])
        if not group: continue
        x=[r["config"]["window"] for r in group]; y=[r["macro_f1"]["mean"] for r in group]
        ax.plot(x,y,"o-",label=NAMES[memory],color=COLORS[memory],lw=2)
        if units>1: ax.fill_between(x,[r["macro_f1"]["ci95"][0] for r in group],[r["macro_f1"]["ci95"][1] for r in group],color=COLORS[memory],alpha=.1)
    baseline=next(r for r in runs if r["config"]["window"]==0)
    ax.axhline(baseline["macro_f1"]["mean"],ls="--",color="#555555",label="Single-frame baseline")
    ax.set(xlabel="Past visual context (seconds)",ylabel="Participant mean macro F1",ylim=(0,1),xticks=[1,2,5,10]); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False,loc="best"); finish(fig,ax,prefix+"Does more context improve recognition?",note); save(fig,out,"accuracy_window")
    fig,ax=plt.subplots(figsize=(9,5.5)); markers={0.:"X",1.:"o",2.:"s",5.:"^",10.:"D"}
    for r in runs:
        c=r["config"]; ax.scatter(r["mean_latency_ms"],r["macro_f1"]["mean"],c=COLORS[c["memory"]],marker=markers[c["window"]],s=70)
        ax.annotate(f"{c['window']:g}s",(r["mean_latency_ms"],r["macro_f1"]["mean"]),xytext=(4,5),textcoords="offset points",fontsize=8)
    ax.set(xlabel="Cached sequential inference (milliseconds per observation)",ylabel="Participant mean macro F1",ylim=(0,1),xlim=(0,None)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(handles=[Patch(color=COLORS[m],label=NAMES[m]) for m in COLORS],frameon=False)
    finish(fig,ax,prefix+"Recognition quality versus inference cost","Toward the upper left is preferable. Labels show window duration. Timings exclude video decoding and CLIP extraction.
Latency is one measured run, not a repeated hardware benchmark; no real-time claim follows."); save(fig,out,"accuracy_cost")
    chosen=baseline; files=sorted(prediction_files(chosen)); pred=read_json(files[0]); procedure=read_json(Path(chosen["path"])/"procedure.json"); names=[s["name"] for s in procedure["steps"]]
    fig,ax=plt.subplots(figsize=(12,5)); timeline(ax,pred,names)
    ax.set_title(prefix+f"Where does the single-frame baseline fail?  {pred['video_id']}",loc="left",fontweight="bold")
    fig.text(.01,.01,"Deterministic example: first test video by ID. Red triangle = missed true transition; red x = false transition; orange arrow = matched delay.
Matching uses the saved tolerance and directed labels. A transition does not verify successful task completion.",fontsize=8)
    fig.tight_layout(rect=(0,.24,1,1)); save(fig,out,"timeline")
    cm=np.sum([v["confusion_matrix"] for v in chosen["per_video"]],axis=0); support=cm.sum(1); normalized=cm/np.maximum(1,support[:,None])
    fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(normalized,cmap="Blues",vmin=0,vmax=1); cb=fig.colorbar(im,ax=ax); cb.ax.yaxis.set_major_formatter(PercentFormatter(1)); cb.set_label("Share of true-class observations")
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i,j]: ax.text(j,i,f"{normalized[i,j]:.0%}",ha="center",va="center",fontsize=7,color="white" if normalized[i,j]>.55 else "black")
    ax.set_xticks(range(len(names)),names,rotation=45,ha="right"); ax.set_yticks(range(len(names)),[f"{name} (n={n})" for name,n in zip(names,support)]); ax.set(xlabel="Predicted action",ylabel="True action (observation support)")
    finish(fig,ax,prefix+"Which actions are confused?  Single-frame baseline","Rows normalize to 100% where support is nonzero. Blank rows have no labeled observations. Macro F1 also includes absent classes.
Counts are observations, not independent statistical units. All confusion matrices are pooled from held-out predictions."); save(fig,out,"confusion")
    fig,ax=plt.subplots(figsize=(9,6)); baseline_by_video={v["video_id"]:v for v in baseline["per_video"]}; labels=[]; means=[]; lows=[]; highs=[]; colors=[]
    from pcw.evaluation.metrics import bootstrap
    for r in sorted([r for r in runs if r["config"]["memory"]!="none"],key=lambda r:(r["config"]["window"],r["config"]["memory"])):
        groups={}
        for v in r["per_video"]: groups.setdefault(v["participant_id"] or v["video_id"],[]).append(v["macro_f1"]-baseline_by_video[v["video_id"]]["macro_f1"])
        stats=bootstrap([np.mean(v) for v in groups.values()],r["config"]["seed"],r["config"]["bootstrap_samples"])
        labels.append(f"{r['config']['window']:g}s | {NAMES[r['config']['memory']]}"); means.append(stats["mean"]); lows.append(stats["mean"]-stats["ci95"][0]); highs.append(stats["ci95"][1]-stats["mean"]); colors.append(COLORS[r["config"]["memory"]])
    for i in range(len(labels)): ax.errorbar(means[i],i,xerr=[[lows[i]],[highs[i]]],fmt="o",color=colors[i],capsize=3)
    ax.axvline(0,color="#555555",ls="--"); ax.set_yticks(range(len(labels)),labels); ax.invert_yaxis(); ax.set_xlabel("Paired macro F1 difference from single frame (percentage points)"); ax.xaxis.set_major_formatter(PercentFormatter(1))
    finish(fig,ax,prefix+"How much changes relative to single frame?",note+" Positive values favor the context model; contrasts are exploratory."); save(fig,out,"paired_differences")
