"""Timestamp-synchronized saved-prediction video export."""
from pathlib import Path
import cv2
import numpy as np
from pcw.utils.io import read_json

def render(prediction_path,output,fps=15,show_truth=False):
    pred=read_json(prediction_path); cap=cv2.VideoCapture(pred["video_path"])
    if not cap.isOpened(): raise ValueError("Original video unavailable; synthetic features have no source video")
    if fps<=0: raise ValueError("Positive output fps required")
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height+120))
    if not writer.isOpened(): cap.release(); raise ValueError("Video writer unavailable")
    times=np.array(pred["timestamps"]); c=pred["config"]; previous=-1.; output_index=0
    def draw(frame,now):
        idx=int(np.searchsorted(times,now,side="right"))-1
        canvas=np.zeros((height+120,width,3),dtype=np.uint8); canvas[:height]=frame
        text="Awaiting first prediction" if idx<0 else f"Step {pred['prediction'][idx]} confidence {pred['confidence'][idx]:.2f}"
        cv2.putText(canvas,text,(8,height+22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.putText(canvas,f"t={now:.2f}s W={c['window']}s memory={c['memory']}",(8,height+45),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        for row,key in enumerate(["prediction","truth"] if show_truth else ["prediction"]):
            for j,label in enumerate(pred[key]):
                if times[j]>now: break
                a=int(width*times[j]/pred["duration"]); b=int(width*min(now,times[j+1] if j+1<len(times) else pred["duration"])/pred["duration"])
                color=(90,90,90) if label<0 else tuple(int(v) for v in np.random.default_rng(label).integers(60,255,3))
                cv2.rectangle(canvas,(a,height+60+row*22),(b,height+77+row*22),color,-1)
        return canvas
    try:
        last_frame=None; last_time=0.
        while True:
            ok,frame=cap.read()
            if not ok: break
            now=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
            if now<=previous: raise ValueError("Missing monotonic video timestamps")
            previous=now
            # Hold the most recent available source frame for constant-rate export.
            while last_frame is not None and output_index/fps<now:
                writer.write(draw(last_frame,output_index/fps)); output_index+=1
            last_frame=frame; last_time=now
        if last_frame is not None: writer.write(draw(last_frame,last_time))
    finally: cap.release(); writer.release()
