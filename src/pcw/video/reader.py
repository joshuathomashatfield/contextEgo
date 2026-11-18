"""Decode actual presentation timestamps; never infer time from native FPS."""
from pathlib import Path
import cv2

def frames(path, sample_hz=2., resize=None):
    if sample_hz<=0: raise ValueError("sample_hz must be positive")
    cap=cv2.VideoCapture(str(Path(path)))
    if not cap.isOpened(): raise ValueError(f"Cannot open video: {path}")
    previous=-1.; next_time=0.
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
            if t<0 or t<=previous: raise ValueError("Backend lacks monotonic presentation timestamps; remux or use a PTS-aware backend")
            previous=t
            if t+1e-9<next_time: continue
            if resize: frame=cv2.resize(frame,resize)
            yield t,cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            next_time=(int(t*sample_hz)+1)/sample_hz
    finally: cap.release()
