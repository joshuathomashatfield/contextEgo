"""Download GTEA rectified 15-fps videos and published action labels.

URLs were recovered from the archived official Georgia Tech dataset page.
This is the original 28-video GTEA, not GTEA Gaze or EGTEA Gaze+.
"""
import argparse
import hashlib
import json
import shutil
import stat
import time
import urllib.request
import zipfile
from pathlib import Path

SOURCES={
 "gtea_labels_71.zip":"https://www.dropbox.com/s/yb6x4ez10bhx8sb/gtea_labels_71.zip?dl=1",
 "gtea_videos.zip":"https://www.dropbox.com/s/xzdw0mloohtxjn5/gtea_videos.zip?dl=1",
}
PROVENANCE="https://web.archive.org/web/20240000000000id_/http://cbs.ic.gatech.edu/fpv/"

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def fetch(url,path,timeout=60,retries=2):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and zipfile.is_zipfile(path): return dict(status="cached",bytes=path.stat().st_size,sha256=sha256(path))
    partial=path.with_suffix(path.suffix+".part")
    for attempt in range(retries+1):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            headers={"User-Agent":"PCW-GTEA-research/0.1"}
            if offset: headers["Range"]=f"bytes={offset}-"
            request=urllib.request.Request(url,headers=headers)
            with urllib.request.urlopen(request,timeout=timeout) as response:
                resume=response.status==206
                if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"): raise ValueError("Invalid resume offset")
                expected=response.headers.get("Content-Length"); received=0; last=time.monotonic()
                with partial.open("ab" if resume else "wb") as target:
                    while True:
                        chunk=response.read(64*1024)
                        if not chunk: break
                        target.write(chunk); received+=len(chunk)
                        if time.monotonic()-last>10:
                            print(f"{path.name}: {(received+(offset if resume else 0))/1024**2:.1f} MiB",flush=True); last=time.monotonic()
                if expected and received!=int(expected): raise ValueError("Incomplete response")
            if not zipfile.is_zipfile(partial): raise ValueError("Server returned a non-ZIP response; link may be unavailable")
            with zipfile.ZipFile(partial) as z:
                bad=z.testzip()
                if bad: raise ValueError(f"ZIP CRC failure: {bad}")
            partial.replace(path)
            return dict(status="downloaded",bytes=path.stat().st_size,sha256=sha256(path))
        except Exception:
            if attempt==retries: raise
            print(f"Retry {attempt+1}: {path.name}",flush=True)
    raise RuntimeError("Unreachable")

def extract(path,destination):
    """Extract only regular members within the named destination."""
    destination=Path(destination).resolve(); destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            relative=Path(member.filename.replace("\\","/"))
            target=(destination/relative).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr>>16): raise ValueError("Unsafe archive member")
            if member.is_dir(): target.mkdir(parents=True,exist_ok=True); continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.stat().st_size==member.file_size: continue
            with archive.open(member) as source,target.open("wb") as out: shutil.copyfileobj(source,out)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",default="data/raw/gtea")
    p.add_argument("--labels-only",action="store_true"); p.add_argument("--timeout",type=float,default=60.); p.add_argument("--retries",type=int,default=2)
    a=p.parse_args(); root=Path(a.root); root.mkdir(parents=True,exist_ok=True); report=[]
    for name,url in SOURCES.items():
        if a.labels_only and "videos" in name: continue
        print("Fetching",name,flush=True)
        try:
            result=fetch(url,root/"archives"/name,a.timeout,a.retries)
            extract(root/"archives"/name,root/"extracted")
            report.append(dict(file=name,url=url,provenance=PROVENANCE,**result)); print(name,result["status"],flush=True)
        except Exception as error:
            report.append(dict(file=name,url=url,status="failed",error=str(error),provenance=PROVENANCE)); print(name,"FAILED",error,flush=True)
        (root/"download_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if any(r["status"]=="failed" for r in report): raise SystemExit("Download incomplete. Inspect download_report.json; rerun to resume.")

if __name__=="__main__": main()
