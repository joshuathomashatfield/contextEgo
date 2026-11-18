"""Inspect remote metadata to locate existing GTEA sources; no guessed media URLs."""
import json
import urllib.request
from pathlib import Path
root=Path("data/raw/gtea"); root.mkdir(parents=True,exist_ok=True)
urls={"issue_comments.json":"https://api.github.com/repos/yabufarha/ms-tcn/issues/18/comments", "gatech_archive.html":"https://web.archive.org/web/20240000000000id_/http://cbs.ic.gatech.edu/fpv/", "stanford_http.html":"http://ai.stanford.edu/~alireza/GTEA/", "hf_files.json":"https://huggingface.co/api/datasets/dinggd/gtea/tree/main"}
for name,url in urls.items():
 try:
  with urllib.request.urlopen(url,timeout=15) as r: body=r.read()
  (root/name).write_bytes(body)
  if name.endswith(".json"): print(name,body.decode()[:14000],flush=True)
  else:
   from html.parser import HTMLParser
   class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
     if tag=="a": print(dict(attrs).get("href"),flush=True)
   print(name,flush=True); Links().feed(body.decode(errors="replace"))
 except Exception as e: print(name,str(e),flush=True)
