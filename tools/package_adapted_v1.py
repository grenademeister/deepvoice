#!/usr/bin/env python3
"""Build a full V1 submission ZIP by replacing SONICS weights in a verified base ZIP."""
from __future__ import annotations
import argparse, hashlib, io, json, shutil, zipfile
from pathlib import Path
import torch

CHUNK=8*1024*1024

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--base",type=Path,required=True)
 p.add_argument("--checkpoint",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 a=p.parse_args()
 ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
 state=ckpt["state_dict"] if isinstance(ckpt,dict) and "state_dict" in ckpt else ckpt
 b=io.BytesIO(); torch.save(state,b); adapted=b.getvalue()
 target="model/sonics/pytorch_model.bin"
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.base,"r") as src, zipfile.ZipFile(a.output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as dst:
  assert target in src.namelist()
  for info in src.infolist():
   if info.is_dir(): continue
   if info.filename==target:
    dst.writestr(info,adapted)
   else:
    with src.open(info,"r") as r, dst.open(info,"w",force_zip64=True) as w:
     shutil.copyfileobj(r,w,CHUNK)
 with zipfile.ZipFile(a.output,"r") as z:
  roots={x.split("/",1)[0] for x in z.namelist() if x}
  assert roots=={"model","script.py","requirements.txt"},roots
  assert z.testzip() is None
  raw=z.read(target)
  loaded=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
  assert set(loaded)==set(state)
 digest=hashlib.sha256(a.output.read_bytes()).hexdigest()
 print(json.dumps({"output":str(a.output),"bytes":a.output.stat().st_size,"sha256":digest,"checkpoint":str(a.checkpoint),"crc":"passed"},sort_keys=True))
if __name__=="__main__": main()
