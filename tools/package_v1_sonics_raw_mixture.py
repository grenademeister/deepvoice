#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,io,json,shutil,zipfile
from pathlib import Path
import torch
CHUNK=8*1024*1024; TARGET='model/sonics/pytorch_model.bin'
def patch(source):
 old='        voice_audio, music_audio = separate_voice_and_music(\n            audio_path, htdemucs_model, device\n        )\n'
 new='        voice_audio, _separated_music_audio = separate_voice_and_music(\n            audio_path, htdemucs_model, device\n        )\n        # Raw-mixture SONICS ablation: music score intentionally bypasses HTDemucs.\n        music_audio = load_audio(audio_path)\n'
 if old not in source: raise RuntimeError('separation call not found')
 return source.replace(old,new)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);state=ck['state_dict'] if isinstance(ck,dict) and 'state_dict' in ck else ck;b=io.BytesIO();torch.save(state,b);weight=b.getvalue()
 with zipfile.ZipFile(a.base) as src:
  script=patch(src.read('script.py').decode());compile(script,'script.py','exec');a.output.parent.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as dst:
   for info in src.infolist():
    if info.is_dir():continue
    if info.filename==TARGET:dst.writestr(info,weight)
    elif info.filename=='script.py':dst.writestr(info,script.encode())
    else:
     with src.open(info) as r,dst.open(info,'w',force_zip64=True) as w:shutil.copyfileobj(r,w,CHUNK)
 with zipfile.ZipFile(a.output) as z:
  roots={n.split('/',1)[0] for n in z.namelist() if n};assert roots=={'model','script.py','requirements.txt'},roots;assert z.testzip() is None;packed=z.read('script.py').decode();assert '_separated_music_audio' in packed and 'music_audio = load_audio(audio_path)' in packed;loaded=torch.load(io.BytesIO(z.read(TARGET)),map_location='cpu',weights_only=True);assert set(loaded)==set(state)
 print(json.dumps({'output':str(a.output),'bytes':a.output.stat().st_size,'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'crc':'passed','mode':'raw-mixture SONICS; separated DF-Arena voice'},sort_keys=True))
if __name__=='__main__':main()
