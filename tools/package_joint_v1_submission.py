#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,io,json,shutil,zipfile
from pathlib import Path
import torch
CHUNK=8*1024*1024
TARGET='model/sonics/pytorch_model.bin'; FUSION='model/joint_v1_fusion.pt'
def patched(source):
 source=source.replace('SONICS_DIR = LOCAL_MODEL_DIR / "sonics"\n','SONICS_DIR = LOCAL_MODEL_DIR / "sonics"\nJOINT_FUSION_PATH = LOCAL_MODEL_DIR / "joint_v1_fusion.pt"\n')
 old='def combine_file_fake_score(voice_fake, music_fake, voice_present, music_present):\n    voice_score = voice_present * voice_fake\n    music_score = music_present * music_fake\n    return max(voice_score, music_score)\n'
 new='class JointV1Fusion(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.GELU(), torch.nn.Linear(16, 1))\n\n    def forward(self, features):\n        return self.net(features)\n\n\ndef load_joint_fusion(device):\n    payload = torch.load(JOINT_FUSION_PATH, map_location="cpu", weights_only=True)\n    model = JointV1Fusion()\n    model.load_state_dict(payload["fusion_state_dict"], strict=True)\n    return model.to(device).eval()\n\n\ndef _score_logit(value):\n    value = min(max(float(value), 1e-5), 1.0 - 1e-5)\n    return float(np.log(value / (1.0 - value)))\n\n\ndef combine_file_fake_score(voice_fake, music_fake, voice_present, music_present, fusion_model, device):\n    features = torch.tensor([[_score_logit(voice_fake), _score_logit(voice_present), _score_logit(music_present), _score_logit(music_fake)]], dtype=torch.float32, device=device)\n    with torch.inference_mode():\n        return float(torch.sigmoid(fusion_model(features))[0, 0])\n'
 if old not in source: raise RuntimeError('V1 combine rule not found')
 source=source.replace(old,new)
 old='    sonics_model = load_sonics_model(SONICS_DIR, device)\n'; new='    sonics_model = load_sonics_model(SONICS_DIR, device)\n    fusion_model = load_joint_fusion(device)\n'
 if old not in source: raise RuntimeError('SONICS load site not found')
 source=source.replace(old,new)
 old='        file_fake = combine_file_fake_score(\n            voice_fake, music_fake, voice_present, music_present\n        )\n'; new='        file_fake = combine_file_fake_score(\n            voice_fake, music_fake, voice_present, music_present, fusion_model, device\n        )\n'
 if old not in source: raise RuntimeError('V1 call site not found')
 return source.replace(old,new)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--joint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 payload=torch.load(a.joint,map_location='cpu',weights_only=False); sonics=payload['sonics_state_dict']; fusion={'fusion_state_dict':payload['fusion_state_dict'],'features':payload['features'],'df_arena_frozen':True,'panns_frozen':True}
 b=io.BytesIO();torch.save(sonics,b);sonics_bytes=b.getvalue(); b=io.BytesIO();torch.save(fusion,b);fusion_bytes=b.getvalue()
 with zipfile.ZipFile(a.base) as src:
  script=patched(src.read('script.py').decode());compile(script,'script.py','exec');a.output.parent.mkdir(parents=True,exist_ok=True)
  with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as dst:
   for info in src.infolist():
    if info.is_dir(): continue
    if info.filename==TARGET: dst.writestr(info,sonics_bytes)
    elif info.filename=='script.py': dst.writestr(info,script.encode())
    else:
     with src.open(info) as r,dst.open(info,'w',force_zip64=True) as w: shutil.copyfileobj(r,w,CHUNK)
   dst.writestr(FUSION,fusion_bytes)
 with zipfile.ZipFile(a.output) as z:
  roots={n.split('/',1)[0] for n in z.namelist() if n};assert roots=={'model','script.py','requirements.txt'},roots;assert z.testzip() is None;assert FUSION in z.namelist();assert b'load_joint_fusion' in z.read('script.py');check=torch.load(io.BytesIO(z.read(TARGET)),map_location='cpu',weights_only=True);assert set(check)==set(sonics)
 print(json.dumps({'output':str(a.output),'bytes':a.output.stat().st_size,'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'crc':'passed'},sort_keys=True))
if __name__=='__main__':main()
