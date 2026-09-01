#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_curve
PROJECT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(PROJECT),str(PROJECT/"model"),str(PROJECT/"tools")]
from script import load_htdemucs_model, separate_voice_and_music
from sonics_infer import SonicsClassifier, preprocess_window
from train_joint_v1_sonics_fusion import Fusion, logit

def num(x): return 0.0 if x in (None, "") else float(x)
def eer(y,s):
 fpr,tpr,_=roc_curve(y,s,pos_label=1,drop_intermediate=False); fnr=1-tpr
 return float(((fpr+fnr)/2)[np.argmin(np.abs(fpr-fnr))])
def main():
 a=argparse.ArgumentParser(); a.add_argument("--checkpoint",type=Path,required=True);a.add_argument("--component-run",type=Path,required=True);a.add_argument("--evalset",type=Path,default=Path("/root/deepvoice-evalset"));a.add_argument("--output-dir",type=Path,required=True);a.add_argument("--device",default="cuda");q=a.parse_args(); dev=torch.device(q.device);q.output_dir.mkdir(parents=True,exist_ok=True)
 ck=torch.load(q.checkpoint,map_location="cpu",weights_only=False);son=SonicsClassifier(ck["config"]);son.load_state_dict(ck["sonics_state_dict"],strict=True);son=son.to(dev).eval();fusion=Fusion().to(dev);fusion.load_state_dict(ck["fusion_state_dict"],strict=True);fusion.eval();ht=load_htdemucs_model()
 manifest=list(csv.DictReader((q.evalset/"manifests/manifest_balanced.csv").open()));out={}
 for split in ("validation","test"):
  rows=[r for r in manifest if r["split_balanced"]==split]; comps={r["ID"]:r for r in csv.DictReader((q.component_run/f"predictions_{split}.csv").open())};assert set(comps)=={r["sample_id"] for r in rows}
  scores=[]; ym=[]; yf=[];detail=[]
  for i,r in enumerate(rows,1):
   c=comps[r["sample_id"]]; _,m,_=separate_voice_and_music(q.evalset/r["local_path"],ht,dev);w=preprocess_window(m,16000,80000); au=torch.from_numpy(w)[None].to(dev);fixed=torch.tensor([[num(c["VOICE_FAKE_PROB"]),num(c["VOICE_PRESENT_PROB"]),num(c["MUSIC_PRESENT_PROB"])]],device=dev)
   with torch.inference_mode():
    ml=son(au);z=torch.cat([logit(fixed[:,0:1]),logit(fixed[:,1:2]),logit(fixed[:,2:3]),ml],1);fs=float(torch.sigmoid(fusion(z)).item());ms=float(torch.sigmoid(ml).item())
   scores.append(fs);yf.append(num(r["expected_file_fake"]));
   if num(r["expected_music_present"])>.5: ym.append((num(r["expected_music_fake"]),ms))
   detail.append({"ID":r["sample_id"],"file_fake":fs,"music_fake":ms,"df_voice":num(c["VOICE_FAKE_PROB"]),"voice_present":num(c["VOICE_PRESENT_PROB"]),"music_present":num(c["MUSIC_PRESENT_PROB"])})
   if i%25==0 or i==len(rows): print(json.dumps({"split":split,"processed":i,"total":len(rows)}),flush=True)
  yma,pma=map(np.array,zip(*ym)); mixed=np.array([r["audio_domain"]=="mixed" for r in rows]);out[split]={"n":len(rows),"file_eer":eer(np.array(yf),np.array(scores)),"file_eer_mixed":eer(np.array(yf)[mixed],np.array(scores)[mixed]),"music_eer":eer(yma,pma),"music_present_n":len(ym)}
  with (q.output_dir/f"scores_{split}.jsonl").open("w") as f:
   for x in detail:f.write(json.dumps(x)+"\n")
 (q.output_dir/"metrics.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
