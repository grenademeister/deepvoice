#!/usr/bin/env python3
"""Evaluate raw-mixture SONICS in V1 fusion; DF/PANNs components are cached and unchanged."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,roc_curve
PROJECT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(PROJECT),str(PROJECT/'model')]
from sonics_infer import SonicsClassifier,predict_fake
from script import load_audio

def num(x):return 0.0 if x in ('',None) else float(x)
def eer(y,s):
 fpr,tpr,_=roc_curve(y,s,pos_label=1,drop_intermediate=False);fnr=1-tpr;return float(((fpr+fnr)/2)[np.argmin(abs(fpr-fnr))])
def v1(v,m,vp,mp):return max(v*vp,m*mp)
def load(config,ckpt,key,dev):
 m=SonicsClassifier(json.loads(config.read_text()));s=torch.load(ckpt,map_location='cpu',weights_only=key is None);m.load_state_dict(s if key is None else s[key],strict=True);return m.to(dev).eval()
def metrics(rows,p):
 yf=np.array([num(r['expected_file_fake']) for r in rows]);yv=np.array([num(r['expected_voice_fake']) for r in rows]);ym=np.array([num(r['expected_music_fake']) for r in rows]);yvp=np.array([num(r['expected_voice_present']) for r in rows]);ymp=np.array([num(r['expected_music_present']) for r in rows]);pf=np.array([x['file'] for x in p]);pv=np.array([x['voice'] for x in p]);pm=np.array([x['music'] for x in p]);vp=np.array([x['vp'] for x in p]);mp=np.array([x['mp'] for x in p]);fe=eer(yf,pf);ve=eer(yv[yvp>.5],pv[yvp>.5]);me=eer(ym[ymp>.5],pm[ymp>.5]);ads=.5*(1-fe)+.2*(1-ve)+.3*(1-me);cps=.5*roc_auc_score(yvp,vp)+.5*roc_auc_score(ymp,mp);mixed=np.array([r['audio_domain']=='mixed' for r in rows]);return {'n':len(rows),'score':.9*ads+.1*cps,'ads':ads,'cps':cps,'file_eer':fe,'file_eer_mixed':eer(yf[mixed],pf[mixed]),'voice_eer':ve,'music_eer':me,'mixed_n':int(mixed.sum())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--evalset',type=Path,default=Path('/root/deepvoice-evalset'));p.add_argument('--component-run',type=Path,required=True);p.add_argument('--adapted-checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--splits',nargs='+',default=['validation','test']);p.add_argument('--device',default='cuda');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);dev=torch.device(a.device);manifest=list(csv.DictReader((a.evalset/'manifests/manifest_balanced.csv').open()));models={'original':load(PROJECT/'model/sonics/config.json',PROJECT/'model/sonics/pytorch_model.bin',None,dev),'adapted':load(PROJECT/'model/sonics/config.json',a.adapted_checkpoint,'state_dict',dev)};out={}
 for split in a.splits:
  rows=[r for r in manifest if r['split_balanced']==split];comp={r['ID']:r for r in csv.DictReader((a.component_run/f'predictions_{split}.csv').open())};assert set(comp)=={r['sample_id'] for r in rows};pred={k:[] for k in models};f=(a.output_dir/f'raw_sonics_{split}_scores.csv').open('w',newline='');w=csv.DictWriter(f,fieldnames=['ID','original_music','original_file','adapted_music','adapted_file','voice','voice_present','music_present']);w.writeheader()
  for i,r in enumerate(rows,1):
   c=comp[r['sample_id']];audio=load_audio(a.evalset/r['local_path']);v,vp,mp=num(c['VOICE_FAKE_PROB']),num(c['VOICE_PRESENT_PROB']),num(c['MUSIC_PRESENT_PROB']);row={'ID':r['sample_id'],'voice':v,'voice_present':vp,'music_present':mp}
   for name,m in models.items():
    music=predict_fake(m,audio,16000,device=dev);file=v1(v,music,vp,mp);pred[name].append({'file':file,'voice':v,'music':music,'vp':vp,'mp':mp});row[name+'_music']=music;row[name+'_file']=file
   w.writerow(row)
   if i%25==0 or i==len(rows):print(json.dumps({'split':split,'processed':i,'total':len(rows)}),flush=True)
  f.close();out[split]={k:metrics(rows,x) for k,x in pred.items()}
 (a.output_dir/'metrics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
