#!/usr/bin/env python3
"""One-epoch joint V1 SONICS + file-fusion adaptation; DF-Arena/PANNs stay frozen."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_curve

PROJECT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(PROJECT),str(PROJECT/'model')]
from script import load_audio, load_panns_model, predict_presence, load_htdemucs_model, separate_voice_and_music, load_df_arena_model, predict_fake
from sonics_infer import SonicsClassifier, preprocess_window

def num(v): return 0.0 if v in ('',None) else float(v)
def logit(x):
 x=torch.clamp(x,1e-5,1-1e-5); return torch.log(x)-torch.log1p(-x)
def eer(y,s):
 fpr,tpr,_=roc_curve(y,s,pos_label=1,drop_intermediate=False); fnr=1-tpr; return float(((fpr+fnr)/2)[np.argmin(np.abs(fpr-fnr))])

class Fusion(nn.Module):
 def __init__(self):
  super().__init__(); self.net=nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,1))
 def forward(self,x): return self.net(x)

class Dataset(torch.utils.data.Dataset):
 def __init__(self, rows, stems): self.rows=rows; self.stems=stems
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]; stem=self.stems/r['split']/f"{r['sample_id']}.wav"; has=stem.is_file()
  if has:
   a,sr=torchaudio.load(stem); a=a.mean(0).numpy()
   if sr!=16000: a=torchaudio.functional.resample(torch.from_numpy(a)[None],sr,16000)[0].numpy()
   a=preprocess_window(a,16000,80000)
  else: a=np.zeros(80000,np.float32)
  fixed=np.array([num(r['df_voice']),num(r['voice_present']),num(r['music_present'])],np.float32)
  return torch.from_numpy(a),torch.from_numpy(fixed),torch.tensor([num(r['file_fake'])]),torch.tensor([num(r['music_fake'])]),torch.tensor([float(has)])

def feature_rows(manifest,cache,split,device):
 if cache.exists():
  rows=[json.loads(x) for x in cache.read_text().splitlines() if x]
  wanted=[r for r in csv.DictReader(manifest.open()) if r['split']==split]
  if len(rows)==len(wanted): return rows
 raw=[r for r in csv.DictReader(manifest.open()) if r['split']==split]
 cache.parent.mkdir(parents=True,exist_ok=True)
 done={}
 if cache.exists():
  for x in cache.read_text().splitlines():
   if x: done[json.loads(x)['sample_id']]=json.loads(x)
 # PANNs is held separately from DF/HTDemucs to fit GTX 1080 VRAM.
 panns,vi,mi=load_panns_model(device); presence={}
 for i,r in enumerate(raw,1):
  if r['sample_id'] not in done:
   presence[r['sample_id']]=predict_presence(panns,vi,mi,load_audio(r['local_path']))
  if i%100==0: print(json.dumps({'phase':'presence','split':split,'processed':i,'total':len(raw)}),flush=True)
 del panns; torch.cuda.empty_cache()
 ht=load_htdemucs_model(); df,idx=load_df_arena_model(device)
 with cache.open('a') as f:
  for i,r in enumerate(raw,1):
   sid=r['sample_id']
   if sid in done: continue
   voice,_,_=separate_voice_and_music(r['local_path'],ht,device)
   vp,mp=presence[sid]
   out={'sample_id':sid,'split':split,'df_voice':predict_fake(df,idx,voice,device),'voice_present':vp,'music_present':mp,'file_fake':num(r['expected_file_fake']),'music_fake':num(r['expected_music_fake'])}
   f.write(json.dumps(out)+'\n');f.flush()
   if i%25==0: print(json.dumps({'phase':'df_htdemucs','split':split,'processed':i,'total':len(raw)}),flush=True)
 del df,ht; torch.cuda.empty_cache()
 return [json.loads(x) for x in cache.read_text().splitlines() if x]

def evaluate(sonics,fusion,loader,device):
 sonics.eval();fusion.eval(); ys=[];pf=[];ym=[];pm=[]
 with torch.inference_mode():
  for a,x,yf,ymu,h in loader:
   a,x=a.to(device),x.to(device); h=h.to(device); mlog=sonics(a); mlog=torch.where(h.bool(),mlog,torch.zeros_like(mlog))
   z=torch.cat([logit(x[:,0:1]),logit(x[:,1:2]),logit(x[:,2:3]),mlog],1); out=fusion(z)
   ys+=yf.reshape(-1).tolist();pf+=torch.sigmoid(out).reshape(-1).cpu().tolist()
   mask=h.reshape(-1)>0; ym += ymu.reshape(-1)[mask.cpu()].tolist(); pm += torch.sigmoid(mlog).reshape(-1)[mask].cpu().tolist()
 return {'file_eer':eer(np.array(ys),np.array(pf)),'music_eer':eer(np.array(ym),np.array(pm))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--stems',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--log-every-batches',type=int,default=50);p.add_argument('--device',default='cuda');a=p.parse_args()
 torch.manual_seed(20260831);np.random.seed(20260831);a.run_dir.mkdir(parents=True,exist_ok=True);dev=torch.device(a.device)
 rows={s:feature_rows(a.manifest,a.run_dir/f'features_{s}.jsonl',s,dev) for s in ('train','validation','test')}
 ds={s:Dataset(rows[s],a.stems/'audio') for s in rows}; dl={s:torch.utils.data.DataLoader(ds[s],batch_size=a.batch_size,shuffle=s=='train') for s in ds}
 cfg=json.loads((PROJECT/'model/sonics/config.json').read_text()); sonics=SonicsClassifier(cfg);sonics.load_state_dict(torch.load(PROJECT/'model/sonics/pytorch_model.bin',map_location='cpu',weights_only=True));sonics=sonics.to(dev); fusion=Fusion().to(dev)
 opt=torch.optim.AdamW(list(sonics.parameters())+list(fusion.parameters()),lr=a.lr,weight_decay=.01)
 hist=[]
 for ep in range(1,a.epochs+1):
  sonics.train();fusion.train(); losses=[]
  total_batches=len(dl['train'])
  for batch_idx,(au,x,yf,ym,h) in enumerate(dl['train'],1):
   au,x,yf,ym,h=au.to(dev),x.to(dev),yf.to(dev),ym.to(dev),h.to(dev); mlog=sonics(au); z=torch.cat([logit(x[:,0:1]),logit(x[:,1:2]),logit(x[:,2:3]),mlog],1); flog=fusion(z); mask=h.bool()
   fl=F.binary_cross_entropy_with_logits(flog,yf); ml=F.binary_cross_entropy_with_logits(mlog[mask],ym[mask]); loss=fl+.5*ml
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(sonics.parameters())+list(fusion.parameters()),5);opt.step();losses.append((float(loss),float(fl),float(ml)))
   if batch_idx % a.log_every_batches == 0 or batch_idx == total_batches:
    recent=losses[-a.log_every_batches:]
    print(json.dumps({'phase':'train','epoch':ep,'batch':batch_idx,'total_batches':total_batches,'running_loss':float(np.mean([v[0] for v in recent])),'running_file_loss':float(np.mean([v[1] for v in recent])),'running_music_loss':float(np.mean([v[2] for v in recent]))}),flush=True)
  val=evaluate(sonics,fusion,dl['validation'],dev); rec={'epoch':ep,'train_loss':float(np.mean([x[0] for x in losses])),'train_file_loss':float(np.mean([x[1] for x in losses])),'train_music_loss':float(np.mean([x[2] for x in losses])),**{'validation_'+k:v for k,v in val.items()}};hist.append(rec);print(json.dumps(rec),flush=True)
 test=evaluate(sonics,fusion,dl['test'],dev); torch.save({'sonics_state_dict':sonics.cpu().state_dict(),'fusion_state_dict':fusion.cpu().state_dict(),'config':cfg,'features':['df_voice','voice_present','music_present','sonics_logit'],'file_loss':'BCE','music_loss_weight':.5,'df_arena_frozen':True,'panns_frozen':True},a.run_dir/'joint_epoch1.pt'); (a.run_dir/'history.json').write_text(json.dumps(hist,indent=2)+'\n');(a.run_dir/'metrics.json').write_text(json.dumps({'validation':val,'test':test,'counts':{s:len(rows[s]) for s in rows}},indent=2)+'\n')
 print(json.dumps({'validation':val,'test':test},indent=2),flush=True)
if __name__=='__main__':main()
