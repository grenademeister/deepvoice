#!/usr/bin/env python3
"""Full SONICS one-epoch adaptation on raw music-containing synthetic mixtures."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_curve
PROJECT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT/'model'))
from sonics_infer import SonicsClassifier,preprocess_window

def eer(y,s):
 fpr,tpr,_=roc_curve(y,s,pos_label=1,drop_intermediate=False);fnr=1-tpr;return float(((fpr+fnr)/2)[np.argmin(abs(fpr-fnr))])
class RawMusicDataset(torch.utils.data.Dataset):
 def __init__(self,manifest,split):
  self.rows=[r for r in csv.DictReader(Path(manifest).open()) if r['split']==split and float(r['expected_music_present'] or 0)>=.5]
  if not self.rows: raise ValueError(split)
  if any(not Path(r['local_path']).is_file() for r in self.rows): raise FileNotFoundError('missing raw audio')
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];a,sr=torchaudio.load(r['local_path']);a=a.mean(0).numpy()
  if sr!=16000:a=torchaudio.functional.resample(torch.from_numpy(a)[None],sr,16000)[0].numpy()
  return torch.from_numpy(preprocess_window(a,16000,80000)),torch.tensor([float(r['expected_music_fake'])],dtype=torch.float32)
def metrics(model,loader,dev):
 model.eval();ys=[];ps=[];loss=[]
 with torch.inference_mode():
  for a,y in loader:
   z=model(a.to(dev));loss.append(float(F.binary_cross_entropy_with_logits(z,y.to(dev))));ys+=y.reshape(-1).tolist();ps+=torch.sigmoid(z).reshape(-1).cpu().tolist()
 return {'loss':float(np.mean(loss)),'eer':eer(np.array(ys),np.array(ps)),'real_mean':float(np.mean(np.array(ps)[np.array(ys)==0])),'fake_mean':float(np.mean(np.array(ps)[np.array(ys)==1]))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=1);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--log-every-batches',type=int,default=50);p.add_argument('--device',default='cuda');a=p.parse_args();a.run_dir.mkdir(parents=True,exist_ok=True)
 torch.manual_seed(20260901);np.random.seed(20260901);dev=torch.device(a.device);ds={s:RawMusicDataset(a.manifest,s) for s in ('train','validation','test')};dl={s:torch.utils.data.DataLoader(v,batch_size=a.batch_size,shuffle=s=='train',num_workers=0) for s,v in ds.items()}
 cfg=json.loads((PROJECT/'model/sonics/config.json').read_text());m=SonicsClassifier(cfg);m.load_state_dict(torch.load(PROJECT/'model/sonics/pytorch_model.bin',map_location='cpu',weights_only=True),strict=True);m=m.to(dev)
 baseline={s:metrics(m,dl[s],dev) for s in ('validation','test')};print(json.dumps({'baseline_raw':baseline},sort_keys=True),flush=True)
 opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=.01);history=[]
 for ep in range(1,a.epochs+1):
  m.train();losses=[];n=len(dl['train'])
  for b,(x,y) in enumerate(dl['train'],1):
   z=m(x.to(dev));loss=F.binary_cross_entropy_with_logits(z,y.to(dev));opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();losses.append(float(loss))
   if b%a.log_every_batches==0 or b==n: print(json.dumps({'phase':'train','epoch':ep,'batch':b,'total_batches':n,'running_loss':float(np.mean(losses[-a.log_every_batches:]))}),flush=True)
  val=metrics(m,dl['validation'],dev);rec={'epoch':ep,'train_loss':float(np.mean(losses)),**{'validation_'+k:v for k,v in val.items()}};history.append(rec);print(json.dumps(rec,sort_keys=True),flush=True)
 final={'baseline_raw':baseline,'validation':metrics(m,dl['validation'],dev),'test':metrics(m,dl['test'],dev),'counts':{s:len(v) for s,v in ds.items()},'trainable_parameters':sum(x.numel() for x in m.parameters()),'input':'raw_mixture_16k_no_separation','df_arena_frozen':True}
 torch.save({'state_dict':m.cpu().state_dict(),'config':cfg,'input':'raw_mixture_16k_no_separation','seed':20260901,'unfreeze_all':True},a.run_dir/'best.pt');(a.run_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n');(a.run_dir/'metrics.json').write_text(json.dumps(final,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.figure(figsize=(5,4));plt.plot([r['epoch'] for r in history],[r['train_loss'] for r in history],marker='o',label='train BCE');plt.plot([r['epoch'] for r in history],[r['validation_loss'] for r in history],marker='o',label='validation BCE');plt.legend();plt.grid(alpha=.3);plt.xlabel('epoch');plt.ylabel('BCE');plt.tight_layout();plt.savefig(a.run_dir/'learning_curves.png',dpi=160)
 print(json.dumps(final,indent=2),flush=True)
if __name__=='__main__':main()
