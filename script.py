#!/usr/bin/env python3
"""Online-mixing V4: frozen experts, trainable SONICS head and fusion MLP."""
import argparse
import csv
import hashlib
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models import artifactnet, dfarena, panns, separator, sonics
from models.fusion import Fusion

SR, SECONDS = 16000, 10
N = SR * SECONDS
SIX = ('df_voice', 'sonics_stem', 'artifact_raw', 'artifact_stem', 'voice_present', 'music_present')


def stable(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


def partition(group):
    value = stable(group) % 10
    return 'train' if value < 8 else 'validation' if value == 8 else 'test'


def audio(ref):
    if ref.startswith('zip://'):
        archive, member = ref[6:].split('::', 1)
        with zipfile.ZipFile(archive) as z:
            x, rate = sf.read(io.BytesIO(z.read(member)), dtype='float32')
    else:
        x, rate = sf.read(ref, dtype='float32')
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(1)
    if rate != SR:
        x = torchaudio.functional.resample(torch.from_numpy(x), rate, SR).numpy()
    if not np.isfinite(x).all() or not len(x):
        raise ValueError(f'invalid audio: {ref}')
    return x


def crop(x, key):
    if len(x) < N:
        return np.pad(x, (0, N - len(x)), mode='wrap')
    offset = stable(key) % (len(x) - N + 1)
    return x[offset:offset + N]


def rows(path, split):
    with Path(path).open() as f:
        return [r for r in csv.DictReader(f) if r['split'] == split]


class OnlineMixtureDataset(Dataset):
    # One 20-item cycle: 20% music-only, 20% voice-only, 60% equal mixed quadrants.
    kinds = [(0,1,0,0),(0,1,0,1),(0,1,0,0),(0,1,0,1),
             (1,0,0,0),(1,0,1,0),(1,0,0,0),(1,0,1,0),
             *[(1,1,v,m) for v in (0,1) for m in (0,1) for _ in range(3)]]

    def __init__(self, voice_csv, music_csv, split, size, seed=0):
        self.voice = rows(voice_csv, split)
        self.music = rows(music_csv, split)
        self.size, self.seed, self.epoch = size, seed, 0
        self.pools = {
            'voice': {label: [r for r in self.voice if int(r['label']) == label] for label in (0,1)},
            'music': {label: [r for r in self.music if int(r['label']) == label] for label in (0,1)},
        }
        if any(not values for modality in self.pools.values() for values in modality.values()):
            raise ValueError('each split needs both real and fake sources for voice and music')

    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return self.size
    def choose(self, modality, label, index):
        pool = self.pools[modality][label]
        return pool[stable(self.seed, self.epoch, modality, label, index) % len(pool)]

    def __getitem__(self, index):
        vp, mp, vf, mf = self.kinds[index % len(self.kinds)]
        key = (self.seed, self.epoch, index)
        v = crop(audio(self.choose('voice', vf, index)['path']), (*key, 'voice')) if vp else np.zeros(N, np.float32)
        m = crop(audio(self.choose('music', mf, index)['path']), (*key, 'music')) if mp else np.zeros(N, np.float32)
        if vp and mp:
            m *= .2 + (stable(*key, 'gain') % 401) / 1000
        x = v + m
        x /= max(float(np.abs(x).max()), 1.0)
        return torch.from_numpy(x), torch.tensor([vp, mp, vf, mf, int(vf or mf)], dtype=torch.float32)


def build_index(args):
    specs = [('voice', args.voice_real, 0, False), ('voice', args.voice_fake, 1, True),
             ('music', args.music_real, 0, False), ('music', args.music_fake, 1, False)]
    out = Path(args.source_dir); out.mkdir(parents=True, exist_ok=True)
    indexed = {'voice': [], 'music': []}
    for modality, root, label, archive in specs:
        if archive:
            with zipfile.ZipFile(root) as z:
                names = [n for n in z.namelist() if n.lower().endswith(('.wav','.flac','.mp3'))]
            entries = [('zip://' + str(root) + '::' + n, Path(n).stem.rsplit('_generated',1)[0]) for n in names]
        else:
            def group(path):
                stem = path.stem
                return stem.rsplit('_', 2)[0] if modality == 'music' and stem.count('_') >= 2 else stem.rsplit('_', 1)[0]
            entries = [(str(p), group(p)) for p in Path(root).rglob('*') if p.suffix.lower() in {'.wav','.flac','.mp3'}]
        indexed[modality].extend({'path': p, 'label': label, 'group': g, 'split': partition(g)} for p,g in entries)
    for modality, entries in indexed.items():
        path = out / f'{modality}.csv'
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=('path','label','group','split')); writer.writeheader(); writer.writerows(entries)
        print(json.dumps({'manifest': str(path), 'rows': len(entries)}))


def experts(root, device, train_sonics):
    root = Path(root)
    return (separator.load(root/'htdemucs', device), dfarena.load(root/'df_arena_1b', device),
            sonics.load(root/'sonics', device, train_sonics), artifactnet.load(root/'artifactnet'),
            panns.load(root/'panns', device))


def feature_batch(mix, net, device):
    sep, df, music_net, artifact, presence_net = net
    vocals, music = separator.separate(sep, mix.to(device))
    df_score = [dfarena.score(*df, x.detach().cpu().numpy(), device) for x in vocals]
    raw, stem, voice, present_music = [], [], [], []
    pmodel, vi, mi = presence_net
    for x, m in zip(mix, music):
        x, m = x.numpy(), m.detach().cpu().numpy()
        a, b = panns.score(pmodel, vi, mi, x); voice.append(a); present_music.append(b)
        raw.append(artifactnet.score(artifact, x, SR)); stem.append(artifactnet.score(artifact, m, SR))
    sonics_logit = sonics.logits(music_net, music)
    fixed = torch.tensor(np.stack([df_score, raw, stem, voice, present_music], 1), device=device).float().clamp(1e-5, 1-1e-5)
    return torch.cat((torch.logit(fixed[:, :1]), sonics_logit[:, None], torch.logit(fixed[:, 1:])), 1)


def loss(logits, labels):
    vp, mp, vf, mf, ff = labels.T
    bce = nn.BCEWithLogitsLoss(reduction='none')
    value = bce(logits[:,0], ff).mean()
    value += (bce(logits[:,1], vf) * vp).sum() / vp.sum().clamp_min(1)
    value += (bce(logits[:,2], mf) * mp).sum() / mp.sum().clamp_min(1)
    return value


def run_epoch(loader, net, fusion, optimizer, device):
    total = 0.0
    for step, (mix, labels) in enumerate(loader, 1):
        scores = feature_batch(mix, net, device); value = loss(fusion(scores), labels.to(device))
        if optimizer:
            optimizer.zero_grad(); value.backward(); optimizer.step()
        total += value.item()
        if optimizer and step % 20 == 0: print(json.dumps({'batch':step, 'loss':total/step}))
    return total / max(1, len(loader))


def train(args):
    device = torch.device(args.device); train_set = OnlineMixtureDataset(args.voice_csv, args.music_csv, 'train', args.train_size, args.seed)
    valid_set = OnlineMixtureDataset(args.voice_csv, args.music_csv, 'validation', args.valid_size, args.seed)
    net = experts(args.models, device, True); fusion = Fusion().to(device)
    optimizer = torch.optim.AdamW([*net[2].classifier.parameters(), *fusion.parameters()], lr=args.lr)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=0)
    history=[]
    for epoch in range(args.epochs):
        train_set.set_epoch(epoch); tr=run_epoch(train_loader, net, fusion, optimizer, device)
        with torch.no_grad(): va=run_epoch(valid_loader, net, fusion, None, device)
        history.append({'epoch':epoch, 'train_loss':tr, 'validation_loss':va}); print(json.dumps(history[-1]))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'sonics_head':net[2].classifier.state_dict(), 'fusion':fusion.state_dict(), 'features':SIX, 'history':history}, args.output)


def arguments():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command', required=True)
    i=sub.add_parser('index'); i.add_argument('--source-dir', required=True); i.add_argument('--voice-real', required=True); i.add_argument('--voice-fake', required=True); i.add_argument('--music-real', required=True); i.add_argument('--music-fake', required=True)
    t=sub.add_parser('train'); t.add_argument('--voice-csv', required=True); t.add_argument('--music-csv', required=True); t.add_argument('--models', default='model'); t.add_argument('--output', required=True); t.add_argument('--epochs', type=int, default=1); t.add_argument('--train-size', type=int, default=20000); t.add_argument('--valid-size', type=int, default=2000); t.add_argument('--batch-size', type=int, default=1); t.add_argument('--lr', type=float, default=1e-4); t.add_argument('--seed', type=int, default=0); t.add_argument('--device', default='cuda')
    return p.parse_args()


if __name__ == '__main__':
    args=arguments(); build_index(args) if args.command == 'index' else train(args)
