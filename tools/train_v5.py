#!/usr/bin/env python3
"""V5 batched trainer — implicit SONICS cache after epoch 1.

Epoch 1: full forward (HTDemucs → DF500M / SONICS raw+stem / Artifact / PANNs) + backprop.
         SONICS raw/stem embeddings are saved to sonics_cache per sample_id.
Epoch 2+: SONICS encoder is skipped, cached embeddings are reused.

Batched: DataLoader batch_size groups, SONICS encoder runs batched per group.
No explicit feature files — cache is on disk, no RAM growth.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import Dataset, DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "model"))

from model.v5_fusion import V5Fusion, V5Meta, payload, logit


def num(v):
    return 0.0 if v in ("", None) else float(v)


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    return float(((fpr + 1 - tpr) / 2)[np.argmin(np.abs(fpr - (1 - tpr)))])


def metrics(rows, pf, pv, pm):
    yf = np.array([num(r["expected_file_fake"]) for r in rows], float)
    yv = np.array([num(r["expected_voice_fake"]) for r in rows], float)
    ym = np.array([num(r["expected_music_fake"]) for r in rows], float)
    yvp = np.array([num(r["expected_voice_present"]) for r in rows], float)
    ymp = np.array([num(r["expected_music_present"]) for r in rows], float)
    fe = eer(yf, np.array(pf, float))
    vm = yvp > 0.5
    mm = ymp > 0.5
    ve = eer(yv[vm], np.array(pv)[vm]) if vm.sum() else 0.0
    me = eer(ym[mm], np.array(pm)[mm]) if mm.sum() else 0.0
    ads = 0.5 * (1 - fe) + 0.2 * (1 - ve) + 0.3 * (1 - me)
    # CPS needs presence probs — caller passes those separately if needed
    return {"file_eer": fe, "voice_eer": ve, "music_eer": me, "ads": ads}


class ManifestDS(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def sonics_embed_batched(model, wavs, dev):
    """wavs: list[np] each (T,) at 16k. Returns np [B, D]. Batched encoder."""
    from sonics_infer import preprocess_window
    import torch
    batch = np.stack([preprocess_window(w, 16000, 80000) for w in wavs])  # [B, 80000]
    t = torch.from_numpy(batch).to(dev)
    with torch.inference_mode():
        spec = model.ft_extractor(t)
        spec = spec.unsqueeze(1)
        spec = F.interpolate(spec, size=model.input_shape, mode="bilinear")
        feats = model.encoder(spec)  # [B, T, D]
        emb = feats.mean(dim=1)
    return emb.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=20260904)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    a.run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = a.run_dir / "sonics_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stems_cache = a.run_dir / "stems_cache"
    stems_cache.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    # --- frozen models ---
    from script import (load_audio, load_panns_model, predict_presence,
                        load_htdemucs_model, separate_voice_and_music,
                        load_artifactnet_model, predict_artifactnet_raw_and_stem,
                        ARTIFACTNET_SAMPLE_RATE)
    from sonics_infer import SonicsClassifier
    import json as _json
    # DF500M via HF cache offline
    from huggingface_hub import snapshot_download
    from transformers import AutoModel
    snap = snapshot_download("Speech-Arena-2025/DF_Arena_500M_V_1", local_files_only=True)
    df = AutoModel.from_pretrained(snap, trust_remote_code=True, local_files_only=True).to(dev).eval()
    fake_idx = int(df.config.label2id["spoof"])
    panns, vi, mi = load_panns_model(dev)
    htd = load_htdemucs_model()
    art = load_artifactnet_model(PROJECT / "model" / "artifactnet")
    cfg = _json.loads((PROJECT / "model" / "sonics" / "config.json").read_text())
    son = SonicsClassifier(cfg)
    son.load_state_dict(torch.load(PROJECT / "model" / "sonics" / "pytorch_model.bin", map_location="cpu", weights_only=True))
    son.to(dev).eval()
    for p in son.parameters(): p.requires_grad_(False)
    # --- manifest ---
    rows_all = list(csv.DictReader(a.manifest.open()))
    # manifest local_path may be relative to deepvoice root
    for r in rows_all:
        if not Path(r["local_path"]).is_absolute():
            r["local_path"] = str(PROJECT / r["local_path"])
    tr_rows = [r for r in rows_all if r["split"] == "train"]
    va_rows = [r for r in rows_all if r["split"] == "validation"]
    print(f"train {len(tr_rows)} val {len(va_rows)}", flush=True)

    fusion = V5Fusion(proj_dim=a.proj_dim, hidden_dim=a.hidden_dim).to(dev)
    opt = torch.optim.AdamW(fusion.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    history = []

    def run_split(split_rows, train: bool):
        # batched DataLoader over rows
        ds = ManifestDS(split_rows)
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=train, collate_fn=lambda x: x, num_workers=0)
        all_pf, all_pv, all_pm, all_vp, all_mp = [], [], [], [], []
        losses = []
        for bi, batch in enumerate(dl):
            # per-batch: run frozen branches
            scalars, raw_embs, stem_embs, labels, masks = [], [], [], [], []
            # collect uncached sonics inputs for batched encoder
            need_raw, need_stem, need_ids = [], [], []
            pending = []
            # Phase 1: Batched HTDemucs with stem file cache (epoch1 computes batched, epoch2 loads)
            voices, accs, raw44s, vps, mps = [], [], [], [], []
            # split batch into cached stems vs need HTDemucs
            need_hdemucs_idx = []
            need_paths = []
            for idx, r in enumerate(batch):
                sid0 = r["sample_id"]
                scp = stems_cache / f"{sid0}.npz"
                if scp.exists():
                    try:
                        d = __import__("numpy").load(str(scp))
                        voices.append(d["voice"]); accs.append(d["acc"])
                        # placeholder for later PANNs/raw
                        need_hdemucs_idx.append(None)
                        continue
                    except Exception:
                        pass
                # need compute
                need_hdemucs_idx.append(idx)
                need_paths.append(r["local_path"])
                voices.append(None); accs.append(None)
            # batched HTDemucs for uncached (single apply_model call)
            if need_paths:
                import torch as _thd
                from script import load_track
                import torchaudio
                # load and normalize per-sample, pad to max
                wavs = []
                means, stds, lens = [], [], []
                max_len = 0
                for pth in need_paths:
                    wf = load_track(pth, htd.audio_channels, htd.samplerate).float()  # [C, T]
                    mono = wf.mean(0)
                    mean = mono.mean(); std = mono.std()
                    if float(std) < 1e-8:
                        # silence: create zeros at target sr
                        length = round(wf.shape[-1] * 16000 / htd.samplerate)
                        wavs.append((__import__("numpy").zeros(max(1,length), dtype=__import__("numpy").float32), __import__("numpy").zeros(max(1,length), dtype=__import__("numpy").float32), mean, std, True))
                        continue
                    norm = (wf - mean) / std
                    wavs.append((norm, mean, std, False))
                    max_len = max(max_len, norm.shape[-1])
                # pad and stack
                batch_norm = []
                valid_idx = []
                for item in wavs:
                    if len(item)==5: # silence case
                        continue
                    norm, mean, std, _ = item
                    pad = max_len - norm.shape[-1]
                    if pad>0:
                        norm = _thd.nn.functional.pad(norm, (0, pad))
                    batch_norm.append(norm)
                    means.append(mean); stds.append(std)
                if batch_norm:
                    batch_tensor = _thd.stack(batch_norm).to(dev)  # [B_need, C, T]
                    with _thd.inference_mode():
                        from demucs.apply import apply_model as _apply
                        sources = _apply(htd, batch_tensor, device=dev, shifts=0, split=True, overlap=0.25, progress=False)  # [B, 4, C, T]
                    # sources: [B, nsrc, C, T]
                    vocal_idx = htd.sources.index("vocals")
                    for bj, orig_idx in enumerate([i for i in need_hdemucs_idx if i is not None]):
                        # denorm
                        mean = means[bj]; std = stds[bj]
                        src = sources[bj] * std + mean  # [nsrc, C, T]
                        voice_t = src[vocal_idx].mean(0, keepdim=True)
                        music_sources = [src[i] for i, n in enumerate(htd.sources) if n != "vocals"]
                        music_t = _thd.stack(music_sources).sum(0).mean(0, keepdim=True)
                        # resample to 16k
                        voice = torchaudio.functional.resample(voice_t, htd.samplerate, 16000)[0].cpu().numpy().astype(__import__("numpy").float32)
                        acc = torchaudio.functional.resample(music_t, htd.samplerate, 16000)[0].cpu().numpy().astype(__import__("numpy").float32)
                        voices[orig_idx] = voice; accs[orig_idx] = acc
                        # save stem cache
                        sid0 = batch[orig_idx]["sample_id"]
                        scp = stems_cache / f"{sid0}.npz"
                        __import__("numpy").savez_compressed(str(scp).replace(".npz",".tmp.npz"), voice=voice, acc=acc)
                        __import__("os").replace(str(scp).replace(".npz",".tmp.npz"), str(scp))
                # handle silence cases (already zero)
                for idx, r in enumerate(batch):
                    if voices[idx] is None:
                        # silence
                        sid0 = r["sample_id"]
                        scp = stems_cache / f"{sid0}.npz"
                        # create zeros if not already
                        if not scp.exists():
                            wf = load_track(r["local_path"], htd.audio_channels, htd.samplerate).float()
                            length = round(wf.shape[-1] * 16000 / htd.samplerate)
                            voice = __import__("numpy").zeros(max(1,length), dtype=__import__("numpy").float32)
                            acc = voice.copy()
                            voices[idx]=voice; accs[idx]=acc
                            __import__("numpy").savez_compressed(str(scp).replace(".npz",".tmp.npz"), voice=voice, acc=acc)
                            __import__("os").replace(str(scp).replace(".npz",".tmp.npz"), str(scp))
                        else:
                            d=__import__("numpy").load(str(scp))
                            voices[idx]=d["voice"]; accs[idx]=d["acc"]
            # PANNs/raw per-sample (cheap)
            for r in batch:
                vp, mp = predict_presence(panns, vi, mi, load_audio(r["local_path"]))
                vps.append(vp); mps.append(mp)
                raw44s.append(load_audio(r["local_path"], ARTIFACTNET_SAMPLE_RATE))
            # Chunked batched DF500M: [B,T] -> [B,2] with micro-batch to stay <8GB
            df_probs = []
            micro = 4  # 4*160k fits in 4GB headroom
            with torch.inference_mode():
                for i in range(0, len(voices), micro):
                    chunk = voices[i:i+micro]
                    max_len = max(v.shape[0] for v in chunk)
                    batch_arr = np.stack([np.pad(v, (0, max_len - v.shape[0])) if v.shape[0] < max_len else v[:max_len] for v in chunk])
                    wavs_t = torch.from_numpy(batch_arr).float().to(dev)
                    out = df(input_values=wavs_t)
                    logits_b = out["logits"] if isinstance(out, dict) else out.logits
                    probs = torch.softmax(logits_b.float(), dim=-1)[:, fake_idx].detach().cpu().numpy()
                    df_probs.extend(probs.tolist())
            # Phase 2: assemble scalars
            for idx, r in enumerate(batch):
                dfp = float(df_probs[idx])
                vp, mp = vps[idx], mps[idx]
                acc = accs[idx]
                raw44 = raw44s[idx]
                ar, ast = predict_artifactnet_raw_and_stem(art, raw_audio=raw44, raw_sample_rate=ARTIFACTNET_SAMPLE_RATE, music_stem=acc, stem_sample_rate=16000)
                sc = np.array([dfp, ar, ast, vp, mp], np.float32)
                scalars.append(logit(sc))
                labels.append([num(r["expected_file_fake"]), num(r["expected_voice_fake"]), num(r["expected_music_fake"])])
                masks.append([1.0, num(r["expected_voice_present"]), num(r["expected_music_present"])])
                all_vp.append(vp); all_mp.append(mp)
                sid = r["sample_id"]
                # SONICS file cache: epoch 1 saves, epoch 2+ reuses
                cp = cache_dir / f"{sid}.npz"
                if cp.exists():
                    try:
                        d = np.load(str(cp))
                        er, es = d["raw"], d["stem"]
                        raw_embs.append(er); stem_embs.append(es)
                    except Exception:
                        pending.append((sid, len(raw_embs), cp))
                        raw_embs.append(None); stem_embs.append(None)
                        need_raw.append(load_audio(r["local_path"]))
                        need_stem.append(acc)
                else:
                    pending.append((sid, len(raw_embs), cp))
                    raw_embs.append(None); stem_embs.append(None)
                    need_raw.append(load_audio(r["local_path"]))
                    need_stem.append(acc)
            # batched SONICS for uncached (and save to disk immediately)
            if need_raw:
                er_batch = sonics_embed_batched(son, need_raw, dev)
                es_batch = sonics_embed_batched(son, need_stem, dev)
                for (sid, idx, cp), er, es in zip(pending, er_batch, es_batch):
                    tmp = str(cp).replace(".npz", ".tmp.npz")
                    np.savez_compressed(tmp, raw=er, stem=es)
                    import os as _os
                    _os.replace(tmp, str(cp))
                    raw_embs[idx] = er
                    stem_embs[idx] = es
            # Fusion micro-batched bsz=4 -> 4 forwards/backprops per 1 HTDemucs (B=16) pass
            micro_f = 4
            if train:
                for s in range(0, len(scalars), micro_f):
                    e = min(s+micro_f, len(scalars))
                    scalars_t = torch.from_numpy(np.stack(scalars[s:e])).to(dev)
                    raw_t = torch.from_numpy(np.stack(raw_embs[s:e])).to(dev)
                    stem_t = torch.from_numpy(np.stack(stem_embs[s:e])).to(dev)
                    y = torch.from_numpy(np.array(labels[s:e], np.float32)).to(dev)
                    m = torch.from_numpy(np.array(masks[s:e], np.float32)).to(dev)
                    logits = fusion(scalars_t, raw_t, stem_t)
                    loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * m).sum() / m.sum().clamp_min(1)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0); opt.step()
                    lv=float(loss.detach().cpu())
                    losses.append(lv)
                if (bi+1) % a.log_every == 0:
                    print(json.dumps({"phase": "train_batch", "epoch": ep, "batch": bi+1, "loss": round(lv,4), "cache_files": len(list(cache_dir.glob("*.npz")))}), flush=True)
            else:
                with torch.inference_mode():
                    for s in range(0, len(scalars), micro_f):
                        e = min(s+micro_f, len(scalars))
                        scalars_t = torch.from_numpy(np.stack(scalars[s:e])).to(dev)
                        raw_t = torch.from_numpy(np.stack(raw_embs[s:e])).to(dev)
                        stem_t = torch.from_numpy(np.stack(stem_embs[s:e])).to(dev)
                        logits = fusion(scalars_t, raw_t, stem_t)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        all_pf.extend(probs[:,0].tolist()); all_pv.extend(probs[:,1].tolist()); all_pm.extend(probs[:,2].tolist())
        if train:
            return float(np.mean(losses)) if losses else 0.0
        else:
            return all_pf, all_pv, all_pm, all_vp, all_mp

    for ep in range(1, a.epochs + 1):
        print(json.dumps({"phase": "epoch_start", "epoch": ep}), flush=True)
        tr_loss = run_split(tr_rows, train=True)
        pf, pv, pm, vp, mp = run_split(va_rows, train=False)
        met = metrics(va_rows, pf, pv, pm)
        # CPS from presence (frozen) — compute AUCs
        try:
            yvp = np.array([num(r["expected_voice_present"]) for r in va_rows], float)
            ymp = np.array([num(r["expected_music_present"]) for r in va_rows], float)
            met["voice_auc"] = float(roc_auc_score(yvp, np.array(vp, float)))
            met["music_auc"] = float(roc_auc_score(ymp, np.array(mp, float)))
            met["cps"] = 0.5 * (met["voice_auc"] + met["music_auc"])
            met["score"] = 0.9 * met["ads"] + 0.1 * met["cps"]
        except Exception:
            met["cps"] = met["score"] = float("nan")
        # save per-epoch predictions + metrics (official contract)
        (a.run_dir / f"epoch_{ep:02d}").mkdir(exist_ok=True)
        (a.run_dir / f"epoch_{ep:02d}" / "metrics.json").write_text(json.dumps(met, indent=2) + "\n")
        with (a.run_dir / f"predictions_validation_epoch{ep:02d}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ID","FILE_FAKE_PROB","VOICE_FAKE_PROB","MUSIC_FAKE_PROB","VOICE_PRESENT_PROB","MUSIC_PRESENT_PROB"])
            w.writeheader()
            for r, a0, b0, c0, d0, e0 in zip(va_rows, pf, pv, pm, vp, mp):
                w.writerow({"ID": r["sample_id"], "FILE_FAKE_PROB": f"{a0:.6f}", "VOICE_FAKE_PROB": f"{b0:.6f}", "MUSIC_FAKE_PROB": f"{c0:.6f}", "VOICE_PRESENT_PROB": d0, "MUSIC_PRESENT_PROB": e0})
        rec = {"epoch": ep, "train_loss": tr_loss, **met, "cached_sonics": len(list(cache_dir.glob("*.npz")))}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        sched.step()

    meta = V5Meta.create(a.proj_dim, a.hidden_dim, "500m")
    # scalar stats from cache? use dummy zero-mean for now (logits already)
    s_mean = np.zeros(5, np.float32); s_std = np.ones(5, np.float32)
    torch.save(payload(fusion.cpu(), meta, s_mean, s_std, history), a.run_dir / "v5_checkpoint.pt")
    (a.run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"done": True, "history": history}), flush=True)


if __name__ == "__main__":
    main()
