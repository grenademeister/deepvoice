#!/usr/bin/env python3
"""Build a source-disjoint 20/20/60 synthetic music+voice corpus on the GPU host."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, random
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

SR = 16000
DURATION = 10
N_SAMPLES = SR * DURATION
SPLIT_FRAC = {"train": .8, "validation": .1, "test": .1}
STRATA = ("music_only_real", "music_only_fake", "voice_only_real", "voice_only_fake", "rv_rm", "rv_fm", "fv_rm", "fv_fm")

def stable(*parts):
    return int.from_bytes(hashlib.sha256("\0".join(map(str, parts)).encode()).digest()[:8], "big")

def split_pool(items, key, seed):
    out = {s: [] for s in SPLIT_FRAC}
    for item in items:
        x = stable(seed, key(item)) % 10000
        split = "train" if x < 8000 else "validation" if x < 9000 else "test"
        out[split].append(item)
    return out

def choose(pool, n, rng, name):
    if not pool:
        raise ValueError(f"empty pool: {name}")
    return [pool[rng.randrange(len(pool))] for _ in range(n)]

def load_file(path, offset):
    waveform, sr = torchaudio.load(str(path))
    waveform = waveform.mean(0)
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform[None], sr, SR)[0]
    return crop_repeat(waveform.numpy(), offset)

def crop_repeat(audio, offset):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("empty audio")
    if audio.size < N_SAMPLES:
        audio = np.tile(audio, (N_SAMPLES + audio.size - 1) // audio.size)
    start = offset % (audio.size - N_SAMPLES + 1)
    return audio[start:start + N_SAMPLES].copy()

def load_asv(item, offset):
    data, sr = sf.read(io.BytesIO(item["bytes"]), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SR:
        data = torchaudio.functional.resample(torch.from_numpy(data)[None], sr, SR)[0].numpy()
    return crop_repeat(data, offset)

def normalize(audio):
    peak = float(np.max(np.abs(audio)))
    return audio if peak < 1e-8 else audio * min(0.98 / peak, 1.0)

def quotas(total):
    if total % 20:
        raise ValueError("total must be divisible by 20")
    q = total // 20
    return {"music_only_real": 2*q, "music_only_fake": 2*q, "voice_only_real": 2*q, "voice_only_fake": 2*q,
            "rv_rm": 3*q, "rv_fm": 3*q, "fv_rm": 3*q, "fv_fm": 3*q}

def read_asv(root):
    result = {0: [], 1: []}
    for parquet in sorted((root / "data").glob("*.parquet")):
        df = pd.read_parquet(parquet, columns=["audio_file_name", "speaker_id", "system_id", "key", "audio"])
        for row in df.to_dict("records"):
            label = int(row["key"])
            result[label].append({"id": row["audio_file_name"], "speaker": row["speaker_id"], "system": row["system_id"], "bytes": row["audio"]["bytes"]})
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("/root/deepvoice/runs/synthetic_av_20260830"))
    ap.add_argument("--sonics", type=Path, default=Path("/root/datasets/sonics/extracted/fake_songs"))
    ap.add_argument("--real-music", type=Path, default=Path("/root/datasets/musiccaps_real"))
    ap.add_argument("--asvspoof", type=Path, default=Path("/root/datasets/asvspoof2019_la"))
    ap.add_argument("--total", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    if args.total % 200:
        raise ValueError("total must be divisible by 200 so every split has exact strata")
    q = quotas(args.total)
    real_music = sorted(args.real_music.glob("*.wav"))
    fake_music = sorted(p for p in args.sonics.rglob("*") if p.suffix.lower() in {".mp3", ".wav", ".flac"})
    if len(real_music) < 100 or len(fake_music) < 100:
        raise ValueError(f"insufficient music: real={len(real_music)} fake={len(fake_music)}")
    voice = read_asv(args.asvspoof)
    if not voice[0] or not voice[1]:
        raise ValueError("ASVspoof lacks a voice class")
    pools = {
        "mr": split_pool(real_music, lambda x: x.stem, args.seed),
        "mf": split_pool(fake_music, lambda x: x.stem, args.seed),
        "vr": split_pool(voice[0], lambda x: x["id"], args.seed),
        "vf": split_pool(voice[1], lambda x: x["id"], args.seed),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for split, frac in SPLIT_FRAC.items():
        n = int(args.total * frac)
        sq = quotas(n)
        cases = []
        cases += [("music_only", None, "real", v) for v in choose(pools["mr"][split], sq["music_only_real"], rng, "mr")]
        cases += [("music_only", None, "fake", v) for v in choose(pools["mf"][split], sq["music_only_fake"], rng, "mf")]
        cases += [("voice_only", "real", None, v) for v in choose(pools["vr"][split], sq["voice_only_real"], rng, "vr")]
        cases += [("voice_only", "fake", None, v) for v in choose(pools["vf"][split], sq["voice_only_fake"], rng, "vf")]
        for name, vl, ml in (("rv_rm", "real", "real"), ("rv_fm", "real", "fake"), ("fv_rm", "fake", "real"), ("fv_fm", "fake", "fake")):
            vp = pools["vr" if vl == "real" else "vf"][split]
            mp = pools["mr" if ml == "real" else "mf"][split]
            cases += [("mixed", vl, ml, (v, m)) for v, m in zip(choose(vp, sq[name], rng, name+"v"), choose(mp, sq[name], rng, name+"m"))]
        rng.shuffle(cases)
        for idx, (domain, vl, ml, sources) in enumerate(cases):
            sample_id = f"{split}_{idx:05d}"
            target = args.output / "audio" / split / f"{sample_id}.wav"
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if domain == "music_only":
                    music = load_file(sources, stable(args.seed, sample_id, "m"))
                    audio = normalize(music); voice_id = ""; music_id = str(sources); vf = ""; mf = int(ml == "fake")
                elif domain == "voice_only":
                    voice_audio = load_asv(sources, stable(args.seed, sample_id, "v"))
                    audio = normalize(voice_audio); voice_id = sources["id"]; music_id = ""; vf = int(vl == "fake"); mf = ""
                else:
                    v, m = sources
                    voice_audio = load_asv(v, stable(args.seed, sample_id, "v"))
                    music = load_file(m, stable(args.seed, sample_id, "m"))
                    music_gain = rng.uniform(.25, .50)
                    audio = normalize(voice_audio + music_gain * music)
                    voice_id = v["id"]; music_id = str(m); vf = int(vl == "fake"); mf = int(ml == "fake")
                sf.write(target, audio, SR, subtype="PCM_16")
                row = {"sample_id": sample_id, "local_path": str(target), "split": split, "audio_domain": domain,
                       "expected_voice_present": int(domain != "music_only"), "expected_music_present": int(domain != "voice_only"),
                       "expected_voice_fake": vf, "expected_music_fake": mf, "expected_file_fake": int(vf == 1 or mf == 1),
                       "voice_source_id": voice_id, "music_source_id": music_id, "split_group": f"voice:{voice_id}|music:{music_id}",
                       "sample_rate": SR, "duration": DURATION}
                rows.append(row)
            except Exception as exc:
                failures.append({"sample_id": sample_id, "error": repr(exc)})
            if (idx + 1) % 100 == 0 or idx + 1 == len(cases): print(json.dumps({"split": split, "done": idx+1, "total": len(cases), "failures": len(failures)}), flush=True)
    fields = list(rows[0]) if rows else []
    with (args.output / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    report = {"total_requested": args.total, "total_written": len(rows), "failures": failures,
              "counts": {"split": dict(Counter(r["split"] for r in rows)), "domain": dict(Counter(r["audio_domain"] for r in rows)),
                         "quadrant": dict(Counter((r["expected_voice_fake"], r["expected_music_fake"]) for r in rows if r["audio_domain"] == "mixed"))},
              "sources": {"sonics": str(args.sonics), "real_music": str(args.real_music), "asvspoof": str(args.asvspoof)}}
    report["counts"]["quadrant"] = {f"voice={a};music={b}": n for (a, b), n in report["counts"]["quadrant"].items()}
    (args.output / "integrity.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if failures or len(rows) != args.total: raise RuntimeError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2, sort_keys=True))
if __name__ == "__main__": main()
