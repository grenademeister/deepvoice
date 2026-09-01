#!/usr/bin/env python3
"""Prepare immutable HTDemucs accompaniment stems from synthetic AV mixtures."""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "tools"))
from prepare_sonics_htdemucs_music_stems import load_htdemucs, load_audio_16k, extract_accompaniment

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    rows = list(csv.DictReader(args.manifest.open()))
    rows = [r for r in rows if int(r["expected_music_present"])]
    if args.limit: rows = rows[:args.limit]
    model = load_htdemucs(PROJECT / "model" / "htdemucs")
    complete, failures = [], []
    expected = int(16000 * args.duration)
    for i, row in enumerate(rows, 1):
        out = args.output / "audio" / row["split"] / f"{row['sample_id']}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not out.exists():
                mixture = load_audio_16k(Path(row["local_path"]), model.samplerate, args.duration)
                stem = extract_accompaniment(mixture, model, args.device, 16000)[0]
                if stem.shape[0] < expected: stem = np.pad(stem, (0, expected - stem.shape[0]))
                stem = stem[:expected]
                if not np.isfinite(stem).all(): raise RuntimeError("nonfinite stem")
                sf.write(out, stem, 16000, subtype="PCM_16")
            item = dict(row); item["filepath"] = str(out); item["target"] = int(row["expected_music_fake"])
            complete.append(item)
        except Exception as e:
            failures.append({"sample_id": row["sample_id"], "error": repr(e)})
        if i % 25 == 0 or i == len(rows): print(json.dumps({"done":i,"total":len(rows),"complete":len(complete),"failed":len(failures)}),flush=True)
    mdir=args.output/"manifests"; mdir.mkdir(parents=True,exist_ok=True)
    fields=list(complete[0]) if complete else []
    for split in ("train","validation","test"):
        with (mdir/f"{split}.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([r for r in complete if r["split"]==split])
    report={"requested":len(rows),"complete":len(complete),"failures":failures,"counts":{s:dict(Counter(r["target"] for r in complete if r["split"]==s)) for s in ("train","validation","test")}}
    (args.output/"integrity.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    if failures or len(complete)!=len(rows): raise RuntimeError(json.dumps(report))
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__ == "__main__": main()
