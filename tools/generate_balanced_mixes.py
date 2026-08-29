#!/usr/bin/env python3
"""Generate additional voice_music_mix to satisfy balanced val/test deficits."""
import csv, random, hashlib, json, subprocess, tempfile
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path("/home/grenade/workspace/deepvoice-evalset")
DERIVED = ROOT/"audio/derived/mixtures"
MANIFEST = ROOT/"manifests/manifest.csv"
random.seed(42)

def to_int(v):
    if v is None or v=="": return 0
    try: return int(float(v))
    except: return 0
def stable_id(a,b):
    return hashlib.sha256(f"{a}|{b}".encode()).hexdigest()[:16]

rows=list(csv.DictReader(open(MANIFEST)))
# pools of raw speech/music by fake label
speech_real=[r for r in rows if not r["source_dataset"].startswith("derived:") and r["audio_domain"]=="speech" and to_int(r["expected_file_fake"])==0]
speech_fake=[r for r in rows if not r["source_dataset"].startswith("derived:") and r["audio_domain"]=="speech" and to_int(r["expected_file_fake"])==1]
music_real=[r for r in rows if not r["source_dataset"].startswith("derived:") and r["audio_domain"]=="music" and to_int(r["expected_file_fake"])==0]
music_fake=[r for r in rows if not r["source_dataset"].startswith("derived:") and r["audio_domain"]=="music" and to_int(r["expected_file_fake"])==1]

print(f"pools: speech_real {len(speech_real)} speech_fake {len(speech_fake)} music_real {len(music_real)} music_fake {len(music_fake)}")

# existing mixes ids
existing = {r["sample_id"] for r in rows if r["source_dataset"]=="derived:voice_music_mix"}
print(f"existing mixes {len(existing)}")

# deficits from previous check: need 60 per quadrant total, have 18,18,10,10
targets = {(0,0):60, (0,1):60, (1,0):60, (1,1):60}
have = Counter((to_int(r["expected_voice_fake"]), to_int(r["expected_music_fake"])) for r in rows if r["source_dataset"]=="derived:voice_music_mix")
print("have quadrants", dict(have))
deficits = {k: max(0, targets[k]-have[k]) for k in targets}
print("deficits", deficits)

# helper to sample diversity: group by source_dataset then round-robin
def sample_diverse(pool, n):
    by_src=defaultdict(list)
    for r in pool:
        by_src[r["source_dataset"]].append(r)
    for v in by_src.values():
        random.shuffle(v)
    # round robin
    result=[]
    srcs=list(by_src.keys())
    random.shuffle(srcs)
    idx=0
    while len(result)<n:
        src=srcs[idx % len(srcs)]
        lst=by_src[src]
        if lst:
            # cycle if needed
            result.append(lst[len(result)//len(srcs) % len(lst)] if len(result)//len(srcs) < len(lst) else random.choice(lst))
            # actually pop round robin correctly: use pointer
            pass
        idx+=1
        if idx> n*10: break
    # simpler: just random sample with source stratification via weighted shuffle
    # fallback to random.choices with uniform
    if len(result)<n:
        # Use random sample with replacement ensuring diversity
        # Build stratified sample
        result=[]
        # distribute n across sources proportionally
        srcs=list(by_src.keys())
        # allocate equally
        per_src = n // len(srcs)
        rem = n % len(srcs)
        for i, src in enumerate(srcs):
            cnt = per_src + (1 if i<rem else 0)
            lst=by_src[src]
            # sample with replacement if cnt > len(lst)
            if cnt <= len(lst):
                result.extend(random.sample(lst, cnt))
            else:
                result.extend([random.choice(lst) for _ in range(cnt)])
        random.shuffle(result)
    return result[:n]

# Prepare pairs
from common import file_sha256, probe

def probe_audio(p):
    import subprocess, json
    # use ffprobe via common.probe if available, else simple
    try:
        from common import probe as cprobe
        return cprobe(p)
    except:
        return {"duration":10, "sample_rate":16000}

# Need to actually generate mixes: for each quadrant, create deficit pairs
pairs=[]
for (vf,mf), deficit in deficits.items():
    voice_pool = speech_real if vf==0 else speech_fake
    music_pool = music_real if mf==0 else music_fake
    voices = sample_diverse(voice_pool, deficit)
    musics = sample_diverse(music_pool, deficit)
    # ensure we don't reuse same pair if stable_id already exists; if collision, resample
    for v,m in zip(voices, musics):
        sid=f"mix_v{vf}m{mf}_{stable_id(v['sample_id'], m['sample_id'])}"
        if sid in existing:
            # find alternative music
            for alt in music_pool:
                sid2=f"mix_v{vf}m{mf}_{stable_id(v['sample_id'], alt['sample_id'])}"
                if sid2 not in existing:
                    m=alt; sid=sid2; break
        pairs.append((vf,mf,v,m,sid))

print(f"to generate {len(pairs)} new mixes")
for vf,mf,v,m,sid in pairs[:3]:
    print(vf,mf,sid, v["sample_id"], m["sample_id"])

# Generate
DERIVED.mkdir(parents=True, exist_ok=True)
generated=[]
for vf,mf,voice,song,sid in pairs:
    dest=DERIVED/f"{sid}.wav"
    if dest.exists():
        print(f"skip exists {sid}")
        continue
    # ffmpeg mix: voice gain 1.0, music 0.35, duration shortest, 10s max
    cmd=["ffmpeg","-v","error","-y","-stream_loop","-1","-i",str(ROOT/voice["local_path"]),"-stream_loop","-1","-i",str(ROOT/song["local_path"]),"-filter_complex","[0:a]volume=1.0[v];[1:a]volume=0.35[m];[v][m]amix=inputs=2:duration=shortest:normalize=0[out]","-map","[out]","-t","10","-ac","1","-ar","16000",str(dest)]
    subprocess.run(cmd, check=True)
    generated.append((sid, dest, voice, song, vf, mf))
    print(f"generated {sid}")

print(f"generated {len(generated)} files")

# Now append records to derived_records.jsonl
# Load existing derived records to avoid duplicate
import json
derived_path=ROOT/"manifests/derived_records.jsonl"
existing_derived_ids=set()
if derived_path.exists():
    for line in derived_path.read_text().splitlines():
        if line.strip():
            existing_derived_ids.add(json.loads(line)["sample_id"])

new_records=[]
for sid,dest,voice,song,vf,mf in generated:
    # build derived record similar to build_derived.make_mixtures
    # need file_sha256 and probe
    from common import file_sha256, probe
    rel=dest.relative_to(ROOT)
    rec=dict(voice)  # copy voice base then override
    rec.update({
        "sample_id": sid,
        "source_dataset": "derived:voice_music_mix",
        "source_reference": voice["sample_id"],
        "local_path": rel.as_posix(),
        "manipulation_type": "voice_music_mix",
        "codec_condition": voice.get("codec_condition","normalized_pcm"),
        "separation_status": voice.get("separation_status","none"),
        "transformation_lineage": f"{voice['transformation_lineage']}+{song['transformation_lineage']}->mix",
        "parent_id": f"mix:{voice['parent_id']}:{song['parent_id']}",
        "split_group": f"mix:{voice['split_group']}:{song['split_group']}",
        "audio_domain": "mixed",
        "vocal_mode": "speech",
        "expected_file_fake": int(vf or mf),
        "expected_voice_fake": vf,
        "expected_music_fake": mf,
        "expected_voice_present": 1,
        "expected_music_present": 1,
        "generator_family": f"voice={voice['generator_family']};music={song['generator_family']}",
        "redistributable": bool(voice.get("redistributable")=="True" and song.get("redistributable")=="True") if isinstance(voice.get("redistributable"), str) else bool(voice.get("redistributable") and song.get("redistributable")),
        "sha256": file_sha256(dest),
        **probe(dest),
    })
    # preserve license from voice
    new_records.append(rec)

# append to derived_records.jsonl
if new_records:
    with open(derived_path, "a") as f:
        for rec in new_records:
            f.write(json.dumps(rec)+"\n")
    print(f"appended {len(new_records)} records to {derived_path}")

# Now rebuild manifest.csv via build_manifest logic (inline) but without forcing mixed to stress
# We'll rebuild fully: re-run manifest generation but patch split assignment for mixed to standard_split
import hashlib
import pandas as pd
def bucket(text, mod=10000):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12],16)%mod
def standard_split(group):
    v=bucket(group)
    if v<7000: return "train"
    if v<8500: return "validation"
    return "test"

records=[]
for path in sorted((ROOT/"manifests").glob("*_records.jsonl")):
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
# deduplicate
seen={}
for r in records:
    seen[r["sample_id"]]=r
records=list(seen.values())
for r in records:
    r.setdefault("codec_condition","normalized_pcm")
    r.setdefault("separation_status","none")
    r.setdefault("retrieval_date","2026-08-27")
    # ONLY force partial to stress, NOT voice_music_mix
    if r["source_dataset"] in {"derived:partial_speech","derived:partial_music"}:
        r["split"]="stress"
    else:
        r["split"]=standard_split(r["split_group"])
    r["fold"]=bucket(r["split_group"],5)
    r["source_holdout_group"]=r["source_dataset"].replace("derived:","")
    r["generator_holdout_group"]=r["generator_family"]
frame=pd.DataFrame(records).sort_values("sample_id").reset_index(drop=True)
out=ROOT/"manifests/manifest.csv"
frame.to_csv(out,index=False)
print(f"rebuilt manifest {len(frame)} rows")
print(frame["split"].value_counts().to_dict())
print(frame[frame.source_dataset=="derived:voice_music_mix"]["split"].value_counts().to_dict())
# summary
import json as js
summary={"records":len(frame),"by_split":frame["split"].value_counts().sort_index().to_dict(),"by_source":frame["source_dataset"].value_counts().sort_index().to_dict()}
print(js.dumps(summary, indent=2))
