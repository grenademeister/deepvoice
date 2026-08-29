#!/usr/bin/env python3
"""Build balanced val/test splits with >60% music+voice, balanced quadrants."""
import csv, random, hashlib
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path("/home/grenade/workspace/deepvoice-evalset")
MANIFEST = ROOT/"manifests/manifest.csv"
OUT = ROOT/"manifests/manifest_balanced.csv"

TARGET_VAL = 180  # 30 voice-only, 30 music-only, 120 mixed (30 per quadrant)
TARGET_TEST = 180
# voice-only <20%, music-only <20%, mixed >60%
# mixed quadrants: RV+RM, RV+FM, FV+RM, FV+FM equal

random.seed(42)

rows = list(csv.DictReader(open(MANIFEST)))
# index by split_group
by_group = defaultdict(list)
for r in rows:
    by_group[r["split_group"]].append(r)

# categorize each row
def to_int(v):
    if v is None or v=="": return 0
    try: return int(float(v))
    except: return 0
def cat(r):
    vp = to_int(r["expected_voice_present"])
    mp = to_int(r["expected_music_present"])
    if vp==1 and mp==1:
        vf = to_int(r["expected_voice_fake"])
        mf = to_int(r["expected_music_fake"])
        return ("mixed", (vf,mf))
    elif vp==1 and mp==0:
        return ("voice_only", None)
    elif vp==0 and mp==1:
        return ("music_only", None)
    else:
        return ("empty", None)

# Build pools by group: groups are atomic, but for mixing we need to classify groups by their rows' category.
# For simplicity, assume groups are homogeneous in category (true for raw, but mixed groups have single row).
# For groups with multiple rows (e.g., derived shifts share parent group), we need to skip those for balanced selection
# to avoid leakage. We will only consider groups where all rows share same category.

group_cat = {}
for g, recs in by_group.items():
    cats = set(cat(r)[0] for r in recs)
    if len(cats)==1:
        group_cat[g]=cats.pop()
    else:
        group_cat[g]="heterogeneous"

# For mixed, also need quadrant
group_quad = {}
for g, recs in by_group.items():
    if group_cat[g]=="mixed":
        # single row per mixed group (voice_music_mix)
        r=recs[0]
        vf=int(float(r["expected_voice_fake"])); mf=int(float(r["expected_music_fake"]))
        group_quad[g]=(vf,mf)

# Count available groups per category
cnt = Counter(group_cat.values())
print("groups by category", cnt)
quad_cnt = Counter(group_quad.values())
print("mixed groups by quadrant", quad_cnt)

# Need to know how many groups needed: each group may have multiple rows (e.g., speech group with shifts).
# But for balanced val/test we want FILE-LEVEL counts, not groups. Safer to select at ROW level with group disjoint constraint.
# So we will select rows, but ensure no group appears in both train and val/test.

# Filter candidate rows: exclude heterogeneous groups for now, also exclude empty/silence_noise if we want strict 3 categories.
# User spec: voice-only <20%, music-only <20%, music+voice >60%. The remaining ~0% can be empty/noise, but we will exclude empty from val/test to hit 60%.

def row_cat(r):
    vp=to_int(r["expected_voice_present"]); mp=to_int(r["expected_music_present"])
    if vp==1 and mp==1: 
        return "mixed"
    elif vp==1 and mp==0:
        return "voice_only"
    elif vp==0 and mp==1:
        return "music_only"
    else:
        return "empty"

# Pools for val/test: all rows
pools = defaultdict(list)
for r in rows:
    pools[row_cat(r)].append(r)

print("rows by category", {k:len(v) for k,v in pools.items()})
# quadrants within mixed
mixed_quads = defaultdict(list)
for r in pools["mixed"]:
    vf=to_int(r["expected_voice_fake"]); mf=to_int(r["expected_music_fake"])
    mixed_quads[(vf,mf)].append(r)
print("mixed rows by quadrant", {k:len(v) for k,v in mixed_quads.items()})

# Check feasibility for TARGET_VAL+TARGET_TEST = 360 total: need 60 voice-only each, 60 music-only, 240 mixed (60 per quadrant)
need_voice = 60
need_music = 60
need_mixed_per_quad = 60
print(f"need voice {need_voice} avail {len(pools['voice_only'])}")
print(f"need music {need_music} avail {len(pools['music_only'])}")
for q in [(0,0),(0,1),(1,0),(1,1)]:
    print(f"need quadrant {q} {need_mixed_per_quad} avail {len(mixed_quads[q])}")

# If insufficient, report and adjust target
# Currently pools: voice_only 773, music_only 505, mixed 56 -> mixed insufficient (max 10 per quadrant for 0,1 and 1,0)
# Need to generate ~200 more mixed first. This script will exit with feasibility check.

if any(len(mixed_quads[q]) < need_mixed_per_quad for q in [(0,0),(0,1),(1,0),(1,1)]):
    print("INFEASIBLE: not enough mixed rows. Need to generate more mixtures first.")
    # suggest required generation
    for q in [(0,0),(0,1),(1,0),(1,1)]:
        deficit = need_mixed_per_quad - len(mixed_quads[q])
        if deficit>0:
            print(f"  deficit quadrant {q}: {deficit}")
