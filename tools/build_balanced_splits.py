#!/usr/bin/env python3
"""Create manifest_balanced.csv with val/test >60% mixed, balanced quadrants, group-disjoint."""
import csv, random, hashlib
from pathlib import Path
from collections import defaultdict, Counter

ROOT=Path("/home/grenade/workspace/deepvoice-evalset")
MANIFEST=ROOT/"manifests/manifest.csv"
OUT=ROOT/"manifests/manifest_balanced.csv"

random.seed(42)

def to_int(v):
    if v in (None,""): return 0
    try: return int(float(v))
    except: return 0

rows=list(csv.DictReader(open(MANIFEST)))
# Keep original manifest rows, we will add column split_balanced
# Group handling: groups must be disjoint across splits
by_group=defaultdict(list)
for r in rows:
    by_group[r["split_group"]].append(r)

def row_cat(r):
    vp=to_int(r["expected_voice_present"]); mp=to_int(r["expected_music_present"])
    if vp==1 and mp==1: return "mixed"
    elif vp==1 and mp==0: return "voice_only"
    elif vp==0 and mp==1: return "music_only"
    else: return "empty"

def quad(r):
    return (to_int(r["expected_voice_fake"]), to_int(r["expected_music_fake"]))

# Pools at group level: we need to select groups, not rows, to keep disjoint
# For mixed, each group has 1 row (derived mixture). For voice_only/music_only, groups may have multiple rows (shifts).
# To achieve exact file counts, we will select at row level but enforce group disjoint by tracking used groups.

# Build row pools
pools=defaultdict(list)
for r in rows:
    # exclude empty/silence_noise from val/test pools (we want strict 20/20/60)
    c=row_cat(r)
    if c in ("voice_only","music_only","mixed"):
        pools[c].append(r)

# For mixed, sub-pool by quadrant
mixed_by_quad=defaultdict(list)
for r in pools["mixed"]:
    mixed_by_quad[quad(r)].append(r)

print("pools:", {k:len(v) for k,v in pools.items()})
print("mixed quads:", {k:len(v) for k,v in mixed_by_quad.items()})

# Targets
VAL_N=150
TEST_N=150
# 25 voice_only (16.7%), 25 music_only (16.7%), 100 mixed (66.7%) with 25 per quadrant
val_targets={"voice_only":25, "music_only":25, "mixed":100}
test_targets={"voice_only":25, "music_only":25, "mixed":100}
mixed_per_quad_val=25
mixed_per_quad_test=25

# Helper to select diverse rows with group disjoint
def select_pools(pool_rows, n, used_groups):
    # group rows by source_dataset for diversity
    by_src=defaultdict(list)
    for r in pool_rows:
        if r["split_group"] in used_groups:
            continue
        by_src[r["source_dataset"]].append(r)
    for lst in by_src.values():
        random.shuffle(lst)
    # round-robin across sources
    result=[]
    srcs=list(by_src.keys())
    random.shuffle(srcs)
    # pointers per source
    ptrs={s:0 for s in srcs}
    while len(result)<n:
        progressed=False
        for s in srcs:
            if len(result)>=n: break
            lst=by_src[s]
            p=ptrs[s]
            if p < len(lst):
                r=lst[p]
                if r["split_group"] in used_groups:
                    ptrs[s]+=1
                    continue
                # ensure group not already used by another selected row (if group has multiple rows, picking one uses whole group)
                if r["split_group"] in used_groups:
                    ptrs[s]+=1
                    continue
                result.append(r)
                used_groups.add(r["split_group"])
                # also mark all rows of same group as used (to avoid picking another row from same group)
                # for shift groups, picking one file should block siblings
                ptrs[s]+=1
                progressed=True
        if not progressed:
            # fallback: random sample from remaining
            remaining=[r for r in pool_rows if r["split_group"] not in used_groups]
            if not remaining:
                break
            r=random.choice(remaining)
            result.append(r)
            used_groups.add(r["split_group"])
    return result

used=set()
val_rows=[]
test_rows=[]
# Select mixed quadrants first
for q in [(0,0),(0,1),(1,0),(1,1)]:
    pool=mixed_by_quad[q]
    sel=select_pools(pool, mixed_per_quad_val, used)
    if len(sel)<mixed_per_quad_val:
        print(f"WARN val quadrant {q} only {len(sel)}/{mixed_per_quad_val}")
    val_rows.extend(sel)
for q in [(0,0),(0,1),(1,0),(1,1)]:
    pool=mixed_by_quad[q]
    sel=select_pools(pool, mixed_per_quad_test, used)
    if len(sel)<mixed_per_quad_test:
        print(f"WARN test quadrant {q} only {len(sel)}/{mixed_per_quad_test}")
    test_rows.extend(sel)

# voice_only
sel=select_pools(pools["voice_only"], val_targets["voice_only"], used)
val_rows.extend(sel)
sel=select_pools(pools["voice_only"], test_targets["voice_only"], used)
test_rows.extend(sel)

# music_only
sel=select_pools(pools["music_only"], val_targets["music_only"], used)
val_rows.extend(sel)
sel=select_pools(pools["music_only"], test_targets["music_only"], used)
test_rows.extend(sel)

print(f"val selected {len(val_rows)} test selected {len(test_rows)} used groups {len(used)}")

# Verify
def summarize(name, sel):
    cats=Counter(row_cat(r) for r in sel)
    quads=Counter(quad(r) for r in sel if row_cat(r)=="mixed")
    print(f"{name} {len(sel)} cats {dict(cats)} quads {dict(quads)}")
    # percentages
    for k in ["voice_only","music_only","mixed"]:
        print(f"  {k} {100*cats[k]/len(sel):.1f}%")
    src=Counter(r["source_dataset"] for r in sel)
    print("  sources", dict(src))

summarize("VAL", val_rows)
summarize("TEST", test_rows)

# Check group disjoint
val_groups={r["split_group"] for r in val_rows}
test_groups={r["split_group"] for r in test_rows}
print("group overlap val/test", len(val_groups & test_groups))

# Now create balanced manifest: copy all rows, add split_balanced column
# train_balanced = all remaining rows not in val/test and not stress, plus stress stays stress
val_ids={r["sample_id"] for r in val_rows}
test_ids={r["sample_id"] for r in test_rows}

# Also need to handle groups with multiple rows (shifts): if one row of a group is selected, all siblings in that group should be excluded from train to keep group disjoint.
# Our used set already contains groups, so train will exclude any row whose group in used

balanced=[]
for r in rows:
    nr=dict(r)
    sid=r["sample_id"]
    grp=r["split_group"]
    if sid in val_ids:
        nr["split_balanced"]="validation"
    elif sid in test_ids:
        nr["split_balanced"]="test"
    elif grp in used:
        # group used in val/test but this sibling row not selected -> exclude from train to avoid leakage; assign to unused
        nr["split_balanced"]="unused_group_sibling"
    elif r["split"]=="stress":
        nr["split_balanced"]="stress"
    else:
        nr["split_balanced"]="train"
    balanced.append(nr)

# Also include stress rows as is
from collections import Counter
cnt=Counter(r["split_balanced"] for r in balanced)
print("balanced counts", dict(cnt))

# Write
fieldnames=list(csv.DictReader(open(MANIFEST)).fieldnames) + ["split_balanced"]
with open(OUT,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in sorted(balanced, key=lambda x: x["sample_id"]):
        w.writerow({k: r.get(k,"") for k in fieldnames})

print(f"wrote {OUT} {len(balanced)} rows")

# Also write summary json
import json
summary={
    "val_targets": val_targets,
    "test_targets": test_targets,
    "val_actual": dict(Counter(row_cat(r) for r in val_rows)),
    "test_actual": dict(Counter(row_cat(r) for r in test_rows)),
    "val_quads": {str(k):v for k,v in Counter(quad(r) for r in val_rows if row_cat(r)=="mixed").items()},
    "test_quads": {str(k):v for k,v in Counter(quad(r) for r in test_rows if row_cat(r)=="mixed").items()},
}
(ROOT/"reports/balanced_manifest_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
