#!/usr/bin/env python3
"""Rebuild parent-disjoint SONICS manifests by reusing existing prepared stems."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from prepare_sonics_htdemucs_music_stems import (
    AUDIO_SUFFIXES, build_source_rows, expand_mixtures, write_manifest,
)


def index_stems(prepared_root: Path) -> dict[str, Path]:
    stems = {}
    for path in (prepared_root / "audio").glob("*/*.wav"):
        if path.stem in stems:
            raise ValueError(f"Duplicate prepared stem ID: {path.stem}")
        stems[path.stem] = path.resolve()
    return stems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--train-variants", type=int, default=4)
    args = parser.parse_args()

    fake_records = args.prepared_root / "fake_records_resolved.jsonl"
    parents = build_source_rows(args.real_dir, fake_records, args.seed, args.max_per_class)
    donors = sorted(str(p) for p in args.voice_dir.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    rows = expand_mixtures(parents, donors, args.train_variants, args.seed)
    stems = index_stems(args.prepared_root)
    missing = [row["id"] for row in rows if row["id"] not in stems]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} reusable stems; first={missing[:5]}")
    for row in rows:
        row["filepath"] = str(stems[row["id"]])

    groups = defaultdict(set)
    for row in rows:
        groups[row["music_parent_id"]].add(row["split"])
    leaks = {parent: sorted(splits) for parent, splits in groups.items() if len(splits) > 1}
    if leaks:
        raise RuntimeError(f"Parent leakage after regrouping: {list(leaks.items())[:5]}")

    manifest_root = args.output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        write_manifest([row for row in rows if row["split"] == split], manifest_root / f"{split}.csv")
    report = {
        "reused_stems": len(rows), "source_prepared_root": str(args.prepared_root.resolve()),
        "parent_groups": len(groups), "cross_split_parent_groups": 0,
        "counts": {split: dict(Counter(int(r["target"]) for r in rows if r["split"] == split))
                   for split in ("train", "validation", "test")},
    }
    (args.output_root / "integrity.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
