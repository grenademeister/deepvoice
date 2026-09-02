#!/usr/bin/env python3
"""Build bounded, group-disjoint source manifests for online V4 mixing.

This script never downloads a complete archive to obtain a subset. Provide local
original source pools, then pass the generated CSVs to ``script.py train``.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

AUDIO = {'.wav', '.flac', '.mp3'}


def stable(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


def split(group):
    value = stable(group) % 10
    return 'train' if value < 8 else 'validation' if value == 8 else 'test'


def select(rows, per_class, seed):
    chosen = []
    for label in (0, 1):
        pool = [row for row in rows if int(row['label']) == label]
        groups = {}
        for row in pool:
            groups.setdefault(row['group'], row)
        if len(groups) < per_class:
            raise ValueError(f'insufficient label={label}: {len(groups)} groups < {per_class}')
        chosen.extend(sorted(groups.values(), key=lambda r: stable(seed, r['group']))[:per_class])
    return chosen


def files(root, label, modality):
    rows = []
    for path in Path(root).rglob('*'):
        if path.suffix.lower() not in AUDIO:
            continue
        stem = path.stem
        group = stem.rsplit('_', 2)[0] if modality == 'music' and stem.count('_') >= 2 else stem.rsplit('_', 1)[0]
        rows.append({'path': str(path), 'label': label, 'group': group})
    return rows


def wavefake(archive):
    rows = []
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if Path(member).suffix.lower() in AUDIO:
                group = Path(member).stem.rsplit('_generated', 1)[0]
                rows.append({'path': f'zip://{archive}::{member}', 'label': 1, 'group': group})
    return rows


def bounded(rows, per_split_class, seed):
    out = []
    for name in ('train', 'validation', 'test'):
        out.extend(select([row for row in rows if split(row['group']) == name], per_split_class, seed))
    for row in out:
        row['split'] = split(row['group'])
    return out


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('path', 'label', 'group', 'split'))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--voice-real', type=Path, required=True)
    p.add_argument('--voice-fake-zip', type=Path, required=True)
    p.add_argument('--music-real', type=Path, required=True)
    p.add_argument('--music-fake', type=Path, required=True)
    p.add_argument('--per-split-class', type=int, default=500)
    p.add_argument('--seed', type=int, default=20260902)
    p.add_argument('--start-train', action='store_true')
    p.add_argument('--models', type=Path)
    p.add_argument('--checkpoint', type=Path)
    args = p.parse_args()
    required = (args.voice_real, args.voice_fake_zip, args.music_real, args.music_fake)
    if not all(path.exists() for path in required):
        raise FileNotFoundError('A required local source pool is absent. Full-archive download is intentionally forbidden; supply an individually-downloadable source pool.')
    voice = bounded(files(args.voice_real, 0, 'voice') + wavefake(args.voice_fake_zip), args.per_split_class, args.seed)
    music = bounded(files(args.music_real, 0, 'music') + files(args.music_fake, 1, 'music'), args.per_split_class, args.seed)
    write(args.output / 'voice.csv', voice); write(args.output / 'music.csv', music)
    print(json.dumps({'voice_rows': len(voice), 'music_rows': len(music), 'per_split_class': args.per_split_class, 'output': str(args.output)}))
    if args.start_train:
        if not args.models or not args.checkpoint:
            raise ValueError('--start-train requires --models and --checkpoint')
        command = ['python', 'script.py', 'train', '--voice-csv', str(args.output/'voice.csv'), '--music-csv', str(args.output/'music.csv'), '--models', str(args.models), '--output', str(args.checkpoint)]
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
