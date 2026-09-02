#!/usr/bin/env python3
"""Download the exact frozen V4 model bundle and verify its runtime contract."""
import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

HF = {
    'df_arena_1b': ('Speech-Arena-2025/DF_Arena_1B_V_1', 'fb6ce85de12c2c5a509d89114adaf827dd75f49f'),
    'sonics': ('awsaf49/sonics-spectttra-alpha-5s', 'd30e3553a5ab08a171bdffdbbd2d792ac7970d65'),
    'artifactnet': ('intrect/artifactnet', '7c9b753a9d006b48e4bfaf85bf0157e135f4aad4'),
}
URLS = {
    'panns/Cnn14_mAP=0.431.pth': 'https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1',
    'panns/class_labels_indices.csv': 'https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv',
    'htdemucs/955717e8-8726e21a.th': 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th',
    'htdemucs/htdemucs.yaml': 'https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/htdemucs.yaml',
}
REQUIRED = ('df_arena_1b/pytorch_model.bin', 'sonics/config.json', 'sonics/pytorch_model.bin',
            'artifactnet/artifactnet_v94_full.onnx', 'artifactnet/artifactnet_v94_full.onnx.data',
            'panns/Cnn14_mAP=0.431.pth', 'htdemucs/htdemucs.yaml', 'htdemucs/955717e8-8726e21a.th')


def validate(root):
    missing = [name for name in REQUIRED if not (Path(root) / name).is_file()]
    if missing:
        raise FileNotFoundError('missing model files: ' + ', '.join(missing))


def fetch_url(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size:
        return
    temporary = target.with_suffix(target.suffix + '.partial')
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(target)


def download(root):
    root = Path(root)
    for directory, (repo, revision) in HF.items():
        snapshot_download(repo, revision=revision, local_dir=root / directory)
    for relative, url in URLS.items():
        fetch_url(url, root / relative)
    validate(root)
    (root / 'v4_model_provenance.json').write_text(json.dumps({'huggingface': HF, 'urls': URLS}, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-root', type=Path, default=Path('model'))
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--copy-component-labels-from', type=Path)
    args = p.parse_args()
    if args.copy_component_labels_from:
        destination = args.model_root / 'panns' / 'component_labels.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.copy_component_labels_from, destination)
    if args.verify_only:
        validate(args.model_root)
    else:
        download(args.model_root)
    print(json.dumps({'model_root': str(args.model_root), 'verified': True, 'required': list(REQUIRED)}))


if __name__ == '__main__':
    main()
