import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from script import build_index


def wav(path):
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)


def test_index_groups_musiccaps_offsets_by_original_video(tmp_path):
    vr, mf = tmp_path / 'vr', tmp_path / 'mf'
    mr = tmp_path / 'mr'; vr.mkdir(); mr.mkdir(); mf.mkdir()
    wav(vr / 'voice.wav'); wav(mr / 'abc_DEF_30_1.wav'); wav(mr / 'abc_DEF_50_2.wav'); wav(mf / 'fake_1_suno_0.wav')
    archive = tmp_path / 'voice.zip'
    import zipfile
    with zipfile.ZipFile(archive, 'w') as z: z.write(vr / 'voice.wav', 'voice_generated.wav')
    out = tmp_path / 'out'
    build_index(argparse.Namespace(source_dir=out, voice_real=vr, voice_fake=archive, music_real=mr, music_fake=mf))
    with (out / 'music.csv').open() as f:
        real = [r for r in csv.DictReader(f) if r['label'] == '0']
    assert {r['group'] for r in real} == {'abc_DEF'}
