import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from script import build_index


def wav(path):
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)


def test_index_keeps_real_and_fake_rows_per_modality(tmp_path):
    vr, mr, mf = (tmp_path / name for name in ('vr', 'mr', 'mf'))
    for directory in (vr, mr, mf): directory.mkdir()
    wav(vr / 'v.wav'); wav(mr / 'm.wav'); wav(mf / 'f.mp3.wav')
    fake_wav = tmp_path / 'fake.wav'; wav(fake_wav)
    archive = tmp_path / 'voice.zip'
    with zipfile.ZipFile(archive, 'w') as z: z.write(fake_wav, 'fake_generated.wav')
    out = tmp_path / 'out'
    build_index(argparse.Namespace(source_dir=out, voice_real=vr, voice_fake=archive, music_real=mr, music_fake=mf))
    for name in ('voice', 'music'):
        with (out / f'{name}.csv').open() as f:
            assert {row['label'] for row in csv.DictReader(f)} == {'0', '1'}
