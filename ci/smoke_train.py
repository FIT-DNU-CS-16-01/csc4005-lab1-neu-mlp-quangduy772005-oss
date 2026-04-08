from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_PREFIXES = [
    'crazing',
    'inclusion',
    'patches',
    'pitted_surface',
    'rolled-in_scale',
    'scratches',
]


def build_fake_dataset(root: Path) -> Path:
    images_dir = root / 'mini_neu_flat'
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for prefix in CLASS_PREFIXES:
        for idx in range(10):
            arr = (rng.random((32, 32)) * 255).astype('uint8')
            Image.fromarray(arr, mode='L').save(images_dir / f'{prefix}_{idx}.jpg')
    return images_dir


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tmp_root = Path(tempfile.mkdtemp(prefix='lab1_smoke_'))
    try:
        data_dir = build_fake_dataset(tmp_root)
        cmd = [
            sys.executable,
            '-m',
            'src.train',
            '--data_dir',
            str(data_dir),
            '--run_name',
            'smoke_test',
            '--epochs',
            '1',
            '--batch_size',
            '8',
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)
        print('Smoke training passed.')
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        shutil.rmtree(repo_root / 'outputs' / 'smoke_test', ignore_errors=True)


if __name__ == '__main__':
    main()
