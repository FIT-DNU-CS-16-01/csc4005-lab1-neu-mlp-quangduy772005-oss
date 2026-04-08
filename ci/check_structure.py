from __future__ import annotations

from pathlib import Path
import sys

REQUIRED = [
    'README.md',
    'REPORT_TEMPLATE.md',
    'requirements.txt',
    'configs/baseline.json',
    'src/dataset.py',
    'src/model.py',
    'src/train.py',
    'src/utils.py',
    'ci/smoke_train.py',
    '.github/workflows/validate-lab1.yml',
]

repo_root = Path(__file__).resolve().parents[1]
missing = [item for item in REQUIRED if not (repo_root / item).exists()]
for item in missing:
    print(f'MISSING: {item}')
if (repo_root / 'data').exists():
    print('MISSING REQUIREMENT: thư mục data không được tồn tại trong starter kit mới')
    missing.append('data/')
if missing:
    sys.exit(1)
print('Structure check passed.')
