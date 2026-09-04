#!/usr/bin/env bash
set -euo pipefail

archive="${1:-submit.zip}"
python3 - "$archive" <<'PY'
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
roots = [Path("model"), Path("script.py"), Path("requirements.txt")]
missing = [str(path) for path in roots if not path.exists()]
if missing:
    raise SystemExit(f"Missing submission paths: {missing}")

archive.unlink(missing_ok=True)
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for root in roots:
        paths = root.rglob("*") if root.is_dir() else (root,)
        for path in paths:
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                zf.write(path)

with zipfile.ZipFile(archive) as zf:
    names = [name.rstrip('/') for name in zf.namelist()]
    required = {'model', 'script.py', 'requirements.txt'}
    top_level = {name.split('/', 1)[0] for name in names if name}
    missing = required - top_level
    unexpected = top_level - required
    if missing or unexpected:
        raise SystemExit(
            f'Invalid archive structure: missing={sorted(missing)} '
            f'unexpected={sorted(unexpected)}'
        )

print(f"Created valid submission archive: {archive}")
PY
