#!/usr/bin/env bash
set -euo pipefail

archive="${1:-submit.zip}"

for required in model script.py requirements.txt; do
    if [[ ! -e "$required" ]]; then
        printf 'Missing required submission path: %s\n' "$required" >&2
        exit 1
    fi
done

rm -f "$archive"
zip -qr "$archive" model script.py requirements.txt

python3 - "$archive" <<'PY'
import sys
import zipfile

archive = sys.argv[1]
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

print(f'Created valid submission archive: {archive}')
PY
