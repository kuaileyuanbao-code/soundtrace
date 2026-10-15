"""Explicit, source-only inventory/check/export; never install or load models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'delivery/source_selection.json'
MANIFEST = ROOT / 'delivery/source_manifest.json'
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
BLOCKED_PARTS = {'jobs', 'voiceprints', 'logs', 'models', 'checkpoints', '.git', '__pycache__'}
TEXT_SUFFIXES = {'.py', '.js', '.cjs', '.html', '.css', '.svg', '.webmanifest', '.md',
                 '.json', '.bat', '.txt', '.patch'}
SECRET_PATTERNS = [
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    r'\bAKIA[0-9A-Z]{16}\b',
    r'\bgh[pousr]_[A-Za-z0-9]{36,}\b',
    r'\bgithub_pat_[A-Za-z0-9_]{40,}\b',
    r'\bsk-[A-Za-z0-9]{32,}\b',
]
AUTHOR_MACHINE_PATH = re.compile(r'(?:C:[\\/]+Users[\\/]|D:[\\/]+OneDrive[\\/])', re.I)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def safe_file(name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or '\\' in name or ':' in name
            or '..' in relative.parts or relative.as_posix() != name):
        raise ValueError(f'Unsafe source path: {name!r}')
    if any(p.lower() in BLOCKED_PARTS or p.lower().startswith('.venv') for p in relative.parts):
        raise ValueError(f'Private/runtime path is not source: {name}')
    if Path(name).suffix not in TEXT_SUFFIXES and name not in {'.gitignore', '.gitattributes'}:
        raise ValueError(f'Unsupported source type: {name}')
    path = ROOT.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError(f'Missing or outside-workspace source: {name}')
    current = path
    while current != ROOT:
        if current.is_symlink() or (current.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
                                   if hasattr(current.lstat(), 'st_file_attributes') else False):
            raise ValueError(f'Reparse/symlink source is not permitted: {name}')
        current = current.parent
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES or b'\0' in data:
        raise ValueError(f'Unexpected large/binary source: {name}')
    text = data.decode('utf-8-sig')
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, text):
            raise ValueError(f'Potential credential in {name}; inspect locally (value suppressed)')
    if AUTHOR_MACHINE_PATH.search(text):
        raise ValueError(f'Author-machine absolute path in {name}')
    if path.suffix == '.py':
        compile(text, name, 'exec')
    return data


def selection():
    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    result = {}
    for category, names in policy['groups'].items():
        for name in names:
            if name.casefold() in {key.casefold() for key in result}:
                raise ValueError(f'Duplicate source: {name}')
            result[name] = category
    return dict(sorted(result.items()))


def refresh():
    files = []
    for name, category in selection().items():
        data = safe_file(name)
        files.append(dict(path=name, category=category, bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest()))
    if sum(item['bytes'] for item in files) > MAX_TOTAL_BYTES:
        raise ValueError('Source selection unexpectedly exceeds 10 MiB')
    MANIFEST.write_text(json.dumps(dict(schema_version=1, observed_at=now(),
        scope='Source-only baseline; not a runtime package; no private recordings or model weights',
        manifest_self_hash_omitted=True, files=files), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return files


def check():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    files = manifest['files']
    selected = selection()
    if len(files) != len(selected) or {item['path']: item['category'] for item in files} != selected:
        raise ValueError('Selection and manifest differ; review then explicitly refresh')
    for item in files:
        data = safe_file(item['path'])
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f"Source changed since manifest: {item['path']}")
    if sum(item['bytes'] for item in files) > MAX_TOTAL_BYTES:
        raise ValueError('Source inventory exceeds size limit')
    safe_file('delivery/source_manifest.json')
    return files


def export(files):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    names = [item['path'] for item in files] + ['delivery/source_manifest.json']
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    if set(filter(None, tracked)) != set(names):
        raise ValueError('Tracked files differ from the reviewed source selection')
    for name in names:
        committed = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
        if committed != safe_file(name):
            raise ValueError(f'Uncommitted source change: {name}')
    parent = ROOT / 'delivery/exports'
    parent.mkdir(exist_ok=True)
    if not parent.resolve().is_relative_to(ROOT.resolve()) or parent.is_symlink():
        raise ValueError('Export directory is outside the intended workspace')
    destination = parent / ('soundtrace-source-' + commit[:12])
    destination.mkdir(exist_ok=False)
    for name in names:
        target = destination.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = safe_file(name)
        target.write_bytes(data)
        if target.read_bytes() != data:
            raise IOError(f'Export mismatch: {name}')
    snapshot = dict(created_at=now(), commit=commit, source_file_count=len(names),
                    source_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                    scope='Source snapshot only. Third-party sources, environments, model weights and private data excluded.')
    (destination / 'SOURCE_SNAPSHOT.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if len(list(p for p in destination.rglob('*') if p.is_file())) != len(names) + 1:
        raise IOError('Unexpected extra files in export')
    return destination.relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--refresh', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--export', action='store_true')
    args = parser.parse_args()
    if args.refresh:
        refresh()
    files = check()
    result = dict(ok=True, files=len(files), source_bytes=sum(item['bytes'] for item in files))
    if args.export:
        result['export'] = export(files)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
