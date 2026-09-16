"""Stage updates and replace the integrated EXE after its old process exits."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid


def pending_path(home):
    return Path(home)/'.manga-live-runtime'/'updates'/'pending.json'


def read_pending(home):
    path = pending_path(home)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    digest = data.get('sha256', '')
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('업데이트 검증 정보가 올바르지 않습니다.')
    home = Path(home).resolve()
    target = home/data['target']
    candidate = path.parent/digest/'Manga Live.exe'
    if (target.parent != home or target.is_symlink() or target.suffix.lower() != '.exe'
            or candidate.is_symlink() or not candidate.resolve().is_relative_to(home)
            or Path(data['target']).name != data['target']):
        raise ValueError('업데이트 경로가 올바르지 않습니다.')
    return data, target, candidate


def spawn(executable, arguments=()):
    environment = dict(os.environ, PYINSTALLER_RESET_ENVIRONMENT='1')
    # A new EXE must locate its own runtime and data, not inherit the helper's paths.
    for name in ('MANGA_LIVE_RUNTIME', 'MANGA_LIVE_DATA_DIR', 'MANGA_LIVE_LAUNCHER'):
        environment.pop(name, None)
    import ctypes
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetDllDirectoryW(None)
    try:
        return subprocess.Popen([str(executable), *map(str, arguments)],
                                cwd=executable.parent, env=environment)
    finally:
        if sys.platform == 'win32' and getattr(sys, 'frozen', False):
            ctypes.windll.kernel32.SetDllDirectoryW(sys._MEIPASS)


def handoff_pending(home):
    from bootstrap import checksum
    pending = read_pending(home)
    if pending is None:
        return False
    data, target, candidate = pending
    cancel = threading.Event()
    if checksum(target, cancel) == data['sha256']:
        pending_path(home).unlink(missing_ok=True)
        return False
    if checksum(candidate, cancel) != data['sha256']:
        raise ValueError('업데이트 파일 검증에 실패했습니다. 버전 확인으로 다시 받아 주세요.')
    spawn(candidate, ['--apply-update', str(pending_path(home))])
    return True


def replace_pending(receipt, executable=None, timeout=120):
    from bootstrap import checksum
    receipt = Path(receipt).resolve()
    home = receipt.parent.parent.parent
    if receipt != pending_path(home):
        raise ValueError('업데이트 기록 경로가 올바르지 않습니다.')
    data, target, candidate = read_pending(home)
    executable = Path(executable or sys.executable).resolve()
    if executable != candidate.resolve() or checksum(candidate, threading.Event()) != data['sha256']:
        raise ValueError('업데이트 실행 파일 검증에 실패했습니다.')
    temporary = home/('.manga-live-launcher-'+uuid.uuid4().hex+'.tmp')
    try:
        shutil.copyfile(candidate, temporary)
        if checksum(temporary, threading.Event()) != data['sha256']:
            raise ValueError('업데이트 복사 검증에 실패했습니다.')
        deadline = time.monotonic()+timeout
        while True:
            try:
                temporary.replace(target)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise OSError('이전 프로그램이 종료되지 않았습니다. Manga Live를 종료한 후 다시 실행하세요.')
                time.sleep(.25)
        receipt.unlink(missing_ok=True)
        spawn(target)
    finally:
        temporary.unlink(missing_ok=True)


def cleanup_updates(home):
    """Delete completed update downloads; keep a verified pending update intact."""
    root = pending_path(home).parent
    if not root.exists() or root.is_symlink() or not root.resolve().is_relative_to(Path(home).resolve()):
        return
    try:
        pending = read_pending(home)
    except (OSError, ValueError, KeyError, TypeError):
        return
    keep = pending[0]['sha256'] if pending else None
    for directory in root.iterdir():
        if (directory.name == keep or not re.fullmatch('[0-9a-f]{64}', directory.name)
                or directory.is_symlink() or not directory.is_dir()):
            continue
        for _ in range(40):
            try:
                shutil.rmtree(directory)
                break
            except FileNotFoundError:
                break
            except OSError:
                time.sleep(.25)
