"""Integrated Manga Live entrypoint: install libraries, then run in this process."""
import ctypes
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import queue
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
import multiprocessing
import self_update

BLOCK = 1024 * 1024


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel.is_set():
        raise Cancelled()


def checksum(path, cancel):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        while data := handle.read(BLOCK):
            check_cancel(cancel)
            result.update(data)
    return result.hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or '\\' in name or ':' in name
            or any(part in ('..', '.') for part in name.split('/'))):
        raise ValueError('설치 파일 경로가 올바르지 않습니다.')
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError('설치 폴더 밖으로 파일을 저장할 수 없습니다.')
    return target


def download(asset, cache, cancel, progress, opener=urllib.request.urlopen):
    destination = safe_path(cache, asset['name'])
    partial = safe_path(cache, asset['name']+'.partial')
    expected_size = asset['size']
    if destination.exists():
        if destination.stat().st_size == expected_size and checksum(destination, cancel) == asset['sha256']:
            return destination
        destination.unlink()
    last_error = None
    for attempt in range(3):
        check_cancel(cancel)
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= expected_size:
                if offset == expected_size and checksum(partial, cancel) == asset['sha256']:
                    partial.replace(destination)
                    return destination
                partial.unlink()
                offset = 0
            headers = {'User-Agent': 'MangaLive-Launcher/1.0', 'Accept-Encoding': 'identity'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            with opener(urllib.request.Request(asset['url'], headers=headers), timeout=30) as response:
                status = response.getcode()
                if status == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        partial.unlink(missing_ok=True)
                        raise ValueError('다운로드 재개 위치가 일치하지 않습니다.')
                elif status == 200:
                    offset = 0
                else:
                    raise OSError(f'다운로드 서버 오류: HTTP {status}')
                with partial.open('ab' if offset else 'wb') as handle:
                    while data := response.read(BLOCK):
                        check_cancel(cancel)
                        offset += len(data)
                        if offset > expected_size:
                            raise ValueError('다운로드 파일 크기가 올바르지 않습니다.')
                        handle.write(data)
                        progress(offset, expected_size)
            if partial.stat().st_size != expected_size:
                raise OSError('다운로드 연결이 끊겼습니다.')
            if checksum(partial, cancel) != asset['sha256']:
                partial.unlink()
                raise ValueError('다운로드 파일 검증에 실패했습니다.')
            partial.replace(destination)
            return destination
        except Cancelled:
            raise
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as exc:
            last_error = exc
            if attempt < 2 and cancel.wait(1):
                raise Cancelled()
    raise RuntimeError(f'필요한 파일을 받지 못했습니다. 인터넷 연결을 확인하고 다시 시도하세요.\n{last_error}')


def installed(directory, manifest, cancel, progress):
    if not (directory/'.complete').is_file():
        return False
    for index, (name, expected) in enumerate(manifest['files'].items()):
        check_cancel(cancel)
        path = safe_path(directory, name)
        if (not path.is_file() or path.stat().st_size != expected['size']
                or checksum(path, cancel) != expected['sha256']):
            return False
        progress('설치된 파일 확인 중', (index+1)/len(manifest['files'])*100)
    return True


def install(manifest, home, cancel, progress):
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:24]
    store = home/'.manga-live-runtime'
    store.mkdir(parents=True, exist_ok=True)
    if store.is_symlink() or store.resolve().parent != home.resolve():
        raise ValueError('설치 폴더 경로가 올바르지 않습니다.')
    # The preparation lock prevents another installer from using these folders.
    for abandoned in store.glob('install-*'):
        if (abandoned.is_dir() and not abandoned.is_symlink()
                and abandoned.resolve().parent == store.resolve()
                and len(abandoned.name) == len('install-')+32
                and all(c in '0123456789abcdef' for c in abandoned.name[8:])):
            try:
                shutil.rmtree(abandoned)
            except OSError:
                pass
    destination = store/identity
    if destination.is_symlink() or destination.resolve().parent != store.resolve():
        raise ValueError('실행 파일 설치 경로가 올바르지 않습니다.')
    if installed(destination, manifest, cancel, progress):
        cleanup_downloads(store/(identity+'-downloads'))
        return safe_path(destination, manifest['entrypoint'])
    cache = store/(identity+'-downloads')
    if cache.is_symlink() or cache.resolve().parent != store.resolve():
        raise ValueError('다운로드 폴더 경로가 올바르지 않습니다.')
    cache.mkdir(exist_ok=True)
    remaining = sum(a['size'] for a in manifest['assets'] if not (cache/a['name']).exists())
    required = sum(v['size'] for v in manifest['files'].values()) + remaining + 128*BLOCK
    if shutil.disk_usage(store).free < required:
        raise OSError(f'설치 공간이 부족합니다. 약 {required/1024**3:.1f} GB의 여유 공간이 필요합니다.')
    archives = []
    for index, asset in enumerate(manifest['assets']):
        archives.append(download(asset, cache, cancel, lambda done,total: progress(
            f'필요한 파일 다운로드 {index+1}/{len(manifest["assets"])} · {done/BLOCK:.0f}/{total/BLOCK:.0f} MB',
            (index+done/total)/len(manifest['assets'])*100)))
    stage = store/('install-'+uuid.uuid4().hex)
    stage.mkdir()
    try:
        seen = set()
        for archive in archives:
            with zipfile.ZipFile(archive) as handle:
                for member in handle.infolist():
                    check_cancel(cancel)
                    name = member.filename
                    if member.is_dir() or name not in manifest['files'] or name in seen:
                        raise ValueError('설치 압축 파일의 구성 정보가 일치하지 않습니다.')
                    if (member.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('바로가기 파일은 설치할 수 없습니다.')
                    expected = manifest['files'][name]
                    if member.file_size != expected['size']:
                        raise ValueError('설치 파일의 크기가 일치하지 않습니다.')
                    target = safe_path(stage, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with handle.open(member) as src, target.open('wb') as out:
                        while data := src.read(BLOCK):
                            check_cancel(cancel)
                            out.write(data)
                            digest.update(data)
                    if digest.hexdigest() != expected['sha256']:
                        raise ValueError('설치 파일 검증에 실패했습니다.')
                    seen.add(name)
                    progress('다운로드 완료 · 설치 중', len(seen)/len(manifest['files'])*100)
        if seen != set(manifest['files']):
            raise ValueError('필요한 설치 파일이 누락되었습니다.')
        (stage/'.complete').write_text(identity, encoding='ascii')
        # Only replace this launcher's exact runtime cache, never user settings/models.
        if destination.exists():
            if destination.resolve().parent != store.resolve() or destination.is_symlink():
                raise ValueError('설치 경로가 올바르지 않습니다.')
            shutil.rmtree(destination)
        stage.replace(destination)
        for archive in archives:
            archive.unlink(missing_ok=True)
        cleanup_downloads(cache)
        return safe_path(destination, manifest['entrypoint'])
    finally:
        if stage.exists() and stage.resolve().parent == store.resolve() and not stage.is_symlink():
            shutil.rmtree(stage)


def cleanup_downloads(cache):
    if cache.is_symlink():
        return
    try:
        if cache.exists():
            for path in cache.iterdir():
                if path.is_file() and (path.suffix == '.zip' or path.name.endswith('.zip.partial')):
                    path.unlink()
            cache.rmdir()
    except OSError:
        pass  # A cleanup failure must not stop a verified installation.


_dll_handles = []


def activate_runtime(entrypoint, home):
    root = entrypoint.parent.resolve()
    sys.path.insert(0, str(root))
    # Packages used by the installer (ctypes, urllib, multiprocessing, ...) are
    # already loaded from its small archive. Expose their full runtime modules.
    for name, module in list(sys.modules.items()):
        paths = getattr(module, '__path__', None)
        directory = root.joinpath(*name.split('.'))
        if paths is not None and directory.is_dir() and str(directory) not in paths:
            module.__path__ = [str(directory), *paths]
    sys._MEIPASS = str(root)
    os.environ['MANGA_LIVE_RUNTIME'] = str(root)
    os.environ['MANGA_LIVE_DATA_DIR'] = str(home)
    os.environ['MANGA_LIVE_LAUNCHER'] = str(Path(sys.executable).resolve())
    os.environ['PATH'] = str(root)+os.pathsep+os.environ.get('PATH', '')
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetDllDirectoryW(str(root))
        for directory in (root, root/'PyQt6'/'Qt6'/'bin', root/'torch'/'lib'):
            if directory.is_dir():
                _dll_handles.append(os.add_dll_directory(str(directory)))
    for hook in sorted((root/'runtime-hooks').glob('*.py')):
        exec(compile(hook.read_bytes(), str(hook), 'exec'), {'__file__':str(hook)})


def launch(entrypoint, home):
    activate_runtime(entrypoint, home)
    multiprocessing.freeze_support()
    import importlib
    application = importlib.import_module('main')
    return application.main()


def main():
    import tkinter as tk
    from tkinter import ttk, messagebox
    home = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    resource = Path(__file__).resolve().parent
    if len(sys.argv) == 3 and sys.argv[1] == '--apply-update':
        try:
            self_update.replace_pending(sys.argv[2])
        except Exception as exc:
            window = tk.Tk(); window.withdraw()
            messagebox.showerror('Manga Live 업데이트 실패', str(exc), parent=window)
            window.destroy()
        return
    # Capture workers reuse this same executable and already installed libraries.
    if '--multiprocessing-fork' in sys.argv or '-c' in sys.argv:
        runtime = Path(os.environ['MANGA_LIVE_RUNTIME'])
        activate_runtime(runtime/'main.py', Path(os.environ['MANGA_LIVE_DATA_DIR']))
        multiprocessing.freeze_support()
        raise RuntimeError('지원하지 않는 작업 프로세스 인수입니다.')
    try:
        if self_update.handoff_pending(home):
            return
    except Exception as exc:
        window = tk.Tk(); window.withdraw()
        messagebox.showwarning('Manga Live 업데이트', str(exc)+'\n기존 버전으로 계속합니다.', parent=window)
        window.destroy()
    threading.Thread(target=self_update.cleanup_updates, args=(home,), daemon=True).start()
    window = tk.Tk()
    window.title('Manga Live 준비')
    window.geometry('520x185')
    window.resizable(False, False)
    status = tk.StringVar(value='처음 실행할 때 필요한 파일을 자동으로 받습니다.')
    ttk.Label(window, textvariable=status, wraplength=480).pack(padx=20, pady=(25,15), anchor='w')
    bar = ttk.Progressbar(window, maximum=100, length=480)
    bar.pack(padx=20)
    events = queue.Queue()
    cancel = threading.Event()
    lock = None
    try:
        import msvcrt
        lock_path = home/'.manga-live-runtime'/'launcher.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = lock_path.open('a+b')
        if lock.seek(0,2) == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        messagebox.showerror('Manga Live', '다른 준비 창이 실행 중이거나 폴더에 쓸 수 없습니다.\n쓰기 가능한 폴더에서 다시 실행하세요.', parent=window)
        if lock:
            lock.close()
        window.destroy()
        return

    last_update = [0.0]
    ready = []
    def progress(text, value):
        now = time.monotonic()
        if now-last_update[0] >= .1 or value == 100:
            last_update[0] = now
            events.put(('progress', text, value))

    def work():
        try:
            manifest = json.loads((resource/'runtime-manifest.json').read_text(encoding='utf-8'))
            executable = install(manifest, home, cancel, progress)
            check_cancel(cancel)
            ready.append(executable)
            events.put(('done',))
        except Cancelled:
            events.put(('done',))
        except Exception as exc:
            events.put(('error', str(exc)))

    def stop():
        cancel.set()
        status.set('취소 중… 다음 실행에서 다운로드를 이어갑니다.')
        button.configure(state='disabled')

    def start():
        cancel.clear()
        window.protocol('WM_DELETE_WINDOW', stop)
        button.configure(text='취소', command=stop, state='normal')
        status.set('필요한 파일 확인 중…')
        threading.Thread(target=work, daemon=True).start()

    button = ttk.Button(window, text='취소', command=stop)
    button.pack(pady=15)
    window.protocol('WM_DELETE_WINDOW', stop)

    def poll():
        try:
            while True:
                event = events.get_nowait()
                if event[0] == 'progress':
                    if not cancel.is_set():
                        status.set(event[1]); bar['value'] = event[2]
                elif event[0] == 'done':
                    window.destroy()
                    return
                elif event[0] == 'error':
                    if cancel.is_set():
                        window.destroy()
                        return
                    status.set('준비하지 못했습니다. 연결·저장 공간·폴더 권한을 확인하세요.')
                    button.configure(text='다시 시도', command=start, state='normal')
                    window.protocol('WM_DELETE_WINDOW', window.destroy)
                    messagebox.showerror('Manga Live 준비 실패', event[1], parent=window)
        except queue.Empty:
            pass
        window.after(100, poll)
    start()
    window.after(100, poll)
    try:
        window.mainloop()
    finally:
        cancel.set()
        lock.close()
    if ready:
        try:
            return launch(ready[0], home)
        except Exception:
            import traceback
            log_dir = home/'logs'
            log_dir.mkdir(exist_ok=True)
            with (log_dir/('manga-live-'+time.strftime('%Y-%m-%d')+'.log')).open('a',encoding='utf-8') as handle:
                traceback.print_exc(file=handle)
            window = tk.Tk(); window.withdraw()
            messagebox.showerror('Manga Live 실행 실패', '프로그램을 시작하지 못했습니다. logs 폴더의 실행 로그를 확인하세요.',parent=window)
            window.destroy()


if __name__ == '__main__':
    main()
