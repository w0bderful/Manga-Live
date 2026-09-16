"""Forward model transfer byte counts without changing models or cache locations."""
from contextlib import contextmanager
import importlib
import io
import threading
import time
from types import ModuleType

_hook_lock = threading.RLock()


class DownloadReporter:
    def __init__(self, name, callback, stopped=lambda: False):
        self.name, self.callback, self.stopped = name, callback, stopped
        self.last = 0.0
        self.active = True
        self.lock = threading.RLock()

    def report(self, received, total, force=False):
        with self.lock:
            if not self.active:
                return
            if self.stopped():
                raise InterruptedError('모델 다운로드를 취소했습니다.')
            received = max(0, int(received or 0))
            total = int(total) if total is not None and total > 0 else None
            if total is not None:
                received = min(received, total)
            now = time.monotonic()
            if force or now-self.last >= .1 or received == total:
                self.last = now
                if self.callback:
                    self.callback((self.name, received, total))


def download_label(info):
    name, received, total = info
    amount = f'{received / 1024**2:.1f} MiB'
    if total is None:
        return f'{name} 다운로드 · {amount} 받음 (전체 크기 확인 중)'
    return f'{name} 다운로드 · {amount} / {total / 1024**2:.1f} MiB'


@contextmanager
def easyocr_downloads(module, reporter):
    utils = getattr(module, 'utils', None)
    if not isinstance(utils, ModuleType):
        yield
        return
    owner = threading.get_ident()
    with _hook_lock:
        original = utils.urlretrieve
        def retrieve(url, filename=None, reporthook=None, data=None):
            if threading.get_ident() != owner:
                return original(url, filename, reporthook, data)
            def hook(blocks, size, total):
                reporter.report(blocks*size, total, force=blocks == 0)
            return original(url, filename, hook, data)
        utils.urlretrieve = retrieve
        try:
            yield
        finally:
            utils.urlretrieve = original
            reporter.active = False


@contextmanager
def huggingface_downloads(reporter):
    """Bridge the installed HF HTTP/Xet byte-bar factory on this loader thread.

    Transformers and MangaOCR own the download calls. This scoped adapter keeps
    their cache/resume/model choices and leaves unrelated threads unchanged.
    """
    progress = importlib.import_module('huggingface_hub.utils.tqdm')
    xet = importlib.import_module('huggingface_hub.utils._xet_progress_reporting')
    from tqdm.auto import tqdm
    owner = threading.get_ident()

    class ByteProgress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.pop('name', None)
            kwargs['disable'] = False
            kwargs['file'] = io.StringIO()
            super().__init__(*args, **kwargs)
            reporter.report(self.n, self.total, force=True)

        def display(self, *args, **kwargs):
            return True

        def update(self, n=1):
            result = super().update(n)
            reporter.report(self.n, self.total)
            return result

        def refresh(self, *args, **kwargs):
            if hasattr(self, 'n'):
                reporter.report(self.n, self.total)
            return super().refresh(*args, **kwargs)

        def close(self):
            # Cleanup must not raise on cancellation or turn a failed transfer
            # into a false 100% completion.
            if hasattr(self, 'n') and not reporter.stopped():
                reporter.report(self.n, self.total, force=True)
            super().close()

    with _hook_lock:
        original = progress._create_progress_bar
        original_init = xet.XetDownloadProgressReporter.__init__
        original_update = xet.XetDownloadProgressReporter.update_progress
        def create(*, cls, log_level, name=None, **kwargs):
            if threading.get_ident() == owner and kwargs.get('unit') == 'B':
                return ByteProgress(**kwargs)
            return original(cls=cls, log_level=log_level, name=name, **kwargs)
        def xet_init(bar, *args, **kwargs):
            original_init(bar, *args, **kwargs)
            if threading.get_ident() == owner:
                bar._manga_live_reporter = reporter
                reporter.report(0, None, force=True)
        def xet_update(bar, group, items=None):
            original_update(bar, group, items)
            active = getattr(bar, '_manga_live_reporter', None)
            if active is not None:
                # Xet compresses/deduplicates data, so reconstructed file size
                # is not a valid denominator for actual network bytes.
                active.report(group.total_transfer_bytes_completed, None)
        progress._create_progress_bar = create
        xet.XetDownloadProgressReporter.__init__ = xet_init
        xet.XetDownloadProgressReporter.update_progress = xet_update
        try:
            yield
        finally:
            progress._create_progress_bar = original
            xet.XetDownloadProgressReporter.__init__ = original_init
            xet.XetDownloadProgressReporter.update_progress = original_update
            reporter.active = False
