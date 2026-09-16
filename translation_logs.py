from datetime import date, datetime, timedelta
import json
import logging
from pathlib import Path
import re
import threading
from uuid import uuid4

log = logging.getLogger(__name__)
LOG_NAME = re.compile(r'(?:translation|manga-live)-(\d{4}-\d{2}-\d{2})\.log')
LEGACY_LOG_NAME = re.compile(r'(?:input|output)-(\d{4}-\d{2}-\d{2})\.jsonl')


class TranslationLogs:
    def __init__(self, directory, retention_days=7):
        self.directory = Path(directory)
        if type(retention_days) is not int or retention_days < 1:
            raise ValueError('로그 보관 기간은 1일 이상이어야 합니다.')
        self.retention_days = retention_days
        self._lock = threading.RLock()
        self._cleanup_date = None

    def cleanup(self, now=None):
        today = (now or datetime.now().astimezone()).date()
        cutoff = today - timedelta(days=self.retention_days - 1)
        with self._lock:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                root = self.directory.resolve()
                for path in self.directory.iterdir():
                    match = LOG_NAME.fullmatch(path.name) or LEGACY_LOG_NAME.fullmatch(path.name)
                    legacy_runtime = path.name == 'manga-live.log'
                    if (not match and not legacy_runtime) or path.is_symlink() or not path.is_file():
                        continue
                    try:
                        file_date = (datetime.fromtimestamp(path.stat().st_mtime).astimezone().date()
                                     if legacy_runtime else date.fromisoformat(match[1]))
                    except ValueError:
                        continue
                    if file_date < cutoff and path.resolve().parent == root:
                        try:
                            path.unlink()
                        except OSError:
                            log.warning('Expired log could not be removed: %s', path.name, exc_info=True)
                path = self.directory / f'translation-{today.isoformat()}.log'
                if path.is_symlink():
                    raise OSError('입출력 로그 경로가 심볼릭 링크입니다.')
                path.touch(exist_ok=True)
                self._cleanup_date = today
                return True
            except OSError:
                log.warning('Translation log initialization or cleanup failed', exc_info=True)
                return False

    def _write(self, stream, record):
        now = datetime.now().astimezone()
        with self._lock:
            try:
                if self._cleanup_date != now.date() and not self.cleanup(now):
                    return
                path = self.directory / f'translation-{now.date().isoformat()}.log'
                if path.is_symlink():
                    raise OSError('입출력 로그 경로가 심볼릭 링크입니다.')
                line = json.dumps({'timestamp': now.isoformat(timespec='seconds'), 'event': stream, **record},
                                  ensure_ascii=False) + '\n'
                with path.open('a', encoding='utf-8') as output:
                    output.write(line)
            except (OSError, UnicodeError):
                log.warning('Translation log write failed', exc_info=True)

    def input(self, text, provider, cached):
        request_id = uuid4().hex
        self._write('input', {'id': request_id, 'provider': provider, 'cached': cached, 'text': text})
        return request_id

    def output(self, request_id, text, provider, cached, error_type=None):
        record = {'id': request_id, 'provider': provider, 'cached': cached,
                  'status': 'error' if error_type else 'success'}
        if error_type:
            record['error_type'] = error_type
        else:
            record['text'] = text
        self._write('output', record)


class DailyRuntimeLogHandler(logging.Handler):
    def __init__(self, directory):
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(datetime.now().astimezone().date())
        with path.open('a', encoding='utf-8'):
            pass

    def _path(self, day):
        path = self.directory / f'manga-live-{day.isoformat()}.log'
        if path.is_symlink():
            raise OSError('실행 로그 경로가 심볼릭 링크입니다.')
        return path

    def emit(self, record):
        try:
            day = datetime.fromtimestamp(record.created).astimezone().date()
            with self._path(day).open('a', encoding='utf-8') as output:
                output.write(self.format(record) + '\n')
        except (OSError, UnicodeError):
            self.handleError(record)
