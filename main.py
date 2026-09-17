"""Application entrypoint: logging, Qt lifetime and the main window."""
from runtime_paths import APP_DIR as ROOT, RESOURCE_DIR
import ctypes
import logging
import sys
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QLabel
from translation_logs import TranslationLogs, DailyRuntimeLogHandler
from controller import Controller
log = logging.getLogger(__name__)

def configure_logging():
    options = dict(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        directory = ROOT / 'logs'
        logging.basicConfig(handlers=[DailyRuntimeLogHandler(directory)], **options)
    except OSError:
        handler = logging.StreamHandler(sys.stderr) if sys.stderr is not None else logging.NullHandler()
        logging.basicConfig(handlers=[handler], **options)
        log.warning('Log file unavailable; continuing without file logging', exc_info=True)
        return '로그 파일을 열 수 없어 파일 기록 없이 실행합니다. logs 폴더의 권한과 잠금 상태를 확인하세요.'
    return ''

def main(on_ready=None):
    if sys.platform != 'win32':
        raise SystemExit('이 프로그램은 Windows 전용입니다.')
    logging_warning = configure_logging()
    io_logger = TranslationLogs(ROOT / 'logs')
    if not io_logger.cleanup():
        logging_warning = '\n'.join(filter(None, [logging_warning,
            '입출력 로그를 준비하지 못했습니다. logs 폴더의 권한을 확인하세요.']))
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('MangaLive.Desktop')
    except OSError:
        log.warning('Windows app identity could not be set', exc_info=True)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(RESOURCE_DIR / 'assets' / 'manga-live.ico')))
    app.setQuitOnLastWindowClosed(False)
    window = Controller(io_logger=io_logger)
    if logging_warning:
        note = QLabel(logging_warning)
        note.setWordWrap(True)
        window.layout().addWidget(note)
    window.show()
    if on_ready is not None:
        on_ready()
    return app.exec()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
