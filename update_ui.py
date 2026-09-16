import logging
import threading
import time

from PyQt6.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QMessageBox

from app_version import VERSION
import update_checker as updates

log = logging.getLogger(__name__)


class VersionUpdater(QObject):
    checked = pyqtSignal(object,str,bool)
    downloaded = pyqtSignal(object,str)
    progress = pyqtSignal(int)

    def __init__(self, parent, home):
        super().__init__(parent)
        self.home = home
        self.closed = threading.Event()
        self.busy = False
        self.retry_after = 0
        self.panel = QWidget(parent)
        row = QHBoxLayout(self.panel)
        row.setContentsMargins(0,0,0,0)
        self.label = QLabel('v'+VERSION)
        try:
            pending = updates.load_state(home).get('pending_version','')
            if pending and updates.version_key(pending) > updates.version_key(VERSION):
                self.label.setText('v'+pending+' · 다음 실행에 적용')
        except (OSError,ValueError):
            pass
        self.label.setWordWrap(True)
        row.addWidget(self.label,1)
        self.button = QPushButton('버전 확인')
        self.button.clicked.connect(lambda:self.check(True))
        row.addWidget(self.button)
        self.timer = QTimer(self)
        self.timer.setInterval(60000)
        self.timer.timeout.connect(self.check)
        self.initial_timer = QTimer(self)
        self.initial_timer.setSingleShot(True)
        self.initial_timer.setInterval(3000)
        self.initial_timer.timeout.connect(self.check)
        self.checked.connect(self.finished_check)
        self.downloaded.connect(self.finished_download)
        self.progress.connect(lambda value:self.label.setText(f'업데이트 다운로드 · {value}%'))

    def start(self):
        self.timer.start()
        self.initial_timer.start()

    def close(self):
        self.closed.set()
        self.timer.stop()
        self.initial_timer.stop()

    def check(self, manual=False):
        if self.closed.is_set() or self.busy:
            return
        if not manual:
            if time.time() < self.retry_after:
                return
            try:
                if not updates.check_due(updates.load_state(self.home)):
                    return
            except (OSError,ValueError):
                log.warning('Update preferences unavailable',exc_info=True)
        self.busy = True
        self.button.setEnabled(False)
        self.label.setText('새 버전 확인 중…')
        def worker():
            result,error = None,''
            try:
                result = updates.fetch_latest()
            except Exception as exc:
                detail = updates.error_detail(exc)
                log.warning('Release check failed (%s): %s',type(exc).__name__,detail)
                error = '버전을 확인하지 못했습니다. '+detail
            if not self.closed.is_set():
                self.checked.emit(result,error,manual)
        threading.Thread(target=worker,daemon=True).start()

    def finished_check(self, result, error, manual):
        if self.closed.is_set():
            return
        self.busy = False
        self.button.setEnabled(True)
        if error:
            self.retry_after = time.time()+3600
            self.label.setText(error)
            if manual:
                QMessageBox.warning(self.parent(),'버전 확인',error)
            return
        self.retry_after = 0
        save_note = ''
        try:
            updates.save_state(self.home,{'last_checked':time.time()})
        except (OSError,ValueError):
            self.retry_after = time.time()+3600
            save_note = ' · 확인 시각 저장 실패'
            log.warning('Update check time could not be saved',exc_info=True)
        if result is None:
            self.label.setText('v'+VERSION+' · 최신 버전'+save_note)
            return
        self.label.setText('v'+VERSION+' · v'+result['version']+' 사용 가능'+save_note)
        target = updates.launcher_path(self.home)
        if target is not None and target.is_file():
            from self_update import read_pending
            try:
                pending = read_pending(self.home)
                if pending and pending[0]['sha256']==result['sha256']:
                    self.label.setText('v'+result['version']+' · 다음 실행에 적용')
                    return
            except (OSError,ValueError,KeyError,TypeError):
                pass
        if target is None:
            message = f'새 버전 v{result["version"]}이 있습니다.\n소스 실행에서는 릴리스 페이지에서 새 EXE를 받아 주세요.\n릴리스 페이지를 열까요?'
        else:
            message = f'새 버전 v{result["version"]}을 다운로드할까요?\n설정과 API 키는 유지되며, 다음 실행부터 적용됩니다.'
        if QMessageBox.question(self.parent(),'업데이트 가능',message,
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.Yes)!=QMessageBox.StandardButton.Yes:
            return
        if target is None:
            QDesktopServices.openUrl(QUrl(result['release_url']))
            return
        self.busy = True
        self.button.setEnabled(False)
        self.label.setText('업데이트 다운로드 중…')
        def worker():
            error = ''
            last = [-1]
            def show_progress(done,total):
                value = int(done/total*100)
                if value!=last[0] and not self.closed.is_set():
                    last[0]=value
                    self.progress.emit(value)
            try:
                updates.apply_update(result,self.home,target,self.closed,show_progress)
            except Exception as exc:
                detail = updates.error_detail(exc)
                log.warning('Launcher update failed (%s): %s',type(exc).__name__,detail)
                error = '업데이트하지 못했습니다. '+detail+' 기존 실행 파일은 유지됩니다.'
            if not self.closed.is_set():
                self.downloaded.emit(result,error)
        threading.Thread(target=worker,daemon=True).start()

    def finished_download(self, result, error):
        if self.closed.is_set():
            return
        self.busy = False
        self.button.setEnabled(True)
        if error:
            self.label.setText('업데이트 실패 · 버전 확인으로 재시도')
            QMessageBox.warning(self.parent(),'업데이트',error)
            return
        note = ''
        try:
            updates.save_state(self.home,{'pending_version':result['version']})
        except (OSError,ValueError):
            note = '\n업데이트 기록을 저장하지 못했지만 새 실행 파일은 준비되었습니다.'
            log.warning('Pending version could not be saved',exc_info=True)
        self.label.setText('v'+result['version']+' · 다음 실행에 적용')
        QMessageBox.information(self.parent(),'업데이트 완료',
            '새 실행 파일을 준비했습니다. 프로그램을 종료한 뒤 평소처럼 실행하면 새 버전이 적용됩니다.'+note)
