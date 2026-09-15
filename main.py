import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.models' / 'huggingface'))

import asyncio
from collections import OrderedDict
import ctypes
from ctypes import wintypes
import logging
import math
import hashlib
import queue
import re
import sys
import threading
import time

import mss
import numpy as np
from PIL import Image, ImageFilter
from PyQt6.QtCore import Qt, QRect, QRectF, QTimer, QObject, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen, QRegion, QImage, QBitmap
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QComboBox, QCheckBox)
from core import Box, changed, text_boxes, relocate, merge_row, scroll_offset, move_rows, restore_occluded
from translation import TranslationClient, validate_device
from ocr_backends import OcrBackend, MODES

log = logging.getLogger(__name__)


def windows_api():
    api = ctypes.WinDLL('user32', use_last_error=True)
    api.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    api.SetWindowDisplayAffinity.restype = wintypes.BOOL
    api.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    api.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    return api


def exclude_capture(widget):
    if sys.getwindowsversion().build < 19041:
        raise RuntimeError('캡처 제외 기능에는 Windows 10 2004 이상이 필요합니다.')
    if not windows_api().SetWindowDisplayAffinity(int(widget.winId()), 0x11):
        raise ctypes.WinError(ctypes.get_last_error())


def allow_capture(widget):
    if not windows_api().SetWindowDisplayAffinity(int(widget.winId()), 0):
        raise ctypes.WinError(ctypes.get_last_error())


def native_region(widget):
    api = windows_api()
    rect, origin = wintypes.RECT(), wintypes.POINT(0, 0)
    hwnd = int(widget.winId())
    if not api.GetClientRect(hwnd, ctypes.byref(rect)) or not api.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(left=origin.x, top=origin.y, width=rect.right, height=rect.bottom)


class Signals(QObject):
    status = pyqtSignal(str)
    result = pyqtSignal(int, object, object)
    ready = pyqtSignal()
    failed = pyqtSignal(str)
    finished = pyqtSignal(int)


class Engine(threading.Thread):

    def __init__(self, signals, device='cpu', ocr_mode='manga_ocr'):
        super().__init__(daemon=True)
        self.signals = signals
        self.device = device
        self.ocr_mode = ocr_mode
        self.jobs = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.generation = 0
        self.cache = OrderedDict()
        self.ocr_cache = OrderedDict()
        self.latest_job = None
        self.cancel_before = 0
        self.detector_size = 960

    async def process_boxes(self, generation, pixels, boxes, mocr, client):

        for i, box in enumerate(boxes):
            if self.stop_event.is_set() or generation < self.cancel_before:
                break
            active_generation, active_pixels, active_box = generation, pixels, box
            latest = self.latest_job
            if latest is not None and latest[0] > generation:


                active_generation, active_pixels, _ = latest
                active_box = relocate(box, pixels, active_pixels)
                if active_box is None:
                    continue
            self.signals.status.emit(f'OCR·번역 중… {i+1}/{len(boxes)}')
            crop = Image.fromarray(active_pixels).crop(active_box.crop())
            key = (crop.size, hashlib.sha256(crop.tobytes()).digest())
            if key in self.ocr_cache:
                source = self.ocr_cache[key]
                self.ocr_cache.move_to_end(key)
            else:
                source = mocr(crop).strip()
                self.ocr_cache[key] = source
                if len(self.ocr_cache) > 256:
                    self.ocr_cache.popitem(last=False)
            if not re.search(r'[\u3040-\u30ff\u3400-\u9fff]', source):
                continue
            translated = await self.translate(client, source)
            if self.stop_event.is_set() or generation < self.cancel_before:
                break


            self.signals.result.emit(active_generation, [(active_box, translated)], active_pixels)

    def valid(self, generation):
        return not self.stop_event.is_set() and generation == self.generation

    def submit(self, job):
        self.latest_job = job
        try:
            self.jobs.get_nowait()
        except queue.Empty:
            pass
        self.jobs.put_nowait(job)

    async def translate(self, client, text):
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]
        result = await client.translate(text, src='ja', dest='ko')
        translated = result.text.strip()
        if not translated:
            raise RuntimeError('빈 번역 결과')
        self.cache[text] = translated
        if len(self.cache) > 512:
            self.cache.popitem(last=False)
        return translated

    def run(self):
        try:
            self.signals.status.emit('모델 로딩 중… 최초 실행 시 모델을 다운로드합니다.')
            import torch
            validate_device(self.device)
            self.signals.status.emit(f'{MODES[self.ocr_mode]} 로딩 중…')
            backend = OcrBackend(self.ocr_mode, self.device, ROOT)
            self.signals.status.emit('NLLB 로컬 번역 모델 로딩 중… 최초 실행 시 다운로드합니다.')
            with asyncio.Runner() as runner:
                async def work():
                    async with TranslationClient(device=self.device) as client:
                        self.signals.ready.emit()
                        while not self.stop_event.is_set():
                            try:
                                generation, pixels, manual = self.jobs.get(timeout=0.2)
                            except queue.Empty:
                                continue
                            if not self.valid(generation):
                                continue
                            started = time.monotonic()
                            try:
                                self.signals.status.emit('글자 영역 감지 중…')
                                with torch.inference_mode():
                                    detect_started = time.monotonic()
                                    boxes = backend.detect(pixels, self.detector_size, manual)
                                    log.info('Detection: %.3fs mode=%s canvas=%s boxes=%s',
                                             time.monotonic()-detect_started, self.ocr_mode, self.detector_size, len(boxes))
                                    await self.process_boxes(generation, pixels, boxes, backend.recognize, client)
                                if self.valid(generation):
                                    self.signals.finished.emit(generation)
                                    if not boxes:
                                        self.signals.status.emit('글자 영역을 찾지 못했습니다. 말풍선 하나 모드를 사용해 보세요.')
                                    log.info('Frame processed: generation=%s regions=%s seconds=%.2f',
                                             generation, len(boxes), time.monotonic()-started)
                            except Exception as exc:
                                log.exception('Frame processing failed')
                                if self.valid(generation):
                                    self.signals.failed.emit(f'처리 실패: {exc} — 다시 번역을 눌러 재시도하세요.')
                runner.run(work())
        except Exception as exc:
            log.exception('Engine initialization failed')
            self.signals.failed.emit(f'모델 초기화 실패: {exc}\n의존성과 인터넷 연결을 확인한 뒤 재실행하세요.')


class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                            Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput |
                            Qt.WindowType.WindowDoesNotAcceptFocus)


        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.rows = []
        self.source_size = (1, 1)
        self.rendered = QImage()
        self.capture_history = {}
        self.setMask(QRegion(-2, -2, 1, 1))

    def current_capture_boxes(self):

        px = max(0, math.ceil(8*self.source_size[0]/max(1, self.width()))-12)
        py = max(0, math.ceil(8*self.source_size[1]/max(1, self.height()))-12)
        return [Box(b.x-px, b.y-py, b.w+2*px, b.h+2*py) for b, _ in self.rows]

    def remember_capture_boxes(self):
        now = time.monotonic()
        self.capture_history = {b: expiry for b, expiry in self.capture_history.items() if expiry > now}
        for box in self.current_capture_boxes():
            self.capture_history[box] = now+0.45

    def capture_boxes(self):
        now = time.monotonic()
        self.capture_history = {b: expiry for b, expiry in self.capture_history.items() if expiry > now}

        return list(set(self.current_capture_boxes()) | self.capture_history.keys())

    def clear(self):
        self.remember_capture_boxes()
        self.rows = []
        self.rendered = QImage()
        self.setMask(QRegion(-2, -2, 1, 1))
        self.update()

    def display(self, rows, source_size, background=None):
        self.remember_capture_boxes()
        self.rows = rows
        self.source_size = source_size
        layer = QImage(self.size(), QImage.Format.Format_RGBA8888)
        layer.fill(0)
        painter = QPainter(layer)
        self.paint_text(painter)
        painter.end()


        glyphs = Image.frombytes('RGBA', (layer.width(), layer.height()),
                                 layer.bits().asstring(layer.sizeInBytes()))
        alpha = glyphs.getchannel('A')
        halo = alpha.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.8))
        halo = halo.point(lambda value: min(255, int(value * 1.8)))
        if background is not None:
            base = Image.fromarray(background).convert('RGBA').resize(glyphs.size, Image.Resampling.BILINEAR)
        else:
            base = Image.new('RGBA', glyphs.size, 'white')
        glow = Image.new('RGBA', glyphs.size, 'white')
        glow.putalpha(halo)
        composed = Image.alpha_composite(Image.alpha_composite(base, glow), glyphs)


        coverage = halo.point(lambda value: 255 if value >= 8 else 0)
        composed.putalpha(coverage)
        self.rendered = QImage(composed.tobytes(), composed.width, composed.height,
                               QImage.Format.Format_RGBA8888).copy()
        mask = QRegion(QBitmap.fromImage(self.rendered.createAlphaMask()))
        self.setMask(mask if not mask.isEmpty() else QRegion(-2, -2, 1, 1))
        self.show()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.rendered)

    def paint_text(self, painter):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        sx, sy = self.width()/self.source_size[0], self.height()/self.source_size[1]
        flags = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextWrapAnywhere
        for box, text in self.rows:
            rect = QRectF(box.x*sx, box.y*sy, box.w*sx, box.h*sy)
            inner = rect.adjusted(2, 2, -2, -2)
            if inner.width() <= 0 or inner.height() <= 0:
                continue

            font = QFont('Malgun Gothic')
            for size in range(23, 0, -1):
                font.setPixelSize(size)
                bounds = QFontMetricsF(font).boundingRect(inner, int(flags), text)
                if bounds.height() <= inner.height() and bounds.width() <= inner.width():
                    break
            painter.setFont(font)
            painter.setPen(QColor('#151515'))
            painter.save()
            painter.setClipRect(inner)
            painter.drawText(inner, int(flags), text)
            painter.restore()


class Selector(QWidget):
    selected = pyqtSignal(QRect)
    cancelled = pyqtSignal()

    def __init__(self, screen):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(screen.geometry())
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.start_point = None
        self.area = QRect()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 85))
        painter.setPen(QPen(QColor('#55ddff'), 2))
        painter.drawRect(self.area)
        painter.setFont(QFont('Malgun Gothic', 14))
        painter.drawText(30, 45, '번역할 영역을 드래그하세요 · Esc: 취소')

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if self.start_point is not None:
            self.area = QRect(self.start_point, event.position().toPoint()).normalized().intersected(self.rect())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.start_point is not None:
            self.area = QRect(self.start_point, event.position().toPoint()).normalized().intersected(self.rect())
            if self.area.width() >= 40 and self.area.height() >= 40:
                area = QRect(self.mapToGlobal(self.area.topLeft()), self.area.size())
                self.close()
                self.selected.emit(area)
            else:
                self.close()
                self.cancelled.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            self.cancelled.emit()


class Controller(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Manga Live · 일본어 → 한국어')
        self.resize(510, 230)
        self.overlay = Overlay()
        self.capture = mss.MSS()
        self.region = None
        self.reference = None
        self.latest = None
        self.last_change = 0
        self.submitted = False
        self.running = False
        self.models_ready = False
        self.result_floor = 0
        self.overlay_excluded = False
        self.control_excluded = False
        self.signals = Signals()
        self.engine = Engine(self.signals)
        layout = QVBoxLayout(self)
        heading = QLabel('화면의 일본어를 원래 위치에 한국어로 표시합니다.')
        layout.addWidget(heading)
        self.screens = QComboBox()
        for screen in QApplication.screens():
            self.screens.addItem(screen.name(), screen)
        layout.addWidget(self.screens)
        device_row = QHBoxLayout()
        self.device_mode = QComboBox()
        self.device_mode.addItem('CPU 모드', 'cpu')
        self.device_mode.addItem('GPU 모드 (NVIDIA CUDA)', 'cuda')
        device_row.addWidget(self.device_mode)
        self.device_apply = QPushButton('CPU/GPU·OCR 모드 적용')
        self.device_apply.clicked.connect(self.change_device)
        device_row.addWidget(self.device_apply)
        layout.addLayout(device_row)
        self.ocr_mode = QComboBox()
        for key, label in MODES.items():
            self.ocr_mode.addItem(label, key)
        layout.addWidget(self.ocr_mode)
        layout.addWidget(QLabel('OCR을 선택한 뒤 모드 적용을 누르세요. OpenCV 감지는 CPU에서 실행됩니다.'))
        row = QHBoxLayout()
        self.select_button = QPushButton('영역 선택')
        self.select_button.clicked.connect(self.select_region)
        row.addWidget(self.select_button)
        self.toggle_button = QPushButton('시작')
        self.toggle_button.clicked.connect(self.toggle)
        row.addWidget(self.toggle_button)
        self.retry_button = QPushButton('다시 번역')
        self.retry_button.clicked.connect(self.reset_frame)
        row.addWidget(self.retry_button)
        layout.addLayout(row)
        self.manual = QCheckBox('선택 영역 전체가 말풍선 하나 (자동 감지 생략)')
        self.manual.toggled.connect(self.reset_frame)
        layout.addWidget(self.manual)
        self.detection_mode = QComboBox()
        self.detection_mode.addItem('빠른 감지 (기본 · 작은 글자는 놓칠 수 있음)', 960)
        self.detection_mode.addItem('균형 감지', 1280)
        self.detection_mode.addItem('정밀 감지 (작은 글씨 · 느림)', 1920)
        self.detection_mode.currentIndexChanged.connect(self.change_detection_mode)
        layout.addWidget(self.detection_mode)
        self.share_overlay = QCheckBox('화면공유에 번역 표시 (Discord에서 전체 화면 공유)')
        self.share_overlay.setChecked(True)
        self.share_overlay.toggled.connect(self.change_sharing_mode)
        layout.addWidget(self.share_overlay)
        self.status = QLabel('준비 중…')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.capture_note = QLabel('')
        self.capture_note.setWordWrap(True)
        layout.addWidget(self.capture_note)
        note = QLabel('말풍선별로 번역이 끝나는 즉시 표시합니다. 스크롤하면 번역 위치를 추적합니다.\n이미지·대사는 서버로 보내지 않습니다. 종료하려면 이 창을 닫으세요.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.connect_engine()
        self.timer = QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self.engine.start()

    def connect_engine(self):
        self.signals.status.connect(self.status.setText)
        self.signals.failed.connect(self.status.setText)
        self.signals.ready.connect(self.ready)
        self.signals.result.connect(self.accept_result)
        self.signals.finished.connect(self.frame_finished)

    def change_device(self):
        self.running = False
        self.models_ready = False
        self.toggle_button.setText('시작')
        self.reset_frame()
        self.pending_device = self.device_mode.currentData()
        self.pending_ocr_mode = self.ocr_mode.currentData()
        self.engine.stop_event.set()
        self.signals.status.disconnect(self.status.setText)
        self.signals.failed.disconnect(self.status.setText)
        self.signals.ready.disconnect(self.ready)
        self.signals.result.disconnect(self.accept_result)
        self.signals.finished.disconnect(self.frame_finished)
        self.device_apply.setEnabled(False)
        self.device_mode.setEnabled(False)
        self.ocr_mode.setEnabled(False)
        self.status.setText('현재 작업 종료 후 선택한 모드로 모델을 다시 로딩합니다…')
        self.device_timer = QTimer(self)
        self.device_timer.setInterval(100)
        self.device_timer.timeout.connect(self.finish_device_change)
        self.device_timer.start()

    def finish_device_change(self):
        if self.engine.is_alive():
            return
        self.device_timer.stop()
        self.signals = Signals()
        self.engine = Engine(self.signals, device=self.pending_device, ocr_mode=self.pending_ocr_mode)
        self.engine.generation = self.result_floor
        self.engine.cancel_before = self.result_floor
        self.engine.detector_size = self.detection_mode.currentData()
        self.connect_engine()
        self.device_apply.setEnabled(True)
        self.device_mode.setEnabled(True)
        self.ocr_mode.setEnabled(True)
        self.engine.start()

    def ready(self):
        self.models_ready = True
        label = 'GPU (CUDA)' if self.engine.device == 'cuda' else 'CPU'
        self.status.setText(f'{label} · {MODES[self.engine.ocr_mode]} 준비 완료 · 시작을 누르세요.')

    def change_detection_mode(self, *_):
        self.engine.detector_size = self.detection_mode.currentData()
        self.reset_frame()

    def change_sharing_mode(self, *_):
        self.reset_frame()
        if self.region is not None:
            self.configure_capture()

    def configure_capture(self):
        try:
            if self.share_overlay.isChecked():
                allow_capture(self.overlay)
                self.overlay_excluded = False
            else:
                exclude_capture(self.overlay)
                self.overlay_excluded = True
        except Exception as exc:
            log.warning('Capture mode configuration failed: %s', exc)
            self.running = False
            self.toggle_button.setText('시작')
            self.status.setText(f'화면공유 설정 실패: {exc}')
            return False
        self.update_capture_mode()
        return True

    def reset_frame(self, *_):
        self.engine.generation += 1
        self.result_floor = self.engine.generation
        self.engine.cancel_before = self.result_floor
        self.engine.latest_job = None
        self.reference = None
        self.latest = None
        self.submitted = False
        self.overlay.clear()

    def select_region(self):
        self.running = False
        self.toggle_button.setText('시작')
        self.reset_frame()
        self.overlay.hide()
        self.hide()
        self.selector = Selector(self.screens.currentData())
        self.selector.selected.connect(self.set_region)
        self.selector.cancelled.connect(self.show)
        self.selector.show()
        self.selector.activateWindow()

    def set_region(self, area):
        self.region = None
        self.overlay.setGeometry(area)
        self.overlay.show()


        self.region = native_region(self.overlay)
        if not self.configure_capture():
            self.region = None
            self.show()
            return
        log.info('Selected region=%s overlay_excluded=%s control_excluded=%s',
                 self.region, self.overlay_excluded, self.control_excluded)
        self.status.setText('영역 선택 완료 · 시작을 누르세요.')
        self.update_capture_mode()
        self.show()

    def update_capture_mode(self):
        self.timer.setInterval(150)
        if not self.overlay_excluded:
            self.capture_note.setText('선택 영역 안에서만 변화와 스크롤을 감지합니다. 가려진 내용만 바뀌면 다시 번역을 누르세요.')
        else:
            self.capture_note.setText('선택 영역 안에서만 화면 변화를 감지합니다.')

    def toggle(self):
        if not self.models_ready or self.region is None:
            self.status.setText('모델 로딩 완료 후 번역 영역을 먼저 선택하세요.')
            return
        self.running = not self.running
        self.toggle_button.setText('일시정지' if self.running else '시작')
        self.reset_frame()
        if self.running:
            self.overlay.show()
        self.status.setText('화면 변화 감시 중…' if self.running else '일시정지')

    def tick(self):
        if not self.running or self.region is None:
            return
        try:
            if (not self.control_excluded and self.isVisible() and
                    self.frameGeometry().intersects(self.overlay.geometry())):
                self.status.setText('제어창을 번역 영역 밖으로 이동하세요.')
                return
            boxes = self.overlay.capture_boxes() if not self.overlay_excluded else []
            if boxes and self.reference is None:


                return
            shot = self.capture.grab(self.region)
            pixels = np.asarray(shot)[:, :, :3][:, :, ::-1].copy()
            if boxes:
                if not changed(self.reference, pixels, ignored=boxes):
                    self.submit_pending_frame()
                    return
                offset = scroll_offset(self.latest, pixels, ignored=boxes)
                if offset is None:
                    self.reset_frame()
                    return
                pixels = restore_occluded(self.latest, pixels, boxes, offset)
                tracked = move_rows(self.overlay.rows, offset, pixels.shape)
                self.overlay.display(tracked, (pixels.shape[1], pixels.shape[0]), pixels)
                self.latest = pixels
                self.reference = pixels.copy()
                self.engine.generation += 1
                self.last_change = time.monotonic()
                self.submitted = False
                return
            previous = self.latest
            self.latest = pixels
            now = time.monotonic()
            if changed(self.reference, pixels):
                self.engine.generation += 1
                self.reference = pixels.copy()
                self.last_change = now
                self.submitted = False
                offset = scroll_offset(previous, pixels)
                if offset is not None:
                    tracked = move_rows(self.overlay.rows, offset, pixels.shape)
                else:
                    tracked = []
                    for box, text in self.overlay.rows:
                        moved = relocate(box, previous, pixels)
                        if moved is not None:
                            tracked.append((moved, text))
                self.overlay.display(tracked, (pixels.shape[1], pixels.shape[0]), pixels)
            self.submit_pending_frame()
        except Exception as exc:
            self.running = False
            self.toggle_button.setText('시작')
            self.reset_frame()
            log.exception('Capture failed')
            self.status.setText(f'캡처 실패: {exc}')

    def submit_pending_frame(self):


        if (self.running and self.latest is not None and not self.submitted and
                time.monotonic()-self.last_change >= 0.25):
            self.submitted = True
            self.engine.submit((self.engine.generation, self.latest, self.manual.isChecked()))

    def accept_result(self, generation, rows, pixels):
        if not self.running or generation < self.result_floor or self.latest is None:
            return
        combined = list(self.overlay.rows)
        for box, text in rows:
            moved = relocate(box, pixels, self.latest)
            if moved is not None:
                combined = merge_row(combined, (moved, text))
        if combined:
            self.overlay.display(combined, (self.latest.shape[1], self.latest.shape[0]), self.latest)
            self.status.setText(f'{len(combined)}개 영역 번역 표시 중 · 나머지 처리 중')

    def frame_finished(self, generation):
        if self.running and generation == self.engine.generation:
            self.status.setText(f'{len(self.overlay.rows)}개 영역 표시 · 화면 변화 대기 중')

    def closeEvent(self, event):
        if hasattr(self, 'device_timer'):
            self.device_timer.stop()
        self.timer.stop()
        self.engine.stop_event.set()
        self.engine.generation += 1
        self.overlay.close()
        if hasattr(self, 'selector'):
            self.selector.close()
        self.capture.close()
        event.accept()
        QApplication.instance().quit()


def main():
    if sys.platform != 'win32':
        raise SystemExit('이 프로그램은 Windows 전용입니다.')
    logging.basicConfig(filename=ROOT/'manga-live.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8')
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = Controller()
    window.show()
    try:
        exclude_capture(window)
        window.control_excluded = True
    except Exception as exc:
        log.warning('Control capture exclusion unavailable: %s', exc)
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
