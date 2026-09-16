import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.models' / 'huggingface'))

import asyncio
from collections import OrderedDict
from dataclasses import replace
from concurrent.futures import Future
import ctypes
from ctypes import wintypes
import logging
import hashlib
import queue
import re
import sys
import threading
import time
import unicodedata

from PIL import Image
from PyQt6.QtCore import Qt, QRect, QRectF, QPointF, QTimer, QObject, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QRegion, QImage, QBitmap, QIcon, QAction, QActionGroup, QTextLayout, QTextOption, QTextCharFormat
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QComboBox, QCheckBox, QLineEdit, QFormLayout,
                            QMenuBar, QDialog, QScrollArea, QGridLayout, QMessageBox, QFontComboBox, QSpinBox)
from core import changed, relocate, merge_row, scroll_offset, move_rows
from translation import (create_translation_client, TRANSLATION_MODES, validate_openai_settings,
                         list_openai_models, get_deepl_usage)
from ocr_backends import OcrBackend, MODES, MODE_DESCRIPTIONS, validate_device
from api_settings import (load_api_key, save_api_keys, load_openai_settings, OPENAI_DEFAULTS,
                          load_translation_provider)
from window_capture import CaptureWithoutApp, CaptureProtectionError
from hotkeys import ACTIONS, HotkeyDialog, WindowsHotkeys, load_settings
from overlay_settings import DEFAULT_TEXT_STYLE, load_text_style, save_text_style
from translation_logs import TranslationLogs, DailyRuntimeLogHandler
from app_settings import (SOURCE_LANGUAGES, DEFAULT_SOURCE_LANGUAGE, load_source_language,
                          save_source_language, validate_source_language)

log = logging.getLogger(__name__)


def request_model_list(base_url, api_key):
    future = Future()

    def fetch():
        try:
            future.set_result(list_openai_models(base_url, api_key))
        except Exception as exc:
            future.set_exception(exc)

    threading.Thread(target=fetch, daemon=True, name='model-list').start()
    return future


def request_deepl_usage(api_key):
    future = Future()

    def fetch():
        try:
            future.set_result(get_deepl_usage(api_key))
        except Exception as exc:
            future.set_exception(exc)

    threading.Thread(target=fetch, daemon=True, name='deepl-usage').start()
    return future


def windows_api():
    api = ctypes.WinDLL('user32', use_last_error=True)
    api.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    api.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    return api


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
    failed = pyqtSignal(object, int, str)
    finished = pyqtSignal(int)


class Engine(threading.Thread):

    def __init__(self, signals, device='cpu', ocr_mode='manga_ocr', api_key='', provider='luna',
                 base_url='', model='', io_logger=None, source_language='ja'):
        super().__init__(daemon=True)
        self.signals = signals
        self.device = device
        self.ocr_mode = ocr_mode
        self.source_language = validate_source_language(source_language)
        self.api_key = api_key
        self.provider = provider
        self.base_url, self.model = base_url, model
        self.io_logger = io_logger
        self.jobs = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.generation = 0
        self.cache = OrderedDict()
        self.ocr_cache = OrderedDict()
        self.latest_job = None
        self.cancel_before = 0
        self.detector_size = None

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
            self.signals.status.emit(f'글자 인식 중… {i+1}/{len(boxes)}')
            x1, y1, x2, y2 = active_box.crop()
            crop = Image.fromarray(active_pixels[y1:y2, x1:x2])
            key = (crop.size, hashlib.sha256(crop.tobytes()).digest())
            if key in self.ocr_cache:
                source = self.ocr_cache[key]
                self.ocr_cache.move_to_end(key)
            else:
                ocr_started = time.monotonic()
                source = mocr(crop).strip()
                log.info('OCR completed: seconds=%.2f chars=%s', time.monotonic()-ocr_started, len(source))
                self.ocr_cache[key] = source
                if len(self.ocr_cache) > 256:
                    self.ocr_cache.popitem(last=False)
            if self.stop_event.is_set() or generation < self.cancel_before:
                break
            pattern = r'[\u3040-\u30ff\u3400-\u9fff]' if self.source_language == 'ja' else r'[A-Za-z\u3040-\u30ff\u3400-\u9fff]'
            if not re.search(pattern, source):
                continue
            if self.source_language == 'en' or (self.source_language == 'auto'
                    and re.search(r'[A-Za-z]', source)
                    and not re.search(r'[\u3040-\u30ff\u3400-\u9fff]', source)):
                active_box = replace(active_box, vertical=False)
            self.signals.status.emit(f'{TRANSLATION_MODES[self.provider]} 응답 대기 중… {i+1}/{len(boxes)}')
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
        cached = text in self.cache
        request_id = self.io_logger.input(text, self.provider, cached) if self.io_logger else None
        try:
            if cached:
                self.cache.move_to_end(text)
                translated = self.cache[text]
            else:
                result = await client.translate(text, src=self.source_language, dest='ko')
                translated = result.text.strip()
                if not translated:
                    raise RuntimeError('빈 번역 결과')
                self.cache[text] = translated
                if len(self.cache) > 512:
                    self.cache.popitem(last=False)
        except Exception as exc:
            if self.io_logger:
                self.io_logger.output(request_id, None, self.provider, cached, type(exc).__name__)
            raise
        if self.io_logger:
            self.io_logger.output(request_id, translated, self.provider, cached)
        return translated

    def run(self):
        try:
            if self.provider != 'openai' and not self.api_key.strip():
                self.signals.failed.emit(self, self.generation, '선택한 번역 서비스의 API 키를 입력하세요. 입력 후 자동으로 적용됩니다.')
                return
            with asyncio.Runner() as runner:
                async def work():
                    backend = None
                    async with create_translation_client(self.provider, self.api_key,
                                                         base_url=self.base_url, model=self.model) as client:
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
                                if backend is None:
                                    import torch
                                    validate_device(self.device)
                                    self.signals.status.emit(f'{MODES[self.ocr_mode]} 첫 로딩 중…')
                                    backend = OcrBackend(self.ocr_mode, self.device, ROOT, self.source_language)
                                    backend.status = self.signals.status.emit
                                if not self.valid(generation):
                                    continue
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
                                    self.signals.failed.emit(self, generation, f'처리 실패: {exc} — 다시 번역을 눌러 재시도하세요.')
                runner.run(work())
        except Exception as exc:
            log.exception('Engine initialization failed')
            self.signals.failed.emit(self, self.generation, f'모델 초기화 실패: {exc}\n의존성과 인터넷 연결을 확인한 뒤 재실행하세요.')


def vertical_text_layout(text, rect, family='Malgun Gothic', max_size=23, bold=False):
    characters = list(unicodedata.normalize('NFC', ' '.join(text.split())))
    font = QFont(family)
    font.setBold(bold)
    if not characters or rect.width() <= 0 or rect.height() <= 0:
        return font, []
    for size in range(max_size, 0, -1):
        font.setPixelSize(size)
        metrics = QFontMetricsF(font)
        cell_width = max(metrics.height(), *(max(metrics.horizontalAdvance(char),
                                                 metrics.boundingRect(char).width())
                                             for char in characters))
        cell_height = metrics.height()
        rows = int(rect.height() // cell_height)
        columns = int(rect.width() // cell_width)
        if rows and columns and rows * columns >= len(characters):
            break
    else:
        return font, []
    used_columns = (len(characters) + rows - 1) // rows
    right = rect.center().x() + used_columns * cell_width / 2
    top = rect.center().y() - min(rows, len(characters)) * cell_height / 2
    cells = []
    for index, char in enumerate(characters):
        column, row = divmod(index, rows)
        cell = QRectF(right - (column + 1) * cell_width, top + row * cell_height,
                      cell_width, cell_height)
        cells.append((char, cell))
    return font, cells


def horizontal_text_layout(text, rect, family='Malgun Gothic', max_size=23, bold=False, outlined=False):
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\u2028')
    font = QFont(family)
    font.setBold(bold)
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    option.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    for size in range(max_size, 0, -1):
        font.setPixelSize(size)
        layout = QTextLayout(text, font)
        layout.setTextOption(option)
        if outlined:
            char_format = QTextCharFormat()
            char_format.setForeground(QColor('white'))
            char_format.setTextOutline(QPen(QColor('#151515'), min(2.5, size*.12),
                                           Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            span = QTextLayout.FormatRange()
            span.start, span.length, span.format = 0, len(text.encode('utf-16-le'))//2, char_format
            layout.setFormats([span])
        layout.beginLayout()
        height = 0.0
        fits = True
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max(0.0, rect.width()))
            line.setPosition(QPointF(0, height))
            height += line.height()
            fits = fits and line.naturalTextWidth() <= rect.width() + 1e-6
        layout.endLayout()
        if fits and height <= rect.height():
            return layout, QPointF(rect.x(), rect.y() + (rect.height()-height)/2)
    return layout, QPointF(rect.x(), rect.y())


class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                            Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput |
                            Qt.WindowType.WindowDoesNotAcceptFocus)


        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.text_style = dict(DEFAULT_TEXT_STYLE)
        self.rows = []
        self.source_size = (1, 1)
        self.rendered = QImage()
        self.setMask(QRegion(-2, -2, 1, 1))

    def clear(self):
        self.rows = []
        self.rendered = QImage()
        self.setMask(QRegion(-2, -2, 1, 1))
        self.update()

    def display(self, rows, source_size):
        self.rows = rows
        self.source_size = source_size
        layer = QImage(self.size(), QImage.Format.Format_RGBA8888)
        layer.fill(0)
        painter = QPainter(layer)
        background_region = QRegion()
        sx, sy = self.width()/source_size[0], self.height()/source_size[1]
        for box, text in rows:
            if text.strip():
                background_region |= QRegion(QRectF(box.x*sx, box.y*sy, box.w*sx, box.h*sy).toAlignedRect())
        opacity = self.text_style['background_opacity']
        if opacity:
            painter.save()
            painter.setClipRegion(background_region)
            painter.fillRect(layer.rect(), QColor(255,255,255,round(255*opacity/100)))
            painter.restore()
        self.paint_text(painter)
        painter.end()


        self.rendered = layer
        mask = background_region if opacity else QRegion(QBitmap.fromImage(self.rendered.createAlphaMask()))
        self.setMask(mask if not mask.isEmpty() else QRegion(-2, -2, 1, 1))
        self.show()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.rendered)

    def paint_text(self, painter):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        sx, sy = self.width()/self.source_size[0], self.height()/self.source_size[1]
        for box, text in self.rows:
            rect = QRectF(box.x*sx, box.y*sy, box.w*sx, box.h*sy)
            padding = min(4, rect.width() / 4, rect.height() / 4)
            inner = rect.adjusted(padding, padding, -padding, -padding)
            if inner.width() <= 0 or inner.height() <= 0:
                continue

            vertical = box.vertical if box.vertical is not None else box.h > box.w * 1.25
            family, max_size = self.text_style['font_family'], self.text_style['font_size']
            if vertical:
                font, cells = vertical_text_layout(text, inner, family, max_size, True)
                painter.save()
                painter.setClipRect(rect)
                painter.setFont(font)
                painter.setPen(QPen(QColor('#151515'), min(2.5, font.pixelSize()*.12),
                                    Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                painter.setBrush(QColor('white'))
                for char, cell in cells:
                    path = QPainterPath()
                    path.addText(QPointF(0,0), font, char)
                    bounds = path.boundingRect()
                    path.translate(cell.center().x()-bounds.center().x(), cell.center().y()-bounds.center().y())
                    painter.drawPath(path)
                    painter.save()
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawPath(path)
                    painter.restore()
                painter.restore()
                continue

            layout, position = horizontal_text_layout(text, inner, family, max_size, True, outlined=True)
            painter.save()
            painter.setClipRect(rect)
            layout.draw(painter, position)
            fill = QTextCharFormat()
            fill.setForeground(QColor('white'))
            fill.setTextOutline(QPen(Qt.PenStyle.NoPen))
            span = QTextLayout.FormatRange()
            span.start, span.length, span.format = 0, len(layout.text().encode('utf-16-le'))//2, fill
            layout.draw(painter, position, [span])
            painter.restore()


class RegionIndicator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                            Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput |
                            Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.hide_timer.timeout.connect(self.hide)

    def show_region(self, area, duration_ms=0):
        self.hide_timer.stop()
        self.setGeometry(area.adjusted(-3, -3, 3, 3))
        self.setMask(QRegion(self.rect()) - QRegion(self.rect().adjusted(3, 3, -3, -3)))
        self.show()
        self.raise_()
        if duration_ms:
            self.hide_timer.start(duration_ms)

    def hideEvent(self, event):
        self.hide_timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#00d7ff'))
        painter.setPen(QPen(QColor('#151515'), 1))
        painter.drawRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5))


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
        self.finished = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 85))
        painter.setPen(QPen(QColor('#55ddff'), 2))
        painter.drawRect(self.area)
        painter.setFont(QFont('Malgun Gothic', 14))
        painter.drawText(30, 45, '번역할 영역을 드래그하세요 · Esc / 마우스 오른쪽 클릭: 취소')

    def mousePressEvent(self, event):
        if self.finished:
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.cancel()
        elif event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if not self.finished and self.start_point is not None:
            self.area = QRect(self.start_point, event.position().toPoint()).normalized().intersected(self.rect())
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.finished and event.button() == Qt.MouseButton.LeftButton and self.start_point is not None:
            self.area = QRect(self.start_point, event.position().toPoint()).normalized().intersected(self.rect())
            if self.area.width() >= 1 and self.area.height() >= 1:
                area = QRect(self.mapToGlobal(self.area.topLeft()), self.area.size())
                self.finished = True
                self.close()
                self.selected.emit(area)
            else:
                self.cancel()

    def cancel(self):
        if not self.finished:
            self.finished = True
            self.start_point = None
            self.close()
            self.cancelled.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancel()


class ModelComboBox(QComboBox):
    requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.open_requested = False

    def showPopup(self):
        self.open_requested = True
        self.requested.emit()

    def show_loaded_models(self):
        should_open = self.open_requested and self.isVisible() and self.hasFocus()
        self.open_requested = False
        if should_open and self.count():
            super().showPopup()

    def hidePopup(self):
        self.open_requested = False
        super().hidePopup()

    def focusOutEvent(self, event):
        self.open_requested = False
        super().focusOutEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.showPopup()
            event.accept()
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.open_requested = False
        elif event.key() in (Qt.Key.Key_Space, Qt.Key.Key_F4) or (
                event.key() == Qt.Key.Key_Down and event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.showPopup()
            event.accept()
            return
        super().keyPressEvent(event)


class Controller(QWidget):
    def __init__(self, io_logger=None):
        super().__init__()
        self.io_logger = io_logger
        self.setWindowTitle('Manga Live · 화면 → 한국어')
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setAutoFillBackground(True)
        self.resize(510, 230)
        self.overlay = Overlay()
        style_error = ''
        try:
            self.overlay.text_style = load_text_style()
        except (OSError, ValueError):
            style_error = '글자 표시 설정을 읽지 못해 기본값을 사용합니다. settings.json을 확인하세요.'
        self.region_indicator = RegionIndicator()
        self.capture = CaptureWithoutApp()
        self.region = None
        self.reference = None
        self.latest = None
        self.last_change = 0
        self.submitted = False
        self.running = False
        self.single_shot = False
        self.worker_ready = False
        self.result_floor = 0
        self.signals = Signals()
        self.api_settings_error = False
        initial_api_key = self.load_initial_api_key('luna')
        initial_deepl_api_key = self.load_initial_api_key('deepl')
        initial_openai_api_key = self.load_initial_api_key('openai')
        try:
            initial_openai_settings = load_openai_settings()
        except (OSError, ValueError):
            self.api_settings_error = True
            initial_openai_settings = dict(OPENAI_DEFAULTS)
        try:
            initial_provider = load_translation_provider()
        except (OSError, ValueError):
            self.api_settings_error = True
            initial_provider = 'luna'
        initial_keys = {'luna': initial_api_key, 'deepl': initial_deepl_api_key,
                        'openai': initial_openai_api_key}
        language_error = ''
        try:
            initial_language = load_source_language()
        except (OSError, ValueError):
            initial_language = DEFAULT_SOURCE_LANGUAGE
            language_error = '원문 언어 설정을 읽지 못해 일본어를 사용합니다. settings.json을 확인하세요.'
        self.engine = Engine(self.signals, device='cuda', api_key=initial_keys[initial_provider],
                             provider=initial_provider, base_url=initial_openai_settings['base_url'],
                             model=initial_openai_settings['model'], io_logger=self.io_logger,
                             source_language=initial_language)
        self.main_layout = layout = QVBoxLayout(self)
        self.menu_bar = QMenuBar(self)
        self.menu_bar.setNativeMenuBar(False)
        layout.setMenuBar(self.menu_bar)
        self.open_settings_action = QAction('설정창 열기', self)
        self.addAction(self.open_settings_action)
        self.open_settings_action.setShortcut('Ctrl+,')
        self.open_settings_action.triggered.connect(self.open_settings)
        self.interface_modes = QActionGroup(self)
        self.interface_modes.setExclusive(True)
        self.basic_mode_action = QAction('기본 모드', self, checkable=True)
        self.advanced_mode_action = QAction('고급 모드', self, checkable=True)
        for action, mode in [(self.basic_mode_action, 'basic'), (self.advanced_mode_action, 'advanced')]:
            self.interface_modes.addAction(action)
            self.menu_bar.addAction(action)
            action.triggered.connect(lambda checked, selected=mode: self.set_interface_mode(selected))
        self.always_on_top = QCheckBox('최상단 고정')
        self.always_on_top.setToolTip('프로그램 창을 다른 앱보다 위에 표시합니다.')
        self.always_on_top.setChecked(True)
        self.always_on_top.toggled.connect(self.set_always_on_top)
        self.menu_bar.setCornerWidget(self.always_on_top, Qt.Corner.TopRightCorner)
        self.inline_settings = QScrollArea()
        self.inline_settings.setWidgetResizable(True)
        layout.addWidget(self.inline_settings, 1)
        self.settings_panel = QWidget()
        layout = QVBoxLayout(self.settings_panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(QLabel('모니터'))
        self.screens = QComboBox()
        for screen in QApplication.screens():
            self.screens.addItem(screen.name(), screen)
        layout.addWidget(self.screens)
        layout.addWidget(QLabel('OCR 처리 장치'))
        device_row = QHBoxLayout()
        self.device_mode = QComboBox()
        self.device_mode.addItem('CPU 모드', 'cpu')
        self.device_mode.addItem('GPU 모드 (NVIDIA CUDA)', 'cuda')
        self.device_mode.setCurrentIndex(self.device_mode.findData(self.engine.device))
        device_row.addWidget(self.device_mode)
        layout.addLayout(device_row)
        layout.addWidget(QLabel('OCR 방식'))
        self.ocr_mode = QComboBox()
        for key, label in MODES.items():
            self.ocr_mode.addItem(label, key)
            self.ocr_mode.setItemData(self.ocr_mode.count()-1, MODE_DESCRIPTIONS[key], Qt.ItemDataRole.ToolTipRole)
        layout.addWidget(self.ocr_mode)
        ocr_note = QLabel('OCR 선택은 자동으로 적용됩니다. OpenCV 감지는 CPU에서 실행됩니다.')
        ocr_note.setWordWrap(True)
        layout.addWidget(ocr_note)
        layout.addWidget(QLabel('번역 API'))
        self.translation_mode = QComboBox()
        for key, label in TRANSLATION_MODES.items():
            self.translation_mode.addItem(label, key)
        self.translation_mode.setCurrentIndex(self.translation_mode.findData(initial_provider))
        layout.addWidget(self.translation_mode)
        self.api_key = QLineEdit(initial_api_key)
        self.api_key.setPlaceholderText('Kie API 키 (api-keys.json · kie_api_key에 평문 저장)')
        self.api_key_save_timer = QTimer(self)
        self.api_key_save_timer.setSingleShot(True)
        self.api_key_save_timer.setInterval(500)
        self.api_key_save_timer.timeout.connect(self.change_device)
        layout.addWidget(QLabel('원문 언어'))
        self.source_language = QComboBox()
        self.source_language.setToolTip('자동 감지는 일본어·영어 원문을 인식합니다. 자동 감지·영어는 EasyOCR로 읽고 불확실한 영어는 TrOCR로 보완합니다.')
        for code, label in SOURCE_LANGUAGES.items():
            self.source_language.addItem(label, code)
        self.source_language.setCurrentIndex(self.source_language.findData(initial_language))
        layout.addWidget(self.source_language)
        self.language_note = QLabel(language_error)
        self.language_note.setWordWrap(True)
        self.language_note.setVisible(bool(language_error))
        layout.addWidget(self.language_note)
        self.source_language.currentIndexChanged.connect(self.change_source_language)
        self.device_mode.currentIndexChanged.connect(lambda: self.api_key_save_timer.start())
        self.ocr_mode.currentIndexChanged.connect(lambda: self.api_key_save_timer.start())
        self.api_key.textChanged.connect(lambda: self.api_key_save_timer.start())
        layout.addWidget(self.api_key)
        self.api_key.setVisible(initial_provider == 'luna')
        self.deepl_api_key = QLineEdit(initial_deepl_api_key)
        self.deepl_api_key.setPlaceholderText('DeepL API 키 (Free / Pro 자동 선택 · 평문 저장)')
        self.deepl_api_key.textChanged.connect(lambda: self.api_key_save_timer.start())
        layout.addWidget(self.deepl_api_key)
        self.deepl_api_key.setVisible(initial_provider == 'deepl')
        self.deepl_usage_panel = QWidget()
        usage_layout = QVBoxLayout(self.deepl_usage_panel)
        usage_layout.setContentsMargins(0, 0, 0, 0)
        self.deepl_usage_note = QLabel('DeepL API 키를 입력하세요.')
        self.deepl_usage_note.setWordWrap(True)
        usage_layout.addWidget(self.deepl_usage_note)
        self.main_layout.addWidget(self.deepl_usage_panel)
        self.deepl_usage_panel.setVisible(initial_provider == 'deepl')
        self.deepl_usage_future = None
        self.deepl_usage_generation = 0
        self.deepl_usage_closed = False
        self.deepl_usage_debounce = QTimer(self)
        self.deepl_usage_debounce.setSingleShot(True)
        self.deepl_usage_debounce.setInterval(800)
        self.deepl_usage_debounce.timeout.connect(self.load_deepl_usage)
        self.deepl_usage_poll = QTimer(self)
        self.deepl_usage_poll.setInterval(100)
        self.deepl_usage_poll.timeout.connect(self.finish_deepl_usage)
        self.deepl_usage_refresh = QTimer(self)
        self.deepl_usage_refresh.setInterval(300_000)
        self.deepl_usage_refresh.timeout.connect(self.load_deepl_usage)
        self.deepl_api_key.textChanged.connect(self.invalidate_deepl_usage)
        self.invalidate_deepl_usage()
        self.openai_panel = QWidget()
        openai_form = QFormLayout(self.openai_panel)
        openai_form.setContentsMargins(0, 0, 0, 0)
        self.openai_base_url = QLineEdit(initial_openai_settings['base_url'])
        self.openai_base_url.setPlaceholderText('https://서버주소/v1 또는 전체 /chat/completions 주소')
        self.openai_api_key = QLineEdit(initial_openai_api_key)
        self.openai_api_key.setPlaceholderText('인증이 없는 로컬 서버는 비워도 됩니다')
        for label, field in [('API 주소', self.openai_base_url), ('API 키', self.openai_api_key)]:
            openai_form.addRow(label, field)
            field.textChanged.connect(lambda: self.api_key_save_timer.start())
        self.openai_model = ModelComboBox()
        self.openai_model.setPlaceholderText('눌러서 모델 선택')
        self.openai_model.setMinimumContentsLength(12)
        self.openai_model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        if initial_openai_settings['model']:
            self.openai_model.addItem(initial_openai_settings['model'], initial_openai_settings['model'])
            self.openai_model.setCurrentIndex(0)
        self.openai_model.currentIndexChanged.connect(lambda: self.api_key_save_timer.start())
        self.openai_model.requested.connect(self.load_models)
        openai_form.addRow('모델 선택', self.openai_model)
        self.models_note = QLabel('저장된 모델을 복원했습니다. 목록을 새로 불러올 수 있습니다.'
                                 if initial_openai_settings['model'] else '주소와 키를 입력한 뒤 모델 선택을 누르세요.')
        self.models_note.setWordWrap(True)
        openai_form.addRow(self.models_note)
        self.models_future = None
        self.models_generation = 0
        self.models_timer = QTimer(self)
        self.models_timer.setInterval(100)
        self.models_timer.timeout.connect(self.finish_model_list)
        self.openai_base_url.textChanged.connect(self.invalidate_model_list)
        self.openai_api_key.textChanged.connect(self.invalidate_model_list)
        layout.addWidget(self.openai_panel)
        self.openai_panel.setVisible(initial_provider == 'openai')
        self.api_settings_note = QLabel('API 키 파일을 읽지 못했습니다. 키를 다시 입력하면 자동 저장·적용됩니다. '
                                       '손상된 원본은 저장 시 api-key-backups 폴더에 보관합니다.'
                                       if self.api_settings_error else '')
        self.api_settings_note.setWordWrap(True)
        layout.addWidget(self.api_settings_note)
        self.translation_mode.currentIndexChanged.connect(self.update_translation_fields)
        self.action_panel = QWidget()
        row = QGridLayout(self.action_panel)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self.select_button = QPushButton('영역 선택')
        self.select_button.clicked.connect(self.select_region)
        row.addWidget(self.select_button, 0, 0)
        self.drag_button = QPushButton('드래그 번역')
        self.drag_button.clicked.connect(lambda: self.select_region(single_shot=True))
        row.addWidget(self.drag_button, 0, 1)
        self.toggle_button = QPushButton('번역 시작')
        self.toggle_button.clicked.connect(self.toggle)
        row.addWidget(self.toggle_button, 1, 0)
        self.retry_button = QPushButton('다시 번역')
        self.retry_button.clicked.connect(self.retry_translation)
        row.addWidget(self.retry_button, 1, 1)
        for button in (self.select_button, self.drag_button, self.toggle_button, self.retry_button):
            button.setMinimumHeight(60)
            font = button.font()
            font.setPointSize(14)
            font.setBold(True)
            button.setFont(font)
        self.main_layout.addWidget(self.action_panel)
        self.hotkey_button = QPushButton('단축키 설정')
        self.hotkey_button.clicked.connect(self.configure_hotkeys)
        layout.addWidget(self.hotkey_button)
        self.manual = QCheckBox('선택 영역 전체가 말풍선 하나 (자동 감지 생략)')
        self.manual.toggled.connect(self.reset_frame)
        layout.addWidget(self.manual)
        layout.addWidget(QLabel('감지 해상도'))
        self.detection_mode = QComboBox()
        self.detection_mode.addItem('원본 해상도 (기본 · 축소 없이 감지)', None)
        self.detection_mode.addItem('빠른 감지 (작은 글자는 놓칠 수 있음)', 960)
        self.detection_mode.addItem('균형 감지', 1280)
        self.detection_mode.addItem('정밀 감지 (작은 글씨 · 느림)', 1920)
        self.detection_mode.currentIndexChanged.connect(self.change_detection_mode)
        layout.addWidget(self.detection_mode)
        self.text_style_panel = QWidget()
        style_form = QFormLayout(self.text_style_panel)
        style_form.setContentsMargins(0,0,0,0)
        self.translation_font = QFontComboBox()
        self.translation_font.setCurrentFont(QFont(self.overlay.text_style['font_family']))
        style_form.addRow('번역 글꼴', self.translation_font)
        self.translation_font_size = QSpinBox()
        self.translation_font_size.setRange(8,72)
        self.translation_font_size.setSuffix(' px')
        self.translation_font_size.setValue(self.overlay.text_style['font_size'])
        self.translation_font_size.setToolTip('설정한 크기를 기준으로 표시하며, 영역에 들어가지 않으면 자동으로 줄입니다.')
        style_form.addRow('글자 크기', self.translation_font_size)
        self.text_background_opacity = QSpinBox()
        self.text_background_opacity.setRange(0,100)
        self.text_background_opacity.setSuffix(' %')
        self.text_background_opacity.setValue(self.overlay.text_style['background_opacity'])
        self.text_background_opacity.setToolTip('흰색 배경: 0%는 없음, 100%는 완전 불투명. 글자는 항상 선명하게 표시합니다.')
        style_form.addRow('배경 불투명도', self.text_background_opacity)
        self.text_style_note = QLabel(style_error)
        self.text_style_note.setWordWrap(True)
        self.text_style_note.setVisible(bool(style_error))
        style_form.addRow(self.text_style_note)
        self.translation_font.currentFontChanged.connect(self.change_text_style)
        self.translation_font_size.valueChanged.connect(self.change_text_style)
        self.text_background_opacity.valueChanged.connect(self.change_text_style)
        layout.addWidget(self.text_style_panel)
        self.status = QLabel('준비 중…')
        self.status.setWordWrap(True)
        self.main_layout.addWidget(self.status)
        self.capture_note = QLabel('')
        self.capture_note.setWordWrap(True)
        self.main_layout.addWidget(self.capture_note)
        note = QLabel('말풍선별로 번역이 끝나는 즉시 표시합니다. 스크롤하면 번역 위치를 추적합니다.\n인식한 대사를 선택한 번역 서비스로 전송합니다 (이용 요금 발생 가능). 이미지는 전송하지 않습니다. 종료하려면 이 창을 닫으세요.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.connect_engine()
        self.timer = QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self.log_cleanup_timer = QTimer(self)
        self.log_cleanup_timer.setInterval(3_600_000)
        if self.io_logger is not None:
            self.log_cleanup_timer.timeout.connect(self.io_logger.cleanup)
            self.log_cleanup_timer.start()
        self.hotkey_dialog_open = False
        self.hotkeys = WindowsHotkeys(self.activate_hotkey)
        try:
            self.hotkey_settings = load_settings()
            hotkey_errors = []
        except (OSError, ValueError) as exc:
            self.hotkey_settings = {action: '' for action in ACTIONS}
            hotkey_errors = [f'저장된 단축키를 읽지 못했습니다. 단축키 설정에서 다시 저장하세요: {exc}']
        hotkey_errors.extend(self.hotkeys.apply(self.hotkey_settings))
        self.hotkey_note = QLabel()
        self.hotkey_note.setWordWrap(True)
        layout.addWidget(self.hotkey_note)
        self.update_hotkey_note(hotkey_errors)
        layout.addStretch()
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle('Manga Live 설정')
        self.settings_dialog.finished.connect(self.restore_inline_settings)
        dialog_layout = QVBoxLayout(self.settings_dialog)
        self.dialog_settings = QScrollArea()
        self.dialog_settings.setWidgetResizable(True)
        dialog_layout.addWidget(self.dialog_settings)
        close_settings = QPushButton('닫기')
        close_settings.clicked.connect(self.settings_dialog.close)
        dialog_layout.addWidget(close_settings)
        self.interface_mode = None
        self.set_interface_mode('basic')
        self.engine.start()

    def set_interface_mode(self, mode):
        if mode == self.interface_mode:
            return
        self.settings_dialog.hide()
        for scroll in (self.inline_settings, self.dialog_settings):
            if scroll.widget() is self.settings_panel:
                scroll.takeWidget()
        advanced = mode == 'advanced'
        settings_layout = self.settings_panel.layout()
        self.main_layout.removeWidget(self.deepl_usage_panel)
        settings_layout.removeWidget(self.deepl_usage_panel)
        if advanced:
            settings_layout.insertWidget(settings_layout.indexOf(self.deepl_api_key) + 1,
                                         self.deepl_usage_panel)
        else:
            self.main_layout.insertWidget(self.main_layout.indexOf(self.action_panel),
                                          self.deepl_usage_panel)
        self.deepl_usage_panel.setVisible(self.translation_mode.currentData() == 'deepl')
        target = self.inline_settings if advanced else self.dialog_settings
        target.setWidget(self.settings_panel)
        self.settings_panel.show()
        self.inline_settings.setVisible(advanced)
        self.capture_note.setVisible(advanced)
        self.text_style_panel.setVisible(advanced)
        self.interface_mode = mode
        self.basic_mode_action.setChecked(not advanced)
        self.advanced_mode_action.setChecked(advanced)
        self.main_layout.activate()
        available = self.screen().availableGeometry()
        self.resize(560 if advanced else 510, min(850, available.height() - 80) if advanced else self.minimumSizeHint().height())

    def change_text_style(self, *_):
        style = {'font_family': self.translation_font.currentFont().family(),
                 'font_size': self.translation_font_size.value(),
                 'background_opacity': self.text_background_opacity.value()}
        self.overlay.text_style = style
        if self.overlay.rows:
            self.overlay.display(self.overlay.rows, self.overlay.source_size)
        try:
            save_text_style(style)
        except (OSError, ValueError):
            self.text_style_note.setText('글자 표시 설정을 저장하지 못했습니다. settings.json의 상태와 권한을 확인하세요.')
            self.text_style_note.show()
        else:
            self.text_style_note.clear()
            self.text_style_note.hide()

    def open_settings(self):
        if self.inline_settings.widget() is self.settings_panel:
            self.inline_settings.takeWidget()
            self.dialog_settings.setWidget(self.settings_panel)
            self.inline_settings.hide()
            self.settings_panel.show()
            self.main_layout.activate()
            self.resize(self.width(), self.minimumSizeHint().height())
        available = self.screen().availableGeometry()
        self.settings_dialog.resize(560, min(760, available.height() - 80))
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def restore_inline_settings(self):
        if self.interface_mode == 'advanced' and self.dialog_settings.widget() is self.settings_panel:
            self.dialog_settings.takeWidget()
            self.inline_settings.setWidget(self.settings_panel)
            self.inline_settings.show()
            self.settings_panel.show()
            self.resize(560, min(850, self.screen().availableGeometry().height() - 80))

    def update_hotkey_note(self, errors=()):
        buttons = {'select': self.select_button, 'drag': self.drag_button,
                   'toggle': self.toggle_button, 'retry': self.retry_button}
        active = set(self.hotkeys.active.values())
        labels = []
        for action, button in buttons.items():
            key = self.hotkey_settings[action] if action in active else '미지정/비활성'
            button.setToolTip(f'{ACTIONS[action]}: {key}')
            labels.append(f'{ACTIONS[action]}: {key}')
        self.hotkey_note.setText('\n'.join(errors) if errors else ' · '.join(labels))

    def activate_hotkey(self, action):
        if self.hotkey_dialog_open or (hasattr(self, 'selector') and self.selector.isVisible()):
            return
        if hasattr(self, 'device_timer') and self.device_timer.isActive():
            self.status.setText('설정 적용이 끝난 뒤 단축키를 사용하세요.')
            return
        actions = {'select': self.select_region,
                   'drag': lambda: self.select_region(single_shot=True),
                   'toggle': self.toggle, 'retry': self.retry_translation}
        actions[action]()

    def configure_hotkeys(self):
        if self.hotkey_dialog_open:
            return
        self.hotkey_dialog_open = True
        self.hotkeys.clear()
        try:
            parent = self.settings_dialog if self.settings_dialog.isVisible() else self
            dialog = HotkeyDialog(parent, self.hotkeys, self.hotkey_settings)
            if dialog.exec():
                self.hotkey_settings = dialog.settings
                errors = []
            else:
                errors = self.hotkeys.apply(self.hotkey_settings)
            self.update_hotkey_note(errors)
        finally:
            self.hotkey_dialog_open = False

    def connect_engine(self):
        self.signals.status.connect(self.status.setText)
        self.signals.failed.connect(self.processing_failed)
        self.signals.ready.connect(self.ready)
        self.signals.result.connect(self.accept_result)
        self.signals.finished.connect(self.frame_finished)

    def set_always_on_top(self, enabled):
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        if visible:
            self.show()

    def update_translation_fields(self):
        provider = self.translation_mode.currentData()
        self.api_key.setVisible(provider == 'luna')
        self.deepl_api_key.setVisible(provider == 'deepl')
        self.deepl_usage_panel.setVisible(provider == 'deepl')
        self.invalidate_deepl_usage()
        self.openai_panel.setVisible(provider == 'openai')
        self.worker_ready = False
        self.engine.stop_event.set()
        self.reset_for_settings()
        self.change_device()

    def reset_for_settings(self):
        resume_snapshot = self.single_shot and self.running
        snapshot = self.latest if resume_snapshot else None
        self.running = False
        self.reset_frame()
        self.running = resume_snapshot
        self.latest = snapshot
        self.reference = snapshot.copy() if snapshot is not None else None
        self.toggle_button.setText('취소' if resume_snapshot else '번역 시작')

    def invalidate_deepl_usage(self):
        self.deepl_usage_generation += 1
        self.deepl_usage_debounce.stop()
        self.deepl_usage_refresh.stop()
        has_key = bool(self.deepl_api_key.text().strip())
        self.deepl_usage_note.setText('남은 한도 확인 대기 중…' if has_key else 'DeepL API 키를 입력하세요.')
        if not self.deepl_usage_closed and has_key and self.translation_mode.currentData() == 'deepl':
            self.deepl_usage_debounce.start()
            self.deepl_usage_refresh.start()

    def load_deepl_usage(self):
        if (self.deepl_usage_closed or self.translation_mode.currentData() != 'deepl'
                or self.deepl_usage_future is not None or not self.deepl_api_key.text().strip()):
            return
        self.deepl_usage_debounce.stop()
        self.deepl_usage_request_generation = self.deepl_usage_generation
        self.deepl_usage_note.setText('남은 한도 조회 중…')
        try:
            self.deepl_usage_future = request_deepl_usage(self.deepl_api_key.text().strip())
        except Exception:
            self.deepl_usage_note.setText('사용량 조회를 시작하지 못했습니다. 다시 조회하세요.')
            return
        self.deepl_usage_poll.start()

    def finish_deepl_usage(self):
        if self.deepl_usage_future is None or not self.deepl_usage_future.done():
            return
        future, self.deepl_usage_future = self.deepl_usage_future, None
        self.deepl_usage_poll.stop()
        if self.deepl_usage_request_generation != self.deepl_usage_generation:
            if not self.deepl_usage_closed and self.translation_mode.currentData() == 'deepl':
                self.deepl_usage_debounce.start()
            return
        try:
            usage = future.result()
        except (ValueError, RuntimeError) as exc:
            self.deepl_usage_note.setText(str(exc))
            return
        except Exception:
            self.deepl_usage_note.setText('DeepL 사용량을 확인하지 못했습니다. 다시 조회하세요.')
            return
        remaining = ('남은 한도: 제한 없음' if usage['remaining'] is None else
                     f"남은 번역 가능 문자: {usage['remaining']:,}자")
        self.deepl_usage_note.setText(remaining)

    def load_initial_api_key(self, provider):
        try:
            return load_api_key(provider)
        except (OSError, ValueError):
            self.api_settings_error = True
            log.warning('Could not load API key settings for %s', provider)
            return ''

    def translation_is_current(self):
        return (self.worker_ready and not self.engine.stop_event.is_set()
                and self.engine.device == self.device_mode.currentData()
                and self.engine.ocr_mode == self.ocr_mode.currentData()
                and self.engine.source_language == self.source_language.currentData()
                and self.engine.provider == self.translation_mode.currentData()
                and self.engine.api_key == self.selected_api_key()
                and (self.engine.provider != 'openai' or
                     (self.engine.base_url, self.engine.model) == self.selected_openai_settings()))

    def selected_openai_settings(self):
        return self.openai_base_url.text().strip(), self.openai_model.currentData() or ''

    def change_source_language(self):
        try:
            save_source_language(self.source_language.currentData())
            self.language_note.clear()
            self.language_note.hide()
        except (OSError, ValueError):
            self.language_note.setText('원문 언어를 저장하지 못했습니다. settings.json의 상태와 권한을 확인하세요.')
            self.language_note.show()
        self.api_key_save_timer.start()

    def invalidate_model_list(self):
        self.models_generation += 1
        self.openai_model.hidePopup()
        self.openai_model.clear()
        self.models_note.setText('주소 또는 키가 변경되었습니다. 모델 선택을 눌러 다시 불러오세요.')

    def load_models(self):
        if self.models_future is not None:
            return
        self.models_note.setText('모델 목록을 불러오는 중…')
        self.models_request_generation = self.models_generation
        try:
            self.models_future = request_model_list(self.openai_base_url.text().strip(),
                                                   self.openai_api_key.text().strip())
        except Exception:
            self.openai_model.open_requested = False
            self.models_note.setText('모델 목록 조회를 시작하지 못했습니다. 다시 시도하세요.')
            return
        self.models_timer.start()

    def finish_model_list(self):
        if self.models_future is None or not self.models_future.done():
            return
        future, self.models_future = self.models_future, None
        self.models_timer.stop()
        if self.models_request_generation != self.models_generation:
            if self.openai_model.open_requested:
                self.load_models()
            return
        try:
            models = future.result()
        except (ValueError, RuntimeError) as exc:
            self.openai_model.open_requested = False
            self.models_note.setText(str(exc))
            return
        except Exception:
            self.openai_model.open_requested = False
            self.models_note.setText('모델 목록을 불러오지 못했습니다. 주소와 서버 상태를 확인하세요.')
            return
        previous = self.openai_model.currentData()
        open_requested = self.openai_model.open_requested
        self.openai_model.blockSignals(True)
        self.openai_model.clear()
        for model in models:
            self.openai_model.addItem(model, model)
        self.openai_model.setCurrentIndex(self.openai_model.findData(previous) if previous else -1)
        self.openai_model.blockSignals(False)
        self.openai_model.open_requested = open_requested
        self.api_key_save_timer.start()
        if not models:
            self.models_note.setText('서버가 제공한 모델이 없습니다. 키와 서버의 모델 설정을 확인하세요.')
        elif previous and previous not in models:
            self.models_note.setText('기존 모델이 목록에 없습니다. 사용할 모델을 다시 선택하세요.')
        else:
            self.models_note.setText(f'모델 {len(models)}개를 불러왔습니다. 사용할 모델을 선택하면 자동으로 적용됩니다.')
        self.openai_model.show_loaded_models()

    def selected_api_key(self):
        field = {'luna': self.api_key, 'deepl': self.deepl_api_key,
                 'openai': self.openai_api_key}[self.translation_mode.currentData()]
        return field.text().strip()

    def save_api_key(self):
        self.api_key_save_timer.stop()
        try:
            save_api_keys(self.api_key.text(), self.deepl_api_key.text(),
                          openai_key=self.openai_api_key.text(), openai_base_url=self.openai_base_url.text(),
                          openai_model=self.openai_model.currentData() or '',
                          translation_provider=self.translation_mode.currentData())
        except (OSError, ValueError):
            self.status.setText('API 키 자동 저장 실패: 설정 파일의 쓰기 권한을 확인하세요.')
            return False
        self.api_settings_error = False
        self.api_settings_note.clear()
        return True

    def change_device(self):
        if hasattr(self, 'device_timer') and self.device_timer.isActive():
            return
        if not self.save_api_key():
            return
        if self.translation_mode.currentData() == 'openai':
            try:
                validate_openai_settings(*self.selected_openai_settings(), self.selected_api_key())
            except ValueError as exc:
                self.status.setText(str(exc))
                return
        elif not self.selected_api_key():
            self.status.setText('선택한 번역 서비스의 API 키가 필요합니다. 키를 입력하면 자동으로 적용됩니다.')
            return
        if (self.translation_is_current() and self.engine.is_alive()
                and self.engine.device == self.device_mode.currentData()
                and self.engine.ocr_mode == self.ocr_mode.currentData()):
            self.status.setText('이미 적용된 설정입니다.')
            return
        self.worker_ready = False
        self.reset_for_settings()
        self.pending_device = self.device_mode.currentData()
        self.pending_ocr_mode = self.ocr_mode.currentData()
        self.pending_source_language = self.source_language.currentData()
        self.pending_api_key = self.selected_api_key()
        self.pending_provider = self.translation_mode.currentData()
        self.pending_base_url, self.pending_model = self.selected_openai_settings()
        self.engine.stop_event.set()
        self.signals.status.disconnect(self.status.setText)
        self.signals.failed.disconnect(self.processing_failed)
        self.signals.ready.disconnect(self.ready)
        self.signals.result.disconnect(self.accept_result)
        self.signals.finished.disconnect(self.frame_finished)
        self.device_mode.setEnabled(False)
        self.ocr_mode.setEnabled(False)
        self.source_language.setEnabled(False)
        self.api_key.setEnabled(False)
        self.deepl_api_key.setEnabled(False)
        self.translation_mode.setEnabled(False)
        self.openai_panel.setEnabled(False)
        self.status.setText('현재 작업 종료 후 설정을 적용합니다…')
        self.device_timer = QTimer(self)
        self.device_timer.setInterval(100)
        self.device_timer.timeout.connect(self.finish_device_change)
        self.device_timer.start()

    def finish_device_change(self):
        if self.engine.is_alive():
            return
        self.device_timer.stop()
        self.signals = Signals()
        self.engine = Engine(self.signals, device=self.pending_device, ocr_mode=self.pending_ocr_mode,
                             api_key=self.pending_api_key, provider=self.pending_provider,
                             base_url=self.pending_base_url, model=self.pending_model,
                             io_logger=self.io_logger, source_language=self.pending_source_language)
        self.engine.generation = self.result_floor
        self.engine.cancel_before = self.result_floor
        self.engine.detector_size = self.detection_mode.currentData()
        self.connect_engine()
        self.device_mode.setEnabled(True)
        self.ocr_mode.setEnabled(True)
        self.source_language.setEnabled(True)
        self.api_key.setEnabled(True)
        self.deepl_api_key.setEnabled(True)
        self.translation_mode.setEnabled(True)
        self.openai_panel.setEnabled(True)
        self.engine.start()

    def ready(self):
        if (self.engine.stop_event.is_set()
                or self.engine.provider != self.translation_mode.currentData()):
            return
        self.worker_ready = True
        if self.single_shot and self.running:
            if self.latest is None:
                self.finish_snapshot_capture(self.engine.generation)
            else:
                self.submit_snapshot()
            return
        label = 'GPU (CUDA)' if self.engine.device == 'cuda' else 'CPU'
        self.status.setText(f'{label} · {MODES[self.engine.ocr_mode]} · {TRANSLATION_MODES[self.engine.provider]} 준비 완료 (OCR은 첫 요청 시 로딩) · 번역 시작을 누르세요.')

    def processing_failed(self, engine, generation, message):
        if engine is not self.engine or not engine.valid(generation):
            return
        if self.single_shot:
            self.running = False
            self.toggle_button.setText('번역 시작')
            self.show()
        self.status.setText(message)

    def change_detection_mode(self, *_):
        self.engine.detector_size = self.detection_mode.currentData()
        self.reset_frame()

    def reset_frame(self, *_):
        self.capture.invalidate()
        if self.single_shot:
            self.running = False
            self.toggle_button.setText('번역 시작')
        self.engine.generation += 1
        self.result_floor = self.engine.generation
        self.engine.cancel_before = self.result_floor
        self.engine.latest_job = None
        self.reference = None
        self.latest = None
        self.submitted = False
        self.overlay.clear()

    def select_region(self, single_shot=False):
        self.region_indicator.hide()
        self.settings_dialog.close()
        self.running = False
        self.toggle_button.setText('번역 시작')
        self.reset_frame()
        self.single_shot = single_shot
        self.region = None
        self.overlay.hide()
        self.selector = Selector(self.screens.currentData())
        self.selector.selected.connect(self.set_region)
        self.selector.cancelled.connect(self.cancel_selection)
        self.selector.show()
        self.selector.activateWindow()

    def cancel_selection(self):
        self.region_indicator.hide()
        self.single_shot = False
        self.status.setText('영역 선택을 취소했습니다.')
        self.show()

    def set_region(self, area):
        self.region_indicator.hide()
        self.region = None
        self.overlay.setGeometry(area)
        self.overlay.show()


        try:
            self.region = native_region(self.overlay)
        except OSError as exc:
            self.running = False
            self.reset_frame()
            self.toggle_button.setText('번역 시작')
            self.overlay.hide()
            self.show()
            log.exception('Selected region lookup failed')
            self.status.setText(f'영역 좌표 조회 실패: {exc} — 영역을 다시 선택하세요.')
            return
        log.info('Selected region=%s', self.region)
        self.region_indicator.show_region(area, duration_ms=3000 if self.single_shot else 0)
        self.update_capture_mode()
        if self.single_shot:
            self.capture_snapshot()
            return
        self.status.setText('영역 선택 완료 · 번역 시작을 누르세요.')
        self.show()

    def retry_translation(self):
        if self.single_shot and self.region is not None:
            self.capture_snapshot()
        else:
            self.reset_frame()

    def capture_snapshot(self):
        self.reset_frame()
        self.running = True
        self.toggle_button.setText('취소')
        self.status.setText('선택 영역 캡처 중…')
        self.overlay.hide()
        generation = self.engine.generation
        QTimer.singleShot(150, lambda: self.finish_snapshot_capture(generation))

    def finish_snapshot_capture(self, generation):
        if (generation != self.engine.generation or not self.single_shot
                or not self.running or self.region is None or self.submitted):
            return
        try:
            pixels = self.capture.grab(self.region)
            if pixels is None:
                QTimer.singleShot(30, lambda: self.finish_snapshot_capture(generation))
                return
            self.latest = pixels
            self.reference = self.latest.copy()
            self.show()
            if self.worker_ready:
                self.submit_snapshot()
            else:
                self.status.setText('영역 캡처 완료 · 번역 설정이 적용되면 자동으로 번역합니다.')
        except Exception as exc:
            self.running = False
            self.toggle_button.setText('번역 시작')
            self.show()
            log.exception('Snapshot capture failed')
            self.status.setText(f'캡처 실패: {exc} — 다시 번역을 눌러 재시도하세요.')
            self.warn_capture_protection(exc)

    def warn_capture_protection(self, error):
        if not isinstance(error, CaptureProtectionError):
            return
        previous = getattr(self, 'capture_warning', None)
        if previous is not None and previous.isVisible():
            return
        if previous is not None:
            previous.deleteLater()
        self.capture_warning = QMessageBox(QMessageBox.Icon.Warning, '화면 캡처 보호 확인',
            str(error) + '\n\nESET을 사용한다면 개요 → 브라우저 화면 보호 → 일시 중지 → 적용 후 '
            '다시 번역하세요. 번역이 끝나면 보호를 다시 켜세요.\n'
            '다른 보안 프로그램을 사용한다면 해당 프로그램의 화면 캡처 보호 설정을 확인하세요.',
            QMessageBox.StandardButton.Ok, self)
        self.capture_warning.open()

    def submit_snapshot(self):
        if not self.translation_is_current():
            self.status.setText('선택한 번역 서비스의 설정을 확인하세요. 변경한 설정은 자동 적용됩니다. 전환 중이면 잠시 기다려주세요.')
            return
        if self.latest is not None and not self.submitted:
            self.submitted = True
            self.status.setText('드래그한 영역 번역 중…')
            self.engine.submit((self.engine.generation, self.latest, self.manual.isChecked()))

    def update_capture_mode(self):
        self.timer.setInterval(150)
        if self.single_shot:
            self.capture_note.setText('Manga Live 아래 창을 읽습니다. 드래그 번역은 화면 변경 후 다시 번역을 누르세요.')
        else:
            self.capture_note.setText('Manga Live 창을 제외하고 아래 화면의 변화를 감지합니다.')

    def toggle(self):
        if self.running:
            self.running = False
            self.reset_frame()
            self.toggle_button.setText('번역 시작')
            self.status.setText('드래그 번역을 취소했습니다.' if self.single_shot else '일시정지')
            return
        if not self.translation_is_current():
            self.status.setText('선택한 번역 서비스의 설정을 확인하세요. 변경한 설정은 자동 적용됩니다. 적용 중이면 잠시 기다려주세요.')
            return
        if self.region is None:
            self.status.setText('영역 선택을 눌러 번역할 화면 영역을 지정하세요.')
            return
        self.single_shot = False
        self.update_capture_mode()
        self.region_indicator.show_region(self.overlay.geometry())
        self.running = not self.running
        self.toggle_button.setText('일시정지' if self.running else '번역 시작')
        self.reset_frame()
        if self.running:
            self.overlay.show()
        self.status.setText('화면 변화 감시 중…' if self.running else '일시정지')

    def tick(self):
        if not self.running or self.region is None or self.single_shot:
            return
        if not self.translation_is_current():
            return
        try:
            pixels = self.capture.grab(self.region)
            if pixels is None:
                return
            previous = self.latest
            self.latest = pixels
            now = time.monotonic()
            if changed(self.reference, pixels):
                self.engine.generation += 1
                self.reference = pixels
                self.last_change = now
                self.submitted = False
                offset = scroll_offset(previous, pixels) if self.overlay.rows else None
                if offset is not None:
                    tracked = move_rows(self.overlay.rows, offset, pixels.shape)
                else:
                    tracked = []
                    for box, text in self.overlay.rows:
                        moved = relocate(box, previous, pixels)
                        if moved is not None:
                            tracked.append((moved, text))
                self.overlay.display(tracked, (pixels.shape[1], pixels.shape[0]))
            self.submit_pending_frame()
        except Exception as exc:
            self.running = False
            self.toggle_button.setText('번역 시작')
            self.reset_frame()
            log.exception('Capture failed')
            self.status.setText(f'캡처 실패: {exc}')
            self.warn_capture_protection(exc)

    def submit_pending_frame(self):
        if not self.translation_is_current():
            return
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
            self.overlay.display(combined, (self.latest.shape[1], self.latest.shape[0]))
            self.status.setText(f'{len(combined)}개 영역 번역 표시 중 · 나머지 처리 중')

    def frame_finished(self, generation):
        if self.running and generation == self.engine.generation:
            if self.single_shot:
                self.running = False
                self.toggle_button.setText('번역 시작')
                if self.overlay.rows:
                    self.status.setText(f'드래그 번역 완료 · {len(self.overlay.rows)}개 영역 표시 · 다시 번역으로 재실행')
                else:
                    self.status.setText('문자를 인식하지 못했습니다. 원문 언어를 확인하고 영역을 좁히거나 말풍선 하나 모드를 사용해 보세요.')
            else:
                self.status.setText(f'{len(self.overlay.rows)}개 영역 표시 · 화면 변화 대기 중')

    def closeEvent(self, event):
        self.settings_dialog.close()
        self.deepl_usage_closed = True
        self.deepl_usage_debounce.stop()
        self.deepl_usage_poll.stop()
        self.deepl_usage_refresh.stop()
        self.deepl_usage_future = None
        self.deepl_usage_generation += 1
        self.models_timer.stop()
        self.models_future = None
        self.models_generation += 1
        self.hotkeys.close()
        if self.api_key_save_timer.isActive():
            self.save_api_key()
        if hasattr(self, 'device_timer'):
            self.device_timer.stop()
        self.timer.stop()
        self.log_cleanup_timer.stop()
        self.engine.stop_event.set()
        self.engine.generation += 1
        self.overlay.close()
        self.region_indicator.close()
        if hasattr(self, 'selector'):
            self.selector.close()
        self.capture.close()
        event.accept()
        QApplication.instance().quit()


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


def main():
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
    app.setWindowIcon(QIcon(str(ROOT / 'assets' / 'manga-live.ico')))
    app.setQuitOnLastWindowClosed(False)
    window = Controller(io_logger=io_logger)
    if logging_warning:
        note = QLabel(logging_warning)
        note.setWordWrap(True)
        window.layout().addWidget(note)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
