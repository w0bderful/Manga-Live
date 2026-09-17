"""Screen selection, native coordinates and the selected-region indicator."""
import ctypes
from ctypes import wintypes
from PyQt6.QtCore import Qt, QRect, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QRegion
from PyQt6.QtWidgets import QWidget


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
