"""Lightweight animated petals behind the application's controls."""
import math
import random
import time
from pathlib import Path

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPalette
from PyQt6.QtWidgets import QApplication, QWidget
from app_settings import read_settings, update_settings

WINDOW_THEMES = {'system': '시스템과 동일', 'light': '라이트', 'dark': '다크', 'sakura': '벚꽃'}
DEFAULT_WINDOW_THEME = 'sakura'


def validate_window_theme(value):
    if not isinstance(value,str) or value not in WINDOW_THEMES:
        raise ValueError('올바른 창 배경을 선택하세요.')
    return value


def load_window_theme(path=None):
    return validate_window_theme(read_settings(path).get('window_theme',DEFAULT_WINDOW_THEME))


def save_window_theme(value,path=None):
    update_settings({'window_theme':validate_window_theme(value)},path)


STYLE = '''
QWidget { color: #fff5fa; }
QDialog { background: #211e29; }
QLabel, QCheckBox { background: transparent; }
QMenuBar { background: #25212e; padding: 3px; }
QMenuBar::item { background: transparent; padding: 4px 9px; }
QMenuBar::item:selected { background: #614256; border-radius: 4px; }
QMenu { background: #302735; color: #fff5fa; border: 1px solid #ac7b96; }
QMenu::item { padding: 6px 24px; }
QMenu::item:selected { background: #79536b; color: white; }
QProgressBar { background: #24212d; border: 1px solid #7e6579; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #614256; border-radius: 3px; }
QPushButton {
    background: #342b3c; border: 1px solid #ac7b96;
    border-radius: 8px; padding: 6px 10px;
}
QPushButton:hover { background: #554057; border-color: #ffc7e0; }
QPushButton:pressed { background: #70506a; }
QPushButton:focus { border: 2px solid #ffd5e7; }
QPushButton:disabled { background: #2c2832; border-color: #655565; color: #aa9ba9; }
QLineEdit, QComboBox, QSpinBox, QKeySequenceEdit {
    background: #24212d; border: 1px solid #7e6579;
    border-radius: 5px; padding: 4px;
    selection-background-color: #80506f; selection-color: white;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: #ffc7e0; }
QComboBox QAbstractItemView {
    background: #302735; color: #fff5fa;
    selection-background-color: #79536b; selection-color: white;
}
QWidget:disabled { color: #aa9ba9; }
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget { background: transparent; }
QToolTip { background: #302735; color: #fff5fa; border: 1px solid #ac7b96; }
QCheckBox::indicator {
    width: 14px; height: 14px; border: 1px solid #7e6579;
    border-radius: 3px; background: #24212d;
}
QCheckBox::indicator:checked {
    background: #80506f; border-color: #ac7b96;
    image: url("CHECKMARK_ASSET");
}
'''
STYLE = STYLE.replace('CHECKMARK_ASSET',(Path(__file__).resolve().parent/'assets/checkmark.svg').as_posix())


class SakuraBackdrop(QWidget):
    def __init__(self, window):
        super().__init__(window)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.elapsed = 0.0
        self.enabled = False
        self.last_tick = time.monotonic()
        generator = random.Random(812)
        self.petals = [(generator.random(), generator.random(), generator.uniform(9,18),
                        generator.uniform(4,7), generator.uniform(0,math.tau)) for _ in range(28)]
        self.petal = QPainterPath()
        self.petal.moveTo(0,1)
        self.petal.cubicTo(-1.1,.35,-.95,-.75,-.22,-1)
        self.petal.lineTo(0,-.65)
        self.petal.lineTo(.22,-1)
        self.petal.cubicTo(.95,-.75,1.1,.35,0,1)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.advance)
        window.installEventFilter(self)
        self.setGeometry(window.rect())
        self.lower()

    def sync_timer(self):
        if self.enabled and self.isVisible() and not self.window().isMinimized():
            self.last_tick = time.monotonic()
            self.timer.start()
        else:
            self.timer.stop()

    def showEvent(self, event):
        super().showEvent(event)
        self.sync_timer()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Resize:
            self.setGeometry(watched.rect())
            self.lower()
        elif event.type() == QEvent.Type.WindowStateChange:
            self.sync_timer()
        elif event.type() == QEvent.Type.Close:
            self.timer.stop()
        return super().eventFilter(watched,event)

    def advance(self):
        now = time.monotonic()
        self.elapsed += min(.1,max(0,now-self.last_tick))
        self.last_tick = now
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0,0,self.width(),self.height())
        gradient.setColorAt(0,QColor('#211e2c'))
        gradient.setColorAt(1,QColor('#372333'))
        painter.fillRect(self.rect(),gradient)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        count = max(12,min(28,self.width()*self.height()//14000))
        for index,(x,y,speed,size,phase) in enumerate(self.petals[:count]):
            px = x*self.width()+math.sin(self.elapsed*.45+phase)*14
            py = (y*(self.height()+30)+self.elapsed*speed) % (self.height()+30)-15
            painter.save()
            painter.translate(px,py)
            painter.rotate(math.degrees(phase)+self.elapsed*(12 if index%2 else -9))
            painter.scale(size,size)
            painter.setBrush(QColor(246,173+index%3*12,208,135+index%3*25))
            painter.drawPath(self.petal)
            painter.restore()


def apply_window_theme(window, background, theme):
    validate_window_theme(theme)
    window.setStyleSheet('')
    background.enabled = theme == 'sakura'
    background.setVisible(background.enabled)
    background.sync_timer()
    QApplication.styleHints().setColorScheme({
        'light': Qt.ColorScheme.Light, 'dark': Qt.ColorScheme.Dark,
    }.get(theme, Qt.ColorScheme.Unknown))
    if theme != 'sakura':
        window.setPalette(QPalette())
        return
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: '#161616',
        QPalette.ColorRole.WindowText: '#f5f5f5',
        QPalette.ColorRole.Base: '#242424',
        QPalette.ColorRole.AlternateBase: '#303030',
        QPalette.ColorRole.Text: '#f5f5f5',
        QPalette.ColorRole.Button: '#303030',
        QPalette.ColorRole.ButtonText: '#f5f5f5',
        QPalette.ColorRole.Highlight: '#3f639b',
        QPalette.ColorRole.HighlightedText: '#ffffff',
        QPalette.ColorRole.PlaceholderText: '#b8b2be',
        QPalette.ColorRole.ToolTipBase: '#242424',
        QPalette.ColorRole.ToolTipText: '#f5f5f5',
    }
    for role,color in colors.items():
        palette.setColor(role,QColor(color))
    window.setPalette(palette)
    window.setStyleSheet(STYLE)
    background.lower()
