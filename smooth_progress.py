"""Animate the progress fill while keeping reported counts and text exact."""
from PyQt6.QtCore import QEasingCurve, QVariantAnimation
from PyQt6.QtWidgets import QProgressBar, QStyle, QStyleOptionProgressBar, QStylePainter


class SmoothProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._display_fraction = 0.0
        self._target_fraction = 0.0
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(300)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.valueChanged.connect(self._animate_fill)

    def _animate_fill(self, value):
        self._display_fraction = float(value)
        self.update()

    def set_progress(self, done, total, label, *, reset=False):
        was_busy = self.minimum() == self.maximum() == 0
        previous_total = self.maximum()
        self.setRange(0, total)
        self.setValue(done)
        self.setFormat(label)
        target = max(0.0, min(1.0, done / total)) if total > 0 else 0.0
        if reset or was_busy or total != previous_total or target < self._target_fraction:
            self._animation.stop()
            self._target_fraction = target
            self._animate_fill(target)
        elif target != self._target_fraction:
            self._animation.stop()
            self._target_fraction = target
            self._animation.setStartValue(self._display_fraction)
            self._animation.setEndValue(target)
            self._animation.start()

    def paintEvent(self, event):
        if self.minimum() == self.maximum() == 0:
            super().paintEvent(event)
            return
        option = QStyleOptionProgressBar()
        self.initStyleOption(option)
        # Keep the actual %v/%m/%p text; interpolate only the painted fill.
        option.minimum = 0
        option.maximum = 10000
        option.progress = round(self._display_fraction * option.maximum)
        painter = QStylePainter(self)
        painter.drawControl(QStyle.ControlElement.CE_ProgressBar, option)
