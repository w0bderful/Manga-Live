"""Asynchronous model-list selection without blocking the Qt popup."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox


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
