"""Windows global shortcuts and their local settings dialog."""
import ctypes
from ctypes import wintypes

from PyQt6.QtCore import QAbstractNativeEventFilter, Qt, QTimer
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QFormLayout,
                            QHBoxLayout, QKeySequenceEdit, QLabel, QPushButton,
                            QVBoxLayout, QWidget)
from app_settings import SETTINGS_FILE, read_settings, update_settings

ACTIONS = {'select': '영역 선택', 'drag': '드래그 번역',
           'toggle': '번역 시작 / 일시정지 / 취소', 'retry': '다시 번역'}
DEFAULTS = dict(zip(ACTIONS, ('Ctrl+Alt+1', 'Ctrl+Alt+2', 'Ctrl+Alt+3', 'Ctrl+Alt+4')))
WM_HOTKEY = 0x0312
MOD_NOREPEAT = 0x4000


def key_binding(text):
    sequence = QKeySequence.fromString(text, QKeySequence.SequenceFormat.PortableText)
    if not text:
        return None
    if sequence.count() != 1:
        raise ValueError('단축키는 한 번에 누르는 키 조합으로 지정하세요.')
    combination = sequence[0]
    key = combination.key()
    qt_mods = combination.keyboardModifiers()
    modifiers = 0
    for qt_flag, win_flag in ((Qt.KeyboardModifier.AltModifier, 1),
                              (Qt.KeyboardModifier.ControlModifier, 2),
                              (Qt.KeyboardModifier.ShiftModifier, 4),
                              (Qt.KeyboardModifier.MetaModifier, 8)):
        if qt_mods & qt_flag:
            modifiers |= win_flag
    if qt_mods & Qt.KeyboardModifier.KeypadModifier:
        raise ValueError('숫자 패드 대신 일반 숫자 키를 사용하세요.')
    if key == Qt.Key.Key_F12:
        raise ValueError('F12는 Windows 예약 키입니다. 다른 키를 선택하세요.')
    if Qt.Key.Key_A <= key <= Qt.Key.Key_Z or Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
        virtual_key = int(key)
    elif Qt.Key.Key_F1 <= key <= Qt.Key.Key_F24:
        virtual_key = 0x70 + int(key) - int(Qt.Key.Key_F1)
    else:
        special = {Qt.Key.Key_Space: 0x20, Qt.Key.Key_Tab: 0x09,
                   Qt.Key.Key_Return: 0x0D, Qt.Key.Key_Enter: 0x0D,
                   Qt.Key.Key_Escape: 0x1B, Qt.Key.Key_Backspace: 0x08,
                   Qt.Key.Key_Insert: 0x2D, Qt.Key.Key_Delete: 0x2E,
                   Qt.Key.Key_Home: 0x24, Qt.Key.Key_End: 0x23,
                   Qt.Key.Key_PageUp: 0x21, Qt.Key.Key_PageDown: 0x22,
                   Qt.Key.Key_Left: 0x25, Qt.Key.Key_Up: 0x26,
                   Qt.Key.Key_Right: 0x27, Qt.Key.Key_Down: 0x28}
        virtual_key = special.get(key)
        if virtual_key is None:
            raise ValueError('영문·숫자·F키 또는 방향/이동 키를 사용하세요.')
    if not modifiers and not Qt.Key.Key_F1 <= key <= Qt.Key.Key_F24:
        raise ValueError('일반 키에는 Ctrl, Alt, Shift, Win 중 하나를 함께 지정하세요.')
    return modifiers, virtual_key


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError('단축키 설정은 JSON 객체여야 합니다.')
    normalized, bindings, used = {}, {}, set()
    for action in ACTIONS:
        text = settings.get(action, DEFAULTS[action])
        if not isinstance(text, str):
            raise ValueError(f'{ACTIONS[action]} 단축키는 문자열이어야 합니다.')
        binding = key_binding(text)
        if binding is not None and binding in used:
            raise ValueError('여러 기능에 같은 단축키를 지정할 수 없습니다.')
        used.add(binding)
        bindings[action] = binding
        normalized[action] = QKeySequence(text).toString(QKeySequence.SequenceFormat.PortableText)
    return normalized, bindings


def load_settings(path=SETTINGS_FILE):
    return validate_settings(read_settings(path).get('hotkeys', {}))[0]


def save_settings(settings, path=SETTINGS_FILE):
    normalized, _ = validate_settings(settings)
    update_settings({'hotkeys': normalized}, path)


class WindowsHotkeys(QAbstractNativeEventFilter):
    def __init__(self, callback, user32=None):
        super().__init__()
        self.callback = callback
        self.user32 = user32 if user32 is not None else ctypes.WinDLL('user32', use_last_error=True)
        if user32 is None:
            self.user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
            self.user32.RegisterHotKey.restype = wintypes.BOOL
            self.user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
            self.user32.UnregisterHotKey.restype = wintypes.BOOL
        self.active = {}
        self.epoch = 0
        self.next_id = 0x5100
        self.app = QApplication.instance()
        self.app.installNativeEventFilter(self)

    def clear(self):
        self.epoch += 1
        for identifier in self.active:
            self.user32.UnregisterHotKey(None, identifier)
        self.active.clear()

    def apply(self, settings):
        normalized, bindings = validate_settings(settings)
        self.clear()
        errors = []
        for action, binding in bindings.items():
            if binding is None:
                continue
            identifier = self.next_id
            self.next_id += 1
            if self.next_id > 0xBFFF:
                self.next_id = 0x5100
            modifiers, virtual_key = binding
            if self.user32.RegisterHotKey(None, identifier, modifiers | MOD_NOREPEAT, virtual_key):
                self.active[identifier] = action
            else:
                errors.append(f'{ACTIONS[action]} ({normalized[action]}): 등록할 수 없습니다. 다른 앱 사용 또는 Windows 예약 키인지 확인하세요.')
        return errors

    def nativeEventFilter(self, event_type, message):
        if bytes(event_type) in (b'windows_generic_MSG', b'windows_dispatcher_MSG'):
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY and int(msg.wParam) in self.active:
                action, epoch = self.active[int(msg.wParam)], self.epoch
                QTimer.singleShot(0, lambda: self.callback(action) if epoch == self.epoch else None)
                return True, 0
        return False, 0

    def close(self):
        self.clear()
        self.app.removeNativeEventFilter(self)


class HotkeyDialog(QDialog):
    def __init__(self, parent, manager, settings, path=SETTINGS_FILE):
        super().__init__(parent)
        self.manager, self.settings, self.path = manager, dict(settings), path
        self.setWindowTitle('단축키 설정')
        layout = QVBoxLayout(self)
        note = QLabel('입력칸을 클릭하고 원하는 키 조합을 누르세요.\n'
                      '다른 창에서도 작동합니다. 비우면 해당 기능의 단축키가 해제됩니다.\n'
                      '영문·숫자·방향 키는 Ctrl/Alt/Shift/Win과 함께, F키는 단독 사용도 가능합니다.')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.editors = {}
        for action, label in ACTIONS.items():
            editor = QKeySequenceEdit(QKeySequence(settings[action]))
            editor.setMaximumSequenceLength(1)
            editor.setClearButtonEnabled(True)
            # Tab can be assigned; move focus with the mouse instead.
            editor.setFinishingKeyCombinations([])
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(editor)
            clear = QPushButton('해제')
            clear.clicked.connect(editor.clear)
            row_layout.addWidget(clear)
            form.addRow(label, row)
            self.editors[action] = editor
        layout.addLayout(form)
        defaults = QPushButton('기본값 불러오기')
        defaults.clicked.connect(self.restore_defaults)
        layout.addWidget(defaults)
        self.error = QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText('저장')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('취소')
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def restore_defaults(self):
        for action, editor in self.editors.items():
            editor.setKeySequence(QKeySequence(DEFAULTS[action]))

    def save(self):
        proposed = {action: editor.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
                    for action, editor in self.editors.items()}
        try:
            errors = self.manager.apply(proposed)
            if errors:
                raise ValueError('\n'.join(errors))
            save_settings(proposed, self.path)
        except (ValueError, OSError) as exc:
            self.manager.clear()
            self.error.setText(str(exc))
            return
        self.settings = proposed
        self.accept()
