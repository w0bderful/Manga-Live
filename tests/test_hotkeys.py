import ctypes
from ctypes import wintypes
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PyQt6.QtCore import QByteArray
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QApplication, QDialog

import hotkeys
import main


class HotkeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.api = Mock()
        self.api.RegisterHotKey.return_value = True
        self.callback = Mock()
        self.manager = hotkeys.WindowsHotkeys(self.callback, self.api)
        self.addCleanup(self.manager.close)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'hotkeys.json'

    def test_native_key_mapping(self):
        self.assertEqual(hotkeys.key_binding('Ctrl+Alt+1'), (3, 0x31))
        self.assertEqual(hotkeys.key_binding('F8'), (0, 0x77))
        self.assertEqual(hotkeys.key_binding('Ctrl+Shift+Left'), (6, 0x25))
        self.assertIsNone(hotkeys.key_binding(''))

    def test_invalid_reserved_and_multistroke_keys(self):
        for key in ('A', 'Esc', 'Ctrl+F12', 'Ctrl+A, Ctrl+B', 'NotAKey', 'Ctrl+Num+1'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                hotkeys.key_binding(key)

    def test_duplicate_rejected_before_existing_registration_removed(self):
        self.manager.apply(hotkeys.DEFAULTS)
        active = dict(self.manager.active)
        with self.assertRaises(ValueError):
            self.manager.apply({**hotkeys.DEFAULTS, 'drag': 'Ctrl+Alt+1'})
        self.assertEqual(self.manager.active, active)

    def test_save_reload_and_disable(self):
        self.assertEqual(hotkeys.load_settings(self.path), hotkeys.DEFAULTS)
        settings = {**hotkeys.DEFAULTS, 'select': '', 'drag': 'F8'}
        hotkeys.save_settings(settings, self.path)
        self.assertEqual(hotkeys.load_settings(self.path), settings)
        self.assertFalse(self.path.with_suffix('.json.tmp').exists())
        self.manager.apply(settings)
        self.assertEqual(len(self.manager.active), 3)
        self.assertNotIn('select', self.manager.active.values())

    def test_corrupt_file_rejected_without_overwrite(self):
        self.path.write_text('{broken', encoding='utf-8')
        with self.assertRaises(ValueError):
            hotkeys.load_settings(self.path)
        self.assertEqual(self.path.read_text(encoding='utf-8'), '{broken')

    def test_no_repeat_and_cleanup(self):
        self.manager.apply(hotkeys.DEFAULTS)
        for call in self.api.RegisterHotKey.call_args_list:
            self.assertTrue(call.args[2] & hotkeys.MOD_NOREPEAT)
        self.manager.clear()
        self.assertEqual(self.api.UnregisterHotKey.call_count, 4)
        self.assertFalse(self.manager.active)

    def test_native_message_dispatched_once_and_stale_ignored(self):
        self.manager.apply(hotkeys.DEFAULTS)
        identifier = next(iter(self.manager.active))
        msg = wintypes.MSG()
        msg.message, msg.wParam = hotkeys.WM_HOTKEY, identifier
        handled = self.manager.nativeEventFilter(QByteArray(b'windows_dispatcher_MSG'), ctypes.addressof(msg))
        self.assertEqual(handled, (True, 0))
        self.app.processEvents()
        self.callback.assert_called_once_with('select')
        self.manager.nativeEventFilter(QByteArray(b'windows_generic_MSG'), ctypes.addressof(msg))
        self.manager.clear()
        self.app.processEvents()
        self.callback.assert_called_once()
        self.assertEqual(self.manager.nativeEventFilter(QByteArray(b'windows_dispatcher_MSG'), ctypes.addressof(msg)), (False, 0))

    def dialog(self):
        dialog = hotkeys.HotkeyDialog(None, self.manager, hotkeys.DEFAULTS, self.path)
        self.addCleanup(dialog.close)
        return dialog

    def test_conflict_keeps_dialog_open_and_does_not_save(self):
        self.api.RegisterHotKey.return_value = False
        dialog = self.dialog()
        dialog.save()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.manager.active)
        self.assertIn('등록할 수 없습니다', dialog.error.text())

    def test_save_failure_unregisters_proposed_keys(self):
        dialog = self.dialog()
        with patch.object(hotkeys, 'save_settings', side_effect=OSError('read only')):
            dialog.save()
        self.assertFalse(self.manager.active)
        self.assertEqual(dialog.settings, hotkeys.DEFAULTS)
        self.assertIn('read only', dialog.error.text())

    def test_dialog_save_and_defaults(self):
        dialog = self.dialog()
        dialog.editors['select'].clear()
        dialog.editors['drag'].setKeySequence(QKeySequence('F8'))
        dialog.save()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(hotkeys.load_settings(self.path)['drag'], 'F8')
        self.assertEqual(hotkeys.load_settings(self.path)['select'], '')
        dialog.restore_defaults()
        self.assertEqual(dialog.editors['select'].keySequence().toString(), 'Ctrl+Alt+1')

    def test_actions_route_and_selection_guards(self):
        controller = Mock()
        controller.hotkey_dialog_open = False
        controller.selector.isVisible.return_value = False
        controller.device_timer.isActive.return_value = False
        for action in hotkeys.ACTIONS:
            main.Controller.activate_hotkey(controller, action)
        self.assertEqual(controller.select_region.call_count, 2)
        controller.select_region.assert_any_call(single_shot=True)
        controller.toggle.assert_called_once()
        controller.retry_translation.assert_called_once()
        controller.selector.isVisible.return_value = True
        main.Controller.activate_hotkey(controller, 'toggle')
        controller.hotkey_dialog_open = True
        main.Controller.activate_hotkey(controller, 'toggle')
        controller.toggle.assert_called_once()

    def test_cancel_settings_restores_original_shortcuts(self):
        controller = Mock()
        controller.hotkey_dialog_open = False
        controller.hotkeys = self.manager
        controller.hotkey_settings = dict(hotkeys.DEFAULTS)
        self.manager.apply(hotkeys.DEFAULTS)
        dialog = Mock()
        dialog.exec.return_value = QDialog.DialogCode.Rejected
        with patch.object(main, 'HotkeyDialog', return_value=dialog):
            main.Controller.configure_hotkeys(controller)
        self.assertFalse(controller.hotkey_dialog_open)
        self.assertEqual(set(self.manager.active.values()), set(hotkeys.ACTIONS))
        self.assertEqual(controller.hotkey_settings, hotkeys.DEFAULTS)

    def test_controller_initializes_and_releases_shortcuts(self):
        with patch.object(main.Engine, 'start'), \
                patch.object(main, 'CaptureWithoutApp'), \
                patch.object(main, 'load_api_key', return_value=''), \
                patch.object(main, 'load_openai_settings', return_value=dict(main.OPENAI_DEFAULTS)), \
                patch.object(main, 'load_translation_provider', return_value='luna'), \
                patch.object(main, 'load_settings', return_value=dict(hotkeys.DEFAULTS)), \
                patch.object(main, 'WindowsHotkeys', return_value=self.manager), \
                patch.object(main, 'allow_capture'):
            window = main.Controller()
            window.show()
            self.app.processEvents()
            try:
                self.assertEqual(window.hotkey_button.text(), '단축키 설정')
                self.assertIn('Ctrl+Alt+2', window.drag_button.toolTip())
                self.assertEqual(len(self.manager.active), 4)
            finally:
                window.close()
            self.assertFalse(self.manager.active)


if __name__ == '__main__':
    unittest.main()
