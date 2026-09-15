import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from contextlib import ExitStack
from concurrent.futures import Future
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PyQt6.QtWidgets import QApplication
import api_settings
import hotkeys
import main


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def controller(self, load=None, openai_settings=None, provider='luna'):
        stack = ExitStack()
        self.addCleanup(stack.close)
        manager = Mock(active={})
        manager.apply.return_value = []
        for target, options in [
            ('Engine.start', {}), ('CaptureWithoutApp', {}),
            ('load_api_key', {'side_effect': load} if load else {'return_value': ''}),
            ('load_openai_settings', {'return_value': openai_settings or dict(main.OPENAI_DEFAULTS)}),
            ('load_translation_provider', {'return_value': provider}),
            ('save_api_keys', {}),
            ('load_settings', {'return_value': dict(hotkeys.DEFAULTS)}),
            ('WindowsHotkeys', {'return_value': manager}), ('allow_capture', {}),
        ]:
            stack.enter_context(patch('main.' + target, **options))
        window = main.Controller()
        window.timer.stop()
        self.addCleanup(window.close)
        window.region = dict(left=0, top=0, width=30, height=30)
        window.capture.grab.return_value = np.full((30, 30, 3), 255, np.uint8)
        return window

    def select_model(self, window, model):
        if window.openai_model.findData(model) < 0:
            window.openai_model.addItem(model, model)
        window.openai_model.setCurrentIndex(window.openai_model.findData(model))

    def waiting_snapshot(self):
        window = self.controller()
        window.single_shot = window.running = True
        window.finish_snapshot_capture(window.engine.generation)
        self.assertIn('자동으로 번역', window.status.text())
        return window

    def queue_failure(self, window, message):
        engine = window.engine
        generation = engine.generation
        thread = threading.Thread(target=lambda: engine.signals.failed.emit(engine, generation, message))
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_queued_old_failure_does_not_cancel_new_snapshot(self):
        window = self.waiting_snapshot()
        self.queue_failure(window, 'old failure')
        window.reset_frame()
        window.single_shot = window.running = True
        window.status.setText('new snapshot')
        self.app.processEvents()
        self.assertTrue(window.running)
        self.assertEqual(window.status.text(), 'new snapshot')

    def test_queued_failure_from_replaced_engine_is_ignored(self):
        window = self.waiting_snapshot()
        self.queue_failure(window, 'old engine failure')
        window.engine = main.Engine(main.Signals())
        window.status.setText('new engine')
        self.app.processEvents()
        self.assertTrue(window.running)
        self.assertEqual(window.status.text(), 'new engine')

    def test_current_failure_stops_snapshot_and_allows_retry(self):
        window = self.waiting_snapshot()
        self.queue_failure(window, 'current failure')
        self.app.processEvents()
        self.assertFalse(window.running)
        self.assertEqual(window.toggle_button.text(), '시작')
        self.assertEqual(window.status.text(), 'current failure')
        with patch.object(main.QTimer, 'singleShot'):
            window.retry_translation()
        self.assertTrue(window.running)
        self.assertFalse(window.submitted)

    def test_pause_after_settings_edit_cancels_pending_results(self):
        window = self.controller(load=lambda _: 'dummy-key')
        window.worker_ready = True
        window.toggle()
        self.assertTrue(window.running)
        old_generation = window.engine.generation
        pixels = window.capture.grab.return_value
        window.latest = pixels
        window.api_key.setText('edited-key')
        window.toggle()
        self.assertFalse(window.running)
        self.assertEqual(window.toggle_button.text(), '시작')
        self.assertEqual(window.status.text(), '일시정지')
        self.assertGreater(window.engine.cancel_before, old_generation)
        window.accept_result(old_generation, [(main.Box(0, 0, 20, 20), 'late result')], pixels)
        self.assertFalse(window.overlay.rows)
        window.toggle()
        self.assertFalse(window.running)
        self.assertIn('설정 적용', window.status.text())

    def test_region_lookup_failure_recovers_and_allows_reselection(self):
        window = self.controller()
        window.single_shot = True
        window.hide()
        with patch.object(main, 'native_region', side_effect=OSError('window lookup failed')), \
             self.assertLogs(main.log, level='ERROR'):
            window.set_region(main.QRect(0, 0, 100, 100))
        self.assertIsNone(window.region)
        self.assertFalse(window.running)
        self.assertFalse(window.isHidden())
        self.assertTrue(window.overlay.isHidden())
        self.assertIn('영역을 다시 선택', window.status.text())
        region = dict(left=0, top=0, width=100, height=100)
        with patch.object(main, 'native_region', return_value=region), \
             patch.object(main.QTimer, 'singleShot'):
            window.set_region(main.QRect(0, 0, 100, 100))
        self.assertEqual(window.region, region)
        self.assertTrue(window.running)

    def test_snapshot_survives_key_apply_and_is_submitted_once(self):
        window = self.waiting_snapshot()
        pixels = window.latest
        window.api_key.setText('dummy')
        window.change_device()
        window.finish_device_change()
        window.ready()
        window.ready()
        self.assertTrue(window.running)
        self.assertIs(window.latest, pixels)
        self.assertEqual(window.engine.jobs.qsize(), 1)
        self.assertIs(window.engine.jobs.get_nowait()[1], pixels)

    def test_snapshot_survives_switch_to_provider_without_key(self):
        window = self.waiting_snapshot()
        pixels = window.latest
        window.translation_mode.setCurrentIndex(window.translation_mode.findData('deepl'))
        self.assertTrue(window.running)
        self.assertIs(window.latest, pixels)
        window.deepl_api_key.setText('dummy:fx')
        window.change_device()
        window.finish_device_change()
        window.ready()
        self.assertEqual(window.engine.provider, 'deepl')
        self.assertIs(window.engine.jobs.get_nowait()[1], pixels)

    def test_cancel_during_settings_change_does_not_resume(self):
        window = self.waiting_snapshot()
        window.api_key.setText('dummy')
        window.change_device()
        window.toggle()
        window.finish_device_change()
        window.ready()
        self.assertFalse(window.running)
        self.assertIsNone(window.latest)
        self.assertTrue(window.engine.jobs.empty())

    def test_capture_pending_during_apply_restarts_for_new_generation(self):
        window = self.controller()
        window.single_shot = window.running = True
        old_generation = window.engine.generation
        window.api_key.setText('dummy')
        window.change_device()
        window.finish_snapshot_capture(old_generation)
        self.assertIsNone(window.latest)
        window.finish_device_change()
        window.ready()
        self.assertIsNotNone(window.latest)
        self.assertEqual(window.engine.jobs.qsize(), 1)

    def test_pending_capture_callback_does_not_recapture_after_submission(self):
        window = self.controller()
        window.single_shot = window.running = True
        window.api_key.setText('dummy')
        window.change_device()
        window.finish_device_change()
        pixels = window.capture.grab.return_value
        window.capture.grab.side_effect = [None, pixels]
        with patch.object(main.QTimer, 'singleShot'):
            window.ready()
        window.finish_snapshot_capture(window.engine.generation)
        window.finish_snapshot_capture(window.engine.generation)
        self.assertEqual(window.capture.grab.call_count, 2)
        self.assertEqual(window.engine.jobs.qsize(), 1)

    def test_corrupt_settings_open_ui_and_save_recovers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'api-keys.json'
            path.write_text('{broken')
            with patch.object(api_settings, 'API_KEYS_FILE', path), \
                 patch.object(api_settings, 'LEGACY_FILES', {key: Path(folder)/key for key in api_settings.LEGACY_FILES}):
                window = self.controller(api_settings.load_api_key)
                self.assertTrue(window.api_settings_error)
                self.assertIn('읽지 못했습니다', window.api_settings_note.text())
                self.assertEqual(path.read_text(), '{broken')
                window.api_key.setText('dummy')
                with patch.object(main, 'save_api_keys', api_settings.save_api_keys):
                    self.assertTrue(window.save_api_key())
                self.assertFalse(window.api_settings_error)
                self.assertEqual(api_settings.load_api_key(), 'dummy')
                self.assertEqual(next((Path(folder)/'api-key-backups').iterdir()).read_text(), '{broken')

    def test_openai_settings_apply_preserves_snapshot_and_rejects_stale_config(self):
        window = self.waiting_snapshot()
        pixels = window.latest
        window.openai_base_url.setText('http://localhost:1234/v1')
        self.select_model(window, 'local-model')
        window.translation_mode.setCurrentIndex(window.translation_mode.findData('openai'))
        window.finish_device_change()
        window.ready()
        self.assertEqual(window.engine.api_key, '')
        self.assertEqual(window.engine.base_url, 'http://localhost:1234/v1')
        self.assertEqual(window.engine.model, 'local-model')
        self.assertIs(window.engine.jobs.get_nowait()[1], pixels)
        self.assertTrue(window.translation_is_current())
        window.openai_base_url.setText('http://localhost:5678/v1')
        self.assertFalse(window.translation_is_current())
        self.select_model(window, 'local-model')
        window.change_device()
        window.finish_device_change()
        window.ready()
        self.assertEqual(window.engine.base_url, 'http://localhost:5678/v1')
        self.select_model(window, 'different-model')
        self.assertFalse(window.translation_is_current())

    def test_invalid_openai_config_is_reported_before_start(self):
        window = self.controller()
        window.translation_mode.setCurrentIndex(window.translation_mode.findData('openai'))
        self.assertIn('모델 불러오기', window.status.text())
        self.assertFalse(hasattr(window, 'device_timer'))
        self.select_model(window, 'local-model')
        window.openai_base_url.setText('not-a-url')
        window.change_device()
        self.assertIn('API 주소', window.status.text())
        self.assertFalse(hasattr(window, 'device_timer'))

    def test_topmost_toggle_preserves_overlay_and_translation_state(self):
        window = self.waiting_snapshot()
        pixels, generation = window.latest, window.engine.generation
        flag = main.Qt.WindowType.WindowStaysOnTopHint
        self.assertTrue(window.windowFlags() & flag)
        window.always_on_top.setChecked(False)
        self.assertFalse(window.windowFlags() & flag)
        self.assertTrue(window.overlay.windowFlags() & flag)
        window.always_on_top.setChecked(True)
        self.assertTrue(window.windowFlags() & flag)
        self.assertIs(window.latest, pixels)
        self.assertEqual(window.engine.generation, generation)
        self.assertTrue(window.running)

    def test_openai_fields_are_saved_by_controller(self):
        window = self.controller()
        window.openai_base_url.setText('http://localhost:1234/v1')
        window.openai_api_key.setText('dummy-key')
        self.select_model(window, 'local-model')
        self.assertTrue(window.api_key_save_timer.isActive())
        window.save_api_key()
        main.save_api_keys.assert_called_with('', '', openai_key='dummy-key',
                                             openai_base_url='http://localhost:1234/v1',
                                             openai_model='local-model', translation_provider='luna')

    def test_startup_restores_provider_fields_and_engine(self):
        for provider in ('luna', 'deepl', 'openai'):
            with self.subTest(provider=provider):
                window = self.controller(load=lambda service: service + '-key', provider=provider,
                                         openai_settings={'base_url':'http://localhost:1234/v1','model':'local-model'})
                self.assertEqual(window.translation_mode.currentData(), provider)
                self.assertEqual(window.engine.provider, provider)
                self.assertEqual(window.engine.api_key, provider + '-key')
                self.assertEqual(window.api_key.isHidden(), provider != 'luna')
                self.assertEqual(window.deepl_api_key.isHidden(), provider != 'deepl')
                self.assertEqual(window.openai_panel.isHidden(), provider != 'openai')
                self.assertEqual(window.engine.device, 'cuda')
                main.save_api_keys.assert_not_called()
                if provider == 'openai':
                    self.assertEqual(window.engine.model, 'local-model')
                    self.assertEqual(window.engine.base_url, 'http://localhost:1234/v1')

    def test_service_selection_saved_even_before_required_fields_are_ready(self):
        window = self.controller()
        for provider in ('deepl', 'openai', 'luna'):
            window.translation_mode.setCurrentIndex(window.translation_mode.findData(provider))
            self.assertEqual(main.save_api_keys.call_args.kwargs['translation_provider'], provider)

    def fetch_models(self, window, models):
        future = Future()
        with patch.object(main, 'request_model_list', return_value=future):
            window.load_models()
        future.set_result(models)
        window.finish_model_list()

    def test_model_list_load_selection_and_save(self):
        window = self.controller()
        window.openai_base_url.setText('http://localhost:1234/v1')
        window.openai_api_key.setText('dummy')
        future = Future()
        with patch.object(main, 'request_model_list', return_value=future) as request:
            window.load_models()
            window.load_models()
            request.assert_called_once_with('http://localhost:1234/v1', 'dummy')
        self.assertFalse(window.openai_model.isEditable())
        self.assertFalse(window.load_models_button.isEnabled())
        self.assertTrue(window.models_timer.isActive())
        future.set_result(['model-a', 'model-b'])
        window.finish_model_list()
        self.assertEqual(window.openai_model.count(), 2)
        self.assertEqual(window.openai_model.currentIndex(), -1)
        window.openai_model.setCurrentIndex(1)
        self.assertEqual(window.selected_openai_settings()[1], 'model-b')
        self.assertTrue(window.api_key_save_timer.isActive())
        window.save_api_key()
        self.assertEqual(main.save_api_keys.call_args.kwargs['openai_model'], 'model-b')

    def test_saved_model_restored_and_retained_on_refresh(self):
        window = self.controller(openai_settings={'base_url':'http://localhost:1234/v1','model':'saved-model'})
        self.assertEqual(window.openai_model.currentData(), 'saved-model')
        self.fetch_models(window, ['other-model', 'saved-model'])
        self.assertEqual(window.openai_model.currentData(), 'saved-model')
        self.fetch_models(window, ['other-model'])
        self.assertIsNone(window.openai_model.currentData())
        self.assertIn('기존 모델', window.models_note.text())

    def test_url_and_key_changes_invalidate_pending_model_response(self):
        window = self.controller()
        for field, value in [(window.openai_base_url, 'http://localhost:1234/v1'),
                             (window.openai_api_key, 'new-key')]:
            self.select_model(window, 'old-model')
            future = Future()
            with patch.object(main, 'request_model_list', return_value=future):
                window.load_models()
            field.setText(value)
            self.assertEqual(window.openai_model.count(), 0)
            future.set_result(['stale-model'])
            window.finish_model_list()
            self.assertEqual(window.openai_model.count(), 0)
            self.assertTrue(window.load_models_button.isEnabled())
            self.assertIn('다시 불러오세요', window.models_note.text())

    def test_failed_refresh_keeps_selection_and_allows_retry(self):
        window = self.controller()
        self.select_model(window, 'saved-model')
        future = Future()
        with patch.object(main, 'request_model_list', return_value=future):
            window.load_models()
        future.set_exception(RuntimeError('HTTP 401: API 키를 확인하세요.'))
        window.finish_model_list()
        self.assertEqual(window.openai_model.currentData(), 'saved-model')
        self.assertIn('401', window.models_note.text())
        self.assertTrue(window.load_models_button.isEnabled())
        self.fetch_models(window, [])
        self.assertIsNone(window.openai_model.currentData())
        self.assertFalse(window.openai_model.isEnabled())
        self.assertIn('모델이 없습니다', window.models_note.text())

    def test_close_while_loading_does_not_wait_or_apply_late_results(self):
        window = self.controller()
        window.show()
        self.app.processEvents()
        future = Future()
        with patch.object(main, 'request_model_list', return_value=future):
            window.load_models()
        window.close()
        self.assertFalse(window.models_timer.isActive())
        future.set_result(['late-model'])
        window.finish_model_list()
        self.assertEqual(window.openai_model.count(), 0)
