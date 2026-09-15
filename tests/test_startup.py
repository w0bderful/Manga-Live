import io
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import main


class StartupTests(unittest.TestCase):
    def test_engine_initialization_failure_identifies_engine_and_generation(self):
        signals = Mock()
        engine = main.Engine(signals, api_key='dummy')
        engine.generation = 3
        with patch.object(main, 'create_translation_client', side_effect=ValueError('invalid settings')), \
             self.assertLogs(main.log, level='ERROR'):
            engine.run()
        source, generation, message = signals.failed.emit.call_args.args
        self.assertIs(source, engine)
        self.assertEqual(generation, 3)
        self.assertIn('invalid settings', message)

    def test_engine_frame_failure_identifies_engine_and_generation(self):
        signals = Mock()
        engine = main.Engine(signals, api_key='dummy')
        engine.generation = 5
        signals.failed.emit.side_effect = lambda *_: engine.stop_event.set()
        engine.submit((5, None, True))
        with patch.dict('sys.modules', {'torch': Mock()}), \
             patch.object(main, 'validate_device', side_effect=RuntimeError('GPU unavailable')), \
             self.assertLogs(main.log, level='ERROR'):
            engine.run()
        source, generation, message = signals.failed.emit.call_args.args
        self.assertIs(source, engine)
        self.assertEqual(generation, 5)
        self.assertIn('GPU unavailable', message)
        signals.result.emit.assert_not_called()

    def test_log_file_failure_still_opens_ui_and_runs_event_loop(self):
        for stderr in (io.StringIO(), None):
            with self.subTest(stderr=stderr), \
                 patch.object(logging.root, 'handlers', []), \
                 patch.object(logging.root, 'level', logging.WARNING), \
                 patch.object(main.sys, 'stderr', stderr), \
                 patch.object(main.sys, 'platform', 'win32'), \
                 patch.object(logging, 'FileHandler', side_effect=PermissionError('log locked')), \
                 patch.object(main, 'QApplication') as application, \
                 patch.object(main, 'Controller') as controller, \
                 patch.object(main, 'QLabel') as label, \
                 patch.object(main, 'allow_capture'):
                application.return_value.exec.return_value = 0
                try:
                    self.assertEqual(main.main(), 0)
                    controller.return_value.show.assert_called_once()
                    application.return_value.exec.assert_called_once()
                    self.assertIn('파일 기록 없이 실행', label.call_args.args[0])
                    controller.return_value.layout.return_value.addWidget.assert_called_once_with(label.return_value)
                    self.assertTrue(logging.root.handlers)
                    if stderr is not None:
                        self.assertIn('Log file unavailable', stderr.getvalue())
                finally:
                    for handler in logging.root.handlers:
                        handler.close()

    def test_writable_log_file_records_messages(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(main, 'ROOT', Path(folder)), \
             patch.object(logging.root, 'handlers', []), \
             patch.object(logging.root, 'level', logging.WARNING):
            try:
                self.assertEqual(main.configure_logging(), '')
                main.log.info('startup logging test')
            finally:
                for handler in logging.root.handlers:
                    handler.close()
            self.assertIn('startup logging test', (Path(folder)/'manga-live.log').read_text(encoding='utf-8'))
