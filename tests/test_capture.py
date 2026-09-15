import multiprocessing
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import window_capture as capture_module


def synthetic_worker(connection, owner_pid):
    try:
        while True:
            region = connection.recv()
            if region['left'] == -1:
                threading.Event().wait()
            connection.send((True, np.full((2, 2, 3), region['left'], np.uint8)))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


def failed_worker(connection, owner_pid):
    connection.send((False, 'synthetic startup failure'))
    connection.close()


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.capture = capture_module.CaptureWithoutApp()
        self.addCleanup(self.capture.close)
        self.region = dict(left=7, top=0, width=2, height=2)

    def result(self, region):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            pixels = self.capture.grab(region)
            if pixels is not None:
                return pixels
            time.sleep(0.01)
        self.fail('capture did not complete')

    def test_timeout_terminates_worker_and_next_request_recovers(self):
        with patch.object(capture_module, 'capture_worker', synthetic_worker):
            np.testing.assert_array_equal(self.result(self.region), np.full((2, 2, 3), 7))
            pid = self.capture.process.pid
            stalled = {**self.region, 'left': -1}
            self.capture.grab(stalled)
            self.capture.pending = (stalled, time.monotonic()-6, self.capture.version)
            with self.assertRaisesRegex(RuntimeError, '응답이 지연'):
                self.capture.grab(stalled)
            self.assertIsNone(self.capture.pending)
            self.assertNotIn(pid, [p.pid for p in multiprocessing.active_children()])
            np.testing.assert_array_equal(self.result(self.region), np.full((2, 2, 3), 7))

    def test_invalidate_and_region_change_discard_blocked_work(self):
        with patch.object(capture_module, 'capture_worker', synthetic_worker):
            self.result(self.region)
            self.capture.grab({**self.region, 'left': -1})
            self.capture.invalidate()
            self.assertIsNone(self.capture.pending)
            self.assertIsNotNone(self.result(self.region))
            self.capture.grab({**self.region, 'left': -1})
            self.assertIsNotNone(self.result(self.region))

    def test_startup_failure_is_reported_and_retry_works(self):
        with patch.object(capture_module, 'capture_worker', failed_worker):
            with self.assertRaisesRegex(RuntimeError, 'startup failure'):
                self.result(self.region)
        self.assertIsNone(self.capture.process)
        with patch.object(capture_module, 'capture_worker', synthetic_worker):
            self.assertIsNotNone(self.result(self.region))

    def test_worker_reports_native_initialization_failure(self):
        connection = Mock()
        with patch.object(capture_module, 'NativeWindows', side_effect=OSError('synthetic init')) as native:
            capture_module.capture_worker(connection, 12345)
        native.assert_called_once_with(12345)
        success, message = connection.send.call_args.args[0]
        self.assertFalse(success)
        self.assertIn('synthetic init', message)
        connection.close.assert_called_once()

    def test_close_stops_child_and_prevents_new_requests(self):
        with patch.object(capture_module, 'capture_worker', synthetic_worker):
            self.result(self.region)
            pid = self.capture.process.pid
            self.capture.close()
        self.assertNotIn(pid, [p.pid for p in multiprocessing.active_children()])
        self.assertIsNone(self.capture.grab(self.region))
