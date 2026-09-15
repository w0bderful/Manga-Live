import asyncio
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from core import Box
import main
from translation import OpenAICompatibleTranslationClient, chat_endpoint, create_translation_client, list_openai_models


class FakeApiHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.requests.append((self.path, dict(self.headers), None))
        self.respond()

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append((self.path, dict(self.headers), payload))
        self.respond()

    def respond(self):
        response = json.dumps(self.server.reply, ensure_ascii=False).encode('utf-8')
        self.send_response(self.server.status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_):
        pass


class CompatibleApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), FakeApiHandler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.base_url = f'http://127.0.0.1:{cls.server.server_port}/proxy/v1'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=2)

    def setUp(self):
        self.server.status = 200
        self.server.reply = {'choices':[{'message':{'content':'안녕하세요'}, 'finish_reason':'stop'}]}
        self.server.requests = []

    def client(self, key='dummy-key'):
        client = create_translation_client('openai', key, base_url=self.base_url, model='custom-model')
        self.addCleanup(client.close)
        return client

    def test_real_http_request_uses_custom_address_key_model_and_japanese_text(self):
        result = asyncio.run(self.client().translate('こんにちは'))
        self.assertEqual(result.text, '안녕하세요')
        path, headers, payload = self.server.requests[0]
        self.assertEqual(path, '/proxy/v1/chat/completions')
        self.assertEqual(headers['Authorization'], 'Bearer dummy-key')
        self.assertEqual(payload['model'], 'custom-model')
        self.assertFalse(payload['stream'])
        self.assertEqual(payload['messages'][1], {'role':'user', 'content':'こんにちは'})

    def test_local_server_without_key_and_full_endpoint(self):
        client = OpenAICompatibleTranslationClient('', self.base_url+'/chat/completions?version=test', 'local-model')
        self.addCleanup(client.close)
        self.assertEqual(asyncio.run(client.translate('こんにちは')).text, '안녕하세요')
        path, headers, _ = self.server.requests[0]
        self.assertEqual(path, '/proxy/v1/chat/completions?version=test')
        self.assertNotIn('Authorization', headers)

    def test_http_errors_close_connection_and_retry_can_succeed(self):
        client = self.client()
        for status in (401, 403, 404, 429, 500, 302):
            self.server.status = status
            with self.assertRaisesRegex(RuntimeError, str(status)):
                asyncio.run(client.translate('こんにちは'))
            self.assertIsNone(client.connection)
        self.server.status = 200
        self.assertEqual(asyncio.run(client.translate('こんにちは')).text, '안녕하세요')

    def test_empty_malformed_refused_and_truncated_responses_are_rejected(self):
        for reply in ([], {}, {'error':{}}, {'choices':[None]},
                      {'choices':[{'message':{'content':''}}]},
                      {'choices':[{'message':{'content':None, 'refusal':'refused'}}]},
                      {'choices':[{'message':{'content':'partial'}, 'finish_reason':'length'}]}):
            self.server.reply = reply
            with self.subTest(reply=reply), self.assertRaises(RuntimeError):
                asyncio.run(self.client().translate('こんにちは'))

    def test_text_content_parts(self):
        self.server.reply = {'choices':[{'message':{'content':[{'type':'text','text':'안녕'},
                                                              {'type':'text','text':'하세요'}]}}]}
        self.assertEqual(asyncio.run(self.client().translate('こんにちは')).text, '안녕하세요')

    def test_url_validation_and_normalization(self):
        self.assertEqual(chat_endpoint('https://example.com'), ('https', 'example.com', None, '/v1/chat/completions'))
        self.assertEqual(chat_endpoint('https://example.com/v1/chat/completions/')[3], '/v1/chat/completions')
        for url in ('', 'file:///tmp/key', 'ftp://example.com', 'https://', 'https://user:pass@example.com',
                    'https://example.com/#fragment', 'http://localhost:99999', 'https://exa mple.com'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                chat_endpoint(url)
        with self.assertRaises(ValueError):
            OpenAICompatibleTranslationClient('', self.base_url, '')
        with self.assertRaises(ValueError):
            OpenAICompatibleTranslationClient('key\nInjected: header', self.base_url, 'model')

    def test_connection_timeout_is_reported_without_key(self):
        client = self.client()
        client.connection = Mock()
        client.connection.request.side_effect = TimeoutError('dummy-key')
        with self.assertRaisesRegex(RuntimeError, '시간 초과') as error:
            asyncio.run(client.translate('こんにちは'))
        self.assertNotIn('dummy-key', str(error.exception))
        self.assertIsNone(client.connection)

    def test_engine_uses_compatible_endpoint_without_requiring_api_key(self):
        signals = Mock()
        engine = main.Engine(signals, device='cpu', provider='openai', base_url=self.base_url, model='local-model')
        signals.finished.emit.side_effect = lambda *_: engine.stop_event.set()
        backend = Mock()
        backend.detect.return_value = [Box(0, 0, 10, 10)]
        backend.recognize.return_value = 'こんにちは'
        torch = Mock()
        torch.inference_mode.side_effect = nullcontext
        with patch.object(main, 'OcrBackend', return_value=backend), patch.object(main, 'validate_device'), \
             patch.dict('sys.modules', {'torch':torch}):
            engine.submit((0, np.zeros((10, 10, 3), np.uint8), True))
            engine.start()
            engine.join(timeout=5)
            engine.stop_event.set()
            engine.join(timeout=1)
        self.assertFalse(engine.is_alive())
        signals.failed.emit.assert_not_called()
        self.assertEqual(signals.result.emit.call_args.args[1][0][1], '안녕하세요')
        self.assertEqual(self.server.requests[0][2]['model'], 'local-model')

    def test_model_list_authentication_ids_and_endpoint(self):
        self.server.reply = {'data':[{'id':'model-b','owned_by':'test'}, {'id':'model-a'},
                                     {'id':'model-b'}, {'id':''}, None]}
        self.assertEqual(list_openai_models(self.base_url+'/chat/completions?version=1', 'dummy-key'),
                         ['model-a', 'model-b'])
        path, headers, body = self.server.requests[0]
        self.assertEqual(path, '/proxy/v1/models?version=1')
        self.assertEqual(headers['Authorization'], 'Bearer dummy-key')
        self.assertIsNone(body)

    def test_model_list_worker_and_empty_local_server(self):
        self.server.reply = {'data':[]}
        future = main.request_model_list(self.base_url, '')
        self.assertEqual(future.result(timeout=3), [])
        self.assertNotIn('Authorization', self.server.requests[0][1])

    def test_model_list_errors_and_malformed_responses(self):
        for status in (401, 403, 404, 429, 500):
            self.server.status = status
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, str(status)):
                list_openai_models(self.base_url, 'dummy')
        self.server.status = 200
        for data in ([], {}, {'data':'wrong'}, {'data':[{'id':123}]}):
            self.server.reply = data
            with self.subTest(data=data), self.assertRaises(RuntimeError):
                list_openai_models(self.base_url)

    def test_model_list_timeout_closes_connection(self):
        connection = Mock()
        connection.getresponse.side_effect = TimeoutError('dummy-key')
        with patch('translation.http.client.HTTPConnection', return_value=connection):
            with self.assertRaisesRegex(RuntimeError, '시간 초과') as error:
                list_openai_models(self.base_url, 'dummy-key')
        self.assertNotIn('dummy-key', str(error.exception))
        connection.close.assert_called_once()
