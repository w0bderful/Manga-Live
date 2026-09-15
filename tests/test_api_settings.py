from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import api_settings as settings


class ApiSettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'api-keys.json'
        self.legacy = {field: self.root / (field + '.json') for field in settings.LEGACY_FILES}
        for name, value in [('API_KEYS_FILE', self.path), ('LEGACY_FILES', self.legacy)]:
            patcher = patch.object(settings, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_corrupt_file_preserved_and_replaced_on_save(self):
        for content in (b'{broken', b'[]', b'\xff\xfe'):
            with self.subTest(content=content):
                self.path.write_bytes(content)
                with self.assertRaises(ValueError):
                    settings.load_api_key()
                self.assertEqual(self.path.read_bytes(), content)
                settings.save_api_keys('new-luna', 'new-deepl')
                self.assertEqual(settings.load_api_key(), 'new-luna')
                self.assertEqual(settings.load_api_key('deepl'), 'new-deepl')
                self.assertIn(content, [p.read_bytes() for p in (self.root/'api-key-backups').iterdir()])

    def test_invalid_field_does_not_prevent_other_provider_loading(self):
        self.path.write_text('{"kie_api_key": 123, "deepl_api_key": "valid-deepl"}')
        with self.assertRaises(ValueError):
            settings.load_api_key()
        self.assertEqual(settings.load_api_key('deepl'), 'valid-deepl')
        settings.save_api_keys('new-luna', settings.load_api_key('deepl'))
        self.assertEqual(settings.load_api_key('deepl'), 'valid-deepl')

    def test_bad_legacy_file_is_backed_up_and_valid_key_migrated(self):
        self.legacy['kie_api_key'].write_text('{broken')
        self.legacy['deepl_api_key'].write_text('{"DEEPL_API_KEY": "old-deepl"}')
        with self.assertRaises(ValueError):
            settings.load_api_key()
        data = settings.migrate_keys(recover=True)
        self.assertEqual(data['deepl_api_key'], 'old-deepl')
        self.assertFalse(any(path.exists() for path in self.legacy.values()))
        self.assertEqual(next((self.root/'api-key-backups').iterdir()).read_text(), '{broken')

    def test_failed_backup_keeps_original(self):
        self.path.write_text('{broken')
        with patch.object(Path, 'replace', side_effect=PermissionError('blocked')):
            with self.assertRaises(PermissionError):
                settings.save_api_keys('new', '')
        self.assertEqual(self.path.read_text(), '{broken')

    def test_corrupt_legacy_does_not_hide_valid_provider_keys(self):
        self.path.write_text('{"deepl_api_key": "valid-deepl"}')
        self.legacy['kie_api_key'].write_text('{broken')
        self.assertEqual(settings.load_api_key('deepl'), 'valid-deepl')
        with self.assertRaises(ValueError):
            settings.load_api_key('luna')
        settings.save_api_keys('new-luna', settings.load_api_key('deepl'))
        self.assertEqual(settings.load_api_key('deepl'), 'valid-deepl')

    def test_valid_settings_and_environment_fallback(self):
        with patch.dict('os.environ', {'KIE_API_KEY': 'environment-key'}):
            self.assertEqual(settings.load_api_key(), 'environment-key')
            settings.save_api_keys('saved-key', '')
            self.assertEqual(settings.load_api_key(), 'saved-key')
        self.assertFalse((self.root/'api-key-backups').exists())

    def test_openai_settings_round_trip_and_legacy_save_preserves_them(self):
        self.assertEqual(settings.load_openai_settings(), settings.OPENAI_DEFAULTS)
        settings.save_api_keys('luna', 'deepl', openai_key=' custom-key ',
                               openai_base_url=' http://localhost:1234/v1 ', openai_model=' custom-model ')
        self.assertEqual(settings.load_api_key('openai'), 'custom-key')
        self.assertEqual(settings.load_openai_settings(),
                         {'base_url':'http://localhost:1234/v1', 'model':'custom-model'})
        settings.save_api_keys('changed-luna', 'deepl')
        self.assertEqual(settings.load_api_key('openai'), 'custom-key')
        self.assertEqual(settings.load_openai_settings()['model'], 'custom-model')

    def test_invalid_openai_setting_is_backed_up_on_recovery(self):
        self.path.write_text('{"openai_base_url":123,"kie_api_key":"luna"}')
        with self.assertRaises(ValueError):
            settings.load_openai_settings()
        settings.save_api_keys('luna', '', openai_base_url='http://localhost:1234/v1', openai_model='test')
        self.assertEqual(settings.load_openai_settings()['model'], 'test')
        self.assertEqual(len(list((self.root/'api-key-backups').iterdir())), 1)

    def test_translation_provider_round_trip_and_old_save_preserves_choice(self):
        self.assertEqual(settings.load_translation_provider(), 'luna')
        for provider in ('openai', 'deepl', 'luna'):
            settings.save_api_keys('luna-key', 'deepl-key', translation_provider=provider)
            self.assertEqual(settings.load_translation_provider(), provider)
            settings.save_api_keys('new-luna-key', 'deepl-key')
            self.assertEqual(settings.load_translation_provider(), provider)

    def test_unknown_or_invalid_provider_falls_back_without_changing_file(self):
        for content in ('{}', '{"translation_provider":"removed-service"}',
                        '{"translation_provider":null}', '{"translation_provider":[]}'):
            self.path.write_text(content)
            self.assertEqual(settings.load_translation_provider(), 'luna')
            self.assertEqual(self.path.read_text(), content)
