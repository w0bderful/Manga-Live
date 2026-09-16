import json
import os
from pathlib import Path
from uuid import uuid4
from app_settings import read_settings, update_settings

ROOT = Path(__file__).resolve().parent
API_KEYS_FILE = ROOT / 'api-keys.json'
FIELDS = {'luna': ('kie_api_key', 'KIE_API_KEY'), 'deepl': ('deepl_api_key', 'DEEPL_API_KEY'),
          'openai': ('openai_api_key', 'OPENAI_API_KEY')}
OPENAI_DEFAULTS = {'base_url': 'https://api.openai.com/v1', 'model': ''}
LEGACY_FILES = {'kie_api_key': ROOT / 'kie-api-key.json',
                'deepl_api_key': ROOT / 'deepl-api-key.json'}


def read_keys(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('API 키 파일은 JSON 객체여야 합니다.')
    return data


def write_keys(data):
    temporary = API_KEYS_FILE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(API_KEYS_FILE)


def read_key_file(path, fields, recover=False, required=False):
    if not path.exists():
        return {}
    data = {}
    try:
        data = read_keys(path)
        if any((required or field in data) and not isinstance(data.get(field), str)
               for field in fields):
            raise ValueError('API 키는 문자열이어야 합니다.')
        return data
    except (ValueError, UnicodeError):
        if not recover:
            raise
        backup_dir = path.parent / 'api-key-backups'
        backup_dir.mkdir(exist_ok=True)
        backup = backup_dir / f'{path.stem}-{uuid4().hex}{path.suffix}'
        # Keep the original in place until the replacement has been written successfully.
        backup.write_bytes(path.read_bytes())
        return {key: value for key, value in data.items()
                if key not in fields or isinstance(value, str)}


def migrate_keys(recover=False, provider=None):
    data = (read_key_file(API_KEYS_FILE, [field for field, _ in FIELDS.values()], recover=True)
            if recover else read_keys(API_KEYS_FILE))
    migrated = []
    for service, (field, env_name) in FIELDS.items():
        if provider is not None and service != provider:
            continue
        legacy = LEGACY_FILES.get(field)
        if legacy is None or not legacy.exists():
            continue
        old_value = read_key_file(legacy, [env_name], recover, required=True).get(env_name)
        if old_value is None:
            if recover:
                migrated.append(legacy)
            continue
        if field not in data:
            data[field] = old_value.strip()
        migrated.append(legacy)
    if migrated:
        write_keys(data)
        if read_keys(API_KEYS_FILE) != data:
            raise OSError('API 키 이전 저장 확인에 실패했습니다.')
        for legacy in migrated:
            legacy.unlink()
    return data


def load_api_key(provider='luna'):
    field, env_name = FIELDS[provider]
    data = read_keys(API_KEYS_FILE)
    if field not in data:
        data = migrate_keys(provider=provider)
    value = data.get(field, os.environ.get(env_name, ''))
    if not isinstance(value, str):
        raise ValueError('API 키는 문자열이어야 합니다.')
    return value.strip()


def load_openai_settings():
    data = read_settings(API_KEYS_FILE.with_name('settings.json'))
    result = {}
    for name, default in OPENAI_DEFAULTS.items():
        value = data.get('openai_' + name, default)
        if not isinstance(value, str):
            raise ValueError('OpenAI 호환 API 주소와 모델은 문자열이어야 합니다.')
        result[name] = value.strip()
    return result


def load_translation_provider():
    data = read_settings(API_KEYS_FILE.with_name('settings.json'))
    provider = data.get('translation_provider', 'luna')
    return provider if isinstance(provider, str) and provider in FIELDS else 'luna'


def save_api_keys(kie_key, deepl_key, *, openai_key=None, openai_base_url=None, openai_model=None,
                  translation_provider=None):
    if translation_provider is not None and translation_provider not in FIELDS:
        raise ValueError('지원하지 않는 번역 서비스입니다.')
    data = migrate_keys(recover=True)
    data.update(kie_api_key=kie_key.strip(), deepl_api_key=deepl_key.strip())
    if openai_key is not None:
        data['openai_api_key'] = openai_key.strip()
    preferences = {}
    for name, value in [('openai_base_url', openai_base_url), ('openai_model', openai_model),
                        ('translation_provider', translation_provider)]:
        if value is not None:
            preferences[name] = value.strip()
    if preferences:
        update_settings(preferences, API_KEYS_FILE.with_name('settings.json'))
    write_keys({name: value for name, value in data.items()
                if name not in ('openai_base_url', 'openai_model', 'translation_provider')})
