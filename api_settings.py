import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API_KEYS_FILE = ROOT / 'api-keys.json'
FIELDS = {'luna': ('kie_api_key', 'KIE_API_KEY'), 'deepl': ('deepl_api_key', 'DEEPL_API_KEY')}
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


def migrate_keys():
    data = read_keys(API_KEYS_FILE)
    migrated = []
    for field, env_name in FIELDS.values():
        legacy = LEGACY_FILES[field]
        if not legacy.exists():
            continue
        old_value = read_keys(legacy).get(env_name)
        if not isinstance(old_value, str):
            raise ValueError('기존 API 키 파일의 키 형식이 잘못되었습니다.')
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
    data = migrate_keys()
    value = data.get(field, os.environ.get(env_name, ''))
    if not isinstance(value, str):
        raise ValueError('API 키는 문자열이어야 합니다.')
    return value.strip()


def save_api_keys(kie_key, deepl_key):
    data = migrate_keys()
    data.update(kie_api_key=kie_key.strip(), deepl_api_key=deepl_key.strip())
    write_keys(data)
