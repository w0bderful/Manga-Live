import json
from pathlib import Path
import threading

SETTINGS_FILE = Path(__file__).resolve().parent / 'settings.json'
SOURCE_LANGUAGES = {'auto': '자동 언어 감지', 'ja': '일본어', 'en': '영어'}
DEFAULT_SOURCE_LANGUAGE = 'ja'
_lock = threading.RLock()


def read_settings(path=None):
    path = Path(path) if path is not None else SETTINGS_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('설정 파일은 JSON 객체여야 합니다.')
    return data


def write_settings(data, path):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def update_settings(changes, path=None):
    path = Path(path) if path is not None else SETTINGS_FILE
    with _lock:
        data = read_settings(path)
        data.update(changes)
        write_settings(data, path)


def validate_source_language(value):
    if not isinstance(value, str) or value not in SOURCE_LANGUAGES:
        raise ValueError('원문 언어는 자동 언어 감지·일본어·영어 중에서 선택하세요.')
    return value


def load_source_language():
    return validate_source_language(read_settings().get('source_language', DEFAULT_SOURCE_LANGUAGE))


def save_source_language(value):
    update_settings({'source_language': validate_source_language(value)})
