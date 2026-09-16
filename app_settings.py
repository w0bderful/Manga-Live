import json
from pathlib import Path
import threading

SETTINGS_FILE = Path(__file__).resolve().parent / 'settings.json'
SOURCE_LANGUAGES = {'auto': '자동 언어 감지', 'ja': '일본어', 'en': '영어'}
DEFAULT_SOURCE_LANGUAGE = 'ja'
DETECTION_METHODS = {'comic': 'Comic Text Detector', 'opencv': 'OpenCV (기존 방식)'}
DEFAULT_DETECTION_METHOD = 'comic'
UI_DEFAULTS = {
    'monitor': '', 'interface_mode': 'basic', 'always_on_top': True,
    'device': 'cuda', 'detection_size': None,
    'single_balloon': False, 'detection_method': DEFAULT_DETECTION_METHOD,
}
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


def validate_ui_settings(value):
    if not isinstance(value, dict):
        raise ValueError('화면 설정은 JSON 객체여야 합니다.')
    result = {**UI_DEFAULTS, **value}
    choices = {'interface_mode': ('basic', 'advanced'), 'device': ('cpu', 'cuda'),
               'detection_method': tuple(DETECTION_METHODS)}
    for key, options in choices.items():
        if not isinstance(result[key], str) or result[key] not in options:
            raise ValueError(f'잘못된 화면 설정: {key}')
    if not isinstance(result['monitor'], str):
        raise ValueError('잘못된 모니터 설정입니다.')
    for key in ('always_on_top', 'single_balloon'):
        if type(result[key]) is not bool:
            raise ValueError(f'잘못된 화면 설정: {key}')
    size = result['detection_size']
    if size is not None and (type(size) is not int or size not in (960, 1280, 1920)):
        raise ValueError('잘못된 감지 해상도입니다.')
    return {key: result[key] for key in UI_DEFAULTS}


def load_ui_settings(path=None):
    return validate_ui_settings(read_settings(path).get('ui', {}))


def save_ui_settings(value, path=None):
    update_settings({'ui': validate_ui_settings(value)}, path)
