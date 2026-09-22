from json_storage import read_object, write_object
from pathlib import Path
import threading

from runtime_paths import CONFIG_DIR
SETTINGS_FILE = CONFIG_DIR / 'settings.json'
SOURCE_LANGUAGES = {'auto': '자동 언어 감지', 'ja': '일본어', 'en': '영어'}
DEFAULT_SOURCE_LANGUAGE = 'ja'
DETECTION_METHODS = {'comic': 'Comic Text Detector', 'opencv': 'OpenCV (기존 방식)'}
DEFAULT_DETECTION_METHOD = 'comic'
UI_DEFAULTS = {
    'monitor': '', 'interface_mode': 'basic', 'always_on_top': True,
    'device': 'cuda', 'detection_size': None,
    'single_balloon': False, 'detection_method': DEFAULT_DETECTION_METHOD,
    'window_sizes': {},
}
_lock = threading.RLock()


def read_settings(path=None):
    return read_object(Path(path) if path is not None else SETTINGS_FILE, '설정 파일은 JSON 객체여야 합니다.')


def write_settings(data, path):
    write_object(path, data)


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
    window_sizes = result['window_sizes']
    if not isinstance(window_sizes, dict):
        raise ValueError('잘못된 창 크기 설정입니다.')
    result['window_sizes'] = {}
    for mode in ('basic', 'advanced'):
        dimensions = window_sizes.get(mode)
        if dimensions is None:
            continue
        if (not isinstance(dimensions, (list, tuple)) or len(dimensions) != 2
                or any(type(n) is not int or not 100 <= n <= 32768 for n in dimensions)):
            raise ValueError('잘못된 창 크기 설정입니다.')
        result['window_sizes'][mode] = list(dimensions)
    return {key: result[key] for key in UI_DEFAULTS}


def load_ui_settings(path=None):
    return validate_ui_settings(read_settings(path).get('ui', {}))


def save_ui_settings(value, path=None):
    update_settings({'ui': validate_ui_settings(value)}, path)
