import json
from pathlib import Path

SETTINGS_FILE = Path(__file__).resolve().parent / 'overlay-settings.json'
DEFAULT_OPACITY = 100


def validate_opacity(value):
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError('배경 불투명도는 0~100 사이의 정수여야 합니다.')
    return value


def load_opacity(path=SETTINGS_FILE):
    if not path.exists():
        return DEFAULT_OPACITY
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('배경 설정은 JSON 객체여야 합니다.')
    return validate_opacity(data.get('background_opacity', DEFAULT_OPACITY))


def save_opacity(value, path=SETTINGS_FILE):
    data = {'background_opacity': validate_opacity(value)}
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
