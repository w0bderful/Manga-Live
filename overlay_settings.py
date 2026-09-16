from app_settings import SETTINGS_FILE, read_settings, update_settings
DEFAULT_OPACITY = 100


def validate_opacity(value):
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError('배경 불투명도는 0~100 사이의 정수여야 합니다.')
    return value


def load_opacity(path=SETTINGS_FILE):
    data = read_settings(path)
    return validate_opacity(data.get('background_opacity', DEFAULT_OPACITY))


def save_opacity(value, path=SETTINGS_FILE):
    update_settings({'background_opacity': validate_opacity(value)}, path)
