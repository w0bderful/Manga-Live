from app_settings import read_settings, update_settings

DEFAULT_TEXT_STYLE = {'font_family': 'Malgun Gothic', 'font_size': 23, 'background_opacity': 30}


def validate_text_style(value):
    if not isinstance(value, dict):
        raise ValueError('글자 표시 설정은 JSON 객체여야 합니다.')
    result = {name: value.get(name, default) for name, default in DEFAULT_TEXT_STYLE.items()}
    family = result['font_family']
    if not isinstance(family, str) or not family.strip() or len(family) > 100 or any(ord(c) < 32 for c in family):
        raise ValueError('올바른 글꼴 이름을 선택하세요.')
    result['font_family'] = family.strip()
    for name, low, high in [('font_size', 8, 72), ('background_opacity', 0, 100)]:
        if type(result[name]) is not int or not low <= result[name] <= high:
            raise ValueError('글자 크기는 8~72px, 배경 불투명도는 0~100% 범위여야 합니다.')
    return result


def load_text_style(path=None):
    return validate_text_style(read_settings(path).get('text_style', {}))


def save_text_style(value, path=None):
    update_settings({'text_style': validate_text_style(value)}, path)
