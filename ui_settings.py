import json
from pathlib import Path

SETTINGS_FILE = Path(__file__).resolve().parent / 'ui-settings.json'
MENU_ITEMS = {
    'basic': '기본 모드', 'advanced': '고급 모드', 'window': '설정 창 따로 열기',
    'monitor': '모니터', 'provider': '번역 API', 'device': 'OCR 처리 장치',
    'ocr': 'OCR 방식', 'translation': '번역 모드', 'resolution': '감지 해상도',
    'hotkeys': '단축키 설정', 'apply': '설정 적용',
}


def load_menu_items():
    if not SETTINGS_FILE.exists():
        return []
    data = json.loads(SETTINGS_FILE.read_text(encoding='utf-8'))
    items = data.get('menu_items') if isinstance(data, dict) else None
    if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
        raise ValueError('상단 메뉴 설정 형식이 올바르지 않습니다.')
    return [key for key in MENU_ITEMS if key in items]


def save_menu_items(items):
    if any(item not in MENU_ITEMS for item in items):
        raise ValueError('지원하지 않는 메뉴 항목입니다.')
    data = {'menu_items': [key for key in MENU_ITEMS if key in items]}
    temporary = SETTINGS_FILE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(SETTINGS_FILE)
