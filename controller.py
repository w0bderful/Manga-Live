"""Main-window controls and coordination of independent application services."""
from background_tasks import submit_background
import logging
from runtime_paths import APP_DIR as ROOT
import time
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QAction, QActionGroup
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QLineEdit,
    QFormLayout, QMenuBar, QDialog, QScrollArea,
    QGridLayout, QMessageBox, QFontComboBox, QSpinBox,
    QProgressBar,
)
from core import changed, relocate, merge_row, scroll_offset, move_rows
from translation import (
    TRANSLATION_MODES, validate_openai_settings, list_openai_models, get_deepl_usage,
)
from api_settings import (
    load_api_key, save_api_keys, load_openai_settings, OPENAI_DEFAULTS,
    load_translation_provider,
)
from window_capture import CaptureWithoutApp, CaptureProtectionError
from hotkeys import ACTIONS, HotkeyDialog, WindowsHotkeys, load_settings
from overlay_settings import load_text_style, save_text_style
from window_theme import (
    SakuraBackdrop, apply_window_theme, WINDOW_THEMES, DEFAULT_WINDOW_THEME,
    load_window_theme, save_window_theme,
)
from resource_usage import ResourceMonitor
from update_ui import VersionUpdater
from app_settings import (
    SOURCE_LANGUAGES, DEFAULT_SOURCE_LANGUAGE, load_source_language, save_source_language,
    UI_DEFAULTS, load_ui_settings, save_ui_settings, DETECTION_METHODS,
)
from engine import Engine, Signals
from overlay import Overlay
from selection import native_region, RegionIndicator, Selector
from model_combo import ModelComboBox
log = logging.getLogger(__name__)


def request_model_list(base_url, api_key):
    return submit_background(list_openai_models, base_url, api_key, name='model-list')


def request_deepl_usage(api_key):
    return submit_background(get_deepl_usage, api_key, name='deepl-usage')


class Controller(QWidget):
    def __init__(self, io_logger=None):
        super().__init__()
        self.restoring_settings = True
        ui_error = ''
        try:
            initial_ui = load_ui_settings()
        except (OSError, ValueError):
            initial_ui = dict(UI_DEFAULTS)
            ui_error = '화면 설정을 읽지 못해 기본값을 사용합니다. settings.json을 확인하세요.'
        self.io_logger = io_logger
        self.setWindowTitle('Manga Live · 화면 → 한국어')
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, initial_ui['always_on_top'])
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setAutoFillBackground(True)
        self.resize(510, 230)
        self.sakura_background = SakuraBackdrop(self)
        theme_error = ''
        try:
            initial_theme = load_window_theme()
        except (OSError,ValueError):
            initial_theme = DEFAULT_WINDOW_THEME
            theme_error = '창 배경 설정을 읽지 못해 기본값을 사용합니다. settings.json을 확인하세요.'
        self.overlay = Overlay()
        style_error = ''
        try:
            self.overlay.text_style = load_text_style()
        except (OSError, ValueError):
            style_error = '글자 표시 설정을 읽지 못해 기본값을 사용합니다. settings.json을 확인하세요.'
        self.region_indicator = RegionIndicator()
        self.capture = CaptureWithoutApp()
        self.region = None
        self.reference = None
        self.latest = None
        self.last_change = 0
        self.submitted = False
        self.running = False
        self.single_shot = False
        self.worker_ready = False
        self.result_floor = 0
        self.signals = Signals()
        self.api_settings_error = False
        initial_api_key = self.load_initial_api_key('luna')
        initial_deepl_api_key = self.load_initial_api_key('deepl')
        initial_openai_api_key = self.load_initial_api_key('openai')
        try:
            initial_openai_settings = load_openai_settings()
        except (OSError, ValueError):
            self.api_settings_error = True
            initial_openai_settings = dict(OPENAI_DEFAULTS)
        try:
            initial_provider = load_translation_provider()
        except (OSError, ValueError):
            self.api_settings_error = True
            initial_provider = 'luna'
        initial_keys = {'luna': initial_api_key, 'deepl': initial_deepl_api_key,
                        'openai': initial_openai_api_key}
        language_error = ''
        try:
            initial_language = load_source_language()
        except (OSError, ValueError):
            initial_language = DEFAULT_SOURCE_LANGUAGE
            language_error = '원문 언어 설정을 읽지 못해 일본어를 사용합니다. settings.json을 확인하세요.'
        self.engine = Engine(self.signals, device=initial_ui['device'],
                             detection_method=initial_ui['detection_method'],
                             api_key=initial_keys[initial_provider],
                             provider=initial_provider, base_url=initial_openai_settings['base_url'],
                             model=initial_openai_settings['model'], io_logger=self.io_logger,
                             source_language=initial_language)
        self.build_layout(initial_theme, initial_ui)
        self.build_device_controls(initial_ui, theme_error)
        self.build_translation_controls(initial_keys, initial_provider, initial_openai_settings, initial_language, language_error)
        self.build_action_buttons()
        self.build_ocr_controls(initial_ui)
        self.build_text_style_controls(style_error)
        self.build_progress_widgets(initial_ui)
        self.connect_engine()
        self.timer = QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self.log_cleanup_timer = QTimer(self)
        self.log_cleanup_timer.setInterval(3_600_000)
        if self.io_logger is not None:
            self.log_cleanup_timer.timeout.connect(self.io_logger.cleanup)
            self.log_cleanup_timer.start()
        self.hotkey_dialog_open = False
        self.hotkeys = WindowsHotkeys(self.activate_hotkey)
        try:
            self.hotkey_settings = load_settings()
            hotkey_errors = []
        except (OSError, ValueError) as exc:
            self.hotkey_settings = {action: '' for action in ACTIONS}
            hotkey_errors = [f'저장된 단축키를 읽지 못했습니다. 단축키 설정에서 다시 저장하세요: {exc}']
        hotkey_errors.extend(self.hotkeys.apply(self.hotkey_settings))
        self.hotkey_note = QLabel()
        self.hotkey_note.setWordWrap(True)
        self.settings_layout.addWidget(self.hotkey_note)
        self.update_hotkey_note(hotkey_errors)
        self.main_layout.addWidget(self.theme_note)
        self.ui_settings_note = QLabel(ui_error)
        self.ui_settings_note.setWordWrap(True)
        self.ui_settings_note.setVisible(bool(ui_error))
        self.main_layout.addWidget(self.ui_settings_note)
        self.resource_note = QLabel('CPU —  ·  GPU —  ·  VRAM —')
        self.resource_note.setWordWrap(True)
        self.resource_note.setToolTip('현재 프로그램의 사용량입니다. CPU는 전체 논리 코어 기준, GPU는 가장 바쁜 엔진 기준, VRAM은 전용 GPU 메모리입니다. 조회 불가는 드라이버가 정보를 제공하지 않는 경우입니다.')
        self.main_layout.addWidget(self.resource_note)
        self.version_updates = VersionUpdater(self, ROOT)
        self.version_updates.progress_changed.connect(self.update_release_progress)
        self.main_layout.addWidget(self.version_updates.panel)
        self.resource_monitor = ResourceMonitor()
        self.resource_monitor.start()
        self.settings_layout.addStretch()
        self.settings_dialog = QDialog(self)
        self.settings_background = SakuraBackdrop(self.settings_dialog)
        self.settings_dialog.setWindowTitle('Manga Live 설정')
        self.settings_dialog.finished.connect(self.restore_inline_settings)
        dialog_layout = QVBoxLayout(self.settings_dialog)
        self.dialog_settings = QScrollArea()
        self.dialog_settings.setWidgetResizable(True)
        self.dialog_settings.viewport().setAutoFillBackground(False)
        dialog_layout.addWidget(self.dialog_settings)
        close_settings = QPushButton('닫기')
        close_settings.clicked.connect(self.settings_dialog.close)
        dialog_layout.addWidget(close_settings)
        self.interface_mode = None
        self.apply_selected_window_theme()
        self.set_interface_mode(initial_ui['interface_mode'])
        self.restoring_settings = False
        for control in (self.screens, self.device_mode, self.detection_mode, self.detection_method):
            control.currentIndexChanged.connect(self.save_current_ui_settings)
        for control in (self.manual, self.always_on_top):
            control.toggled.connect(self.save_current_ui_settings)
        self.engine.start()

        self.version_updates.start()


    def add_setting(self, label, control, row, column, span=1):
        cell = QWidget()
        cell_layout = QVBoxLayout(cell)
        cell_layout.setContentsMargins(0,0,0,0)
        cell_layout.setSpacing(4)
        cell_layout.addWidget(QLabel(label))
        cell_layout.addWidget(control)
        control.setMinimumContentsLength(10)
        control.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        for index in range(control.count()):
            if not control.itemData(index,Qt.ItemDataRole.ToolTipRole):
                control.setItemData(index,control.itemText(index),Qt.ItemDataRole.ToolTipRole)
        self.settings_grid.addWidget(cell,row,column,1,span)


    def build_layout(self, initial_theme, initial_ui):
        self.main_layout = QVBoxLayout(self)
        self.menu_bar = QMenuBar(self)
        self.menu_bar.setNativeMenuBar(False)
        self.main_layout.setMenuBar(self.menu_bar)
        self.open_settings_action = QAction('설정창 열기', self)
        self.addAction(self.open_settings_action)
        self.open_settings_action.setShortcut('Ctrl+,')
        self.open_settings_action.triggered.connect(self.open_settings)
        self.interface_modes = QActionGroup(self)
        self.interface_modes.setExclusive(True)
        self.basic_mode_action = QAction('기본 모드', self, checkable=True)
        self.advanced_mode_action = QAction('고급 모드', self, checkable=True)
        for action, mode in [(self.basic_mode_action, 'basic'), (self.advanced_mode_action, 'advanced')]:
            self.interface_modes.addAction(action)
            self.menu_bar.addAction(action)
            action.triggered.connect(lambda checked, selected=mode: self.set_interface_mode(selected))
        self.selected_window_theme = initial_theme
        self.theme_menu = self.menu_bar.addMenu('창 배경')
        self.theme_actions = QActionGroup(self)
        self.theme_actions.setExclusive(True)
        self.window_theme_actions = {}
        for key, label in WINDOW_THEMES.items():
            action = self.theme_menu.addAction(label)
            action.setCheckable(True)
            action.setData(key)
            action.setChecked(key == initial_theme)
            self.theme_actions.addAction(action)
            self.window_theme_actions[key] = action
        self.theme_actions.triggered.connect(lambda action: self.change_window_theme(action.data()))
        self.always_on_top = QCheckBox('최상단 고정')
        self.always_on_top.setToolTip('프로그램 창을 다른 앱보다 위에 표시합니다.')
        self.always_on_top.setChecked(initial_ui['always_on_top'])
        self.always_on_top.toggled.connect(self.set_always_on_top)
        self.menu_bar.setCornerWidget(self.always_on_top, Qt.Corner.TopRightCorner)
        self.inline_settings = QScrollArea()
        self.inline_settings.setWidgetResizable(True)
        self.inline_settings.viewport().setAutoFillBackground(False)
        self.main_layout.addWidget(self.inline_settings, 1)
        self.settings_panel = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_panel)
        self.settings_layout.setContentsMargins(8, 8, 8, 8)
        self.settings_grid = QGridLayout()
        self.settings_grid.setHorizontalSpacing(12)
        self.settings_grid.setVerticalSpacing(10)
        self.settings_grid.setColumnStretch(0,1)
        self.settings_grid.setColumnStretch(1,1)
        self.settings_layout.addLayout(self.settings_grid)

    def build_device_controls(self, initial_ui, theme_error):
        self.screens = QComboBox()
        for screen in QApplication.screens():
            self.screens.addItem(screen.name(), screen)
        monitor_index = self.screens.findText(initial_ui['monitor'])
        if monitor_index >= 0:
            self.screens.setCurrentIndex(monitor_index)
        self.add_setting('모니터',self.screens,0,0)
        self.theme_note = QLabel(theme_error)
        self.theme_note.setWordWrap(True)
        self.theme_note.setVisible(bool(theme_error))
        self.device_mode = QComboBox()
        self.device_mode.addItem('CPU 모드', 'cpu')
        self.device_mode.addItem('GPU 모드 (NVIDIA CUDA)', 'cuda')
        self.device_mode.setCurrentIndex(self.device_mode.findData(self.engine.device))
        self.add_setting('OCR 처리 장치',self.device_mode,0,1)
        ocr_note = QLabel('Comic Text Detector와 글자 인식은 선택한 OCR 장치에서 실행됩니다. OpenCV 후보 감지·이미지 보정은 CPU를 사용합니다.')
        ocr_note.setWordWrap(True)
        self.settings_layout.addWidget(ocr_note)

    def build_translation_controls(self, initial_keys, initial_provider, initial_openai_settings, initial_language, language_error):
        self.translation_mode = QComboBox()
        for key, label in TRANSLATION_MODES.items():
            self.translation_mode.addItem(label, key)
        self.translation_mode.setCurrentIndex(self.translation_mode.findData(initial_provider))
        self.add_setting('번역 API',self.translation_mode,1,0)
        self.api_key = QLineEdit(initial_keys['luna'])
        self.api_key.setPlaceholderText('API 키 입력')
        self.api_key_save_timer = QTimer(self)
        self.api_key_save_timer.setSingleShot(True)
        self.api_key_save_timer.setInterval(500)
        self.api_key_save_timer.timeout.connect(self.change_device)
        self.source_language = QComboBox()
        self.source_language.setToolTip('자동 언어 판별은 EasyOCR를 사용합니다. 일본어는 Manga OCR, 영어는 글씨 형태에 따라 EasyOCR 또는 TrOCR를 먼저 사용하고 불확실하면 다른 인식기로 보완합니다.')
        for code, label in SOURCE_LANGUAGES.items():
            self.source_language.addItem(label, code)
        self.source_language.setCurrentIndex(self.source_language.findData(initial_language))
        self.add_setting('원문 언어',self.source_language,1,1)
        self.language_note = QLabel(language_error)
        self.language_note.setWordWrap(True)
        self.language_note.setVisible(bool(language_error))
        self.settings_layout.addWidget(self.language_note)
        self.source_language.currentIndexChanged.connect(self.change_source_language)
        self.device_mode.currentIndexChanged.connect(lambda: self.api_key_save_timer.start())
        self.api_key.textChanged.connect(lambda: self.api_key_save_timer.start())
        self.settings_layout.addWidget(self.api_key)
        self.api_key.setVisible(initial_provider == 'luna')
        self.deepl_api_key = QLineEdit(initial_keys['deepl'])
        self.deepl_api_key.setPlaceholderText('API 키 입력')
        self.deepl_api_key.textChanged.connect(lambda: self.api_key_save_timer.start())
        self.settings_layout.addWidget(self.deepl_api_key)
        self.deepl_api_key.setVisible(initial_provider == 'deepl')
        self.deepl_usage_panel = QWidget()
        usage_layout = QVBoxLayout(self.deepl_usage_panel)
        usage_layout.setContentsMargins(0, 0, 0, 0)
        self.deepl_usage_note = QLabel('DeepL API 키를 입력하세요.')
        self.deepl_usage_note.setWordWrap(True)
        usage_layout.addWidget(self.deepl_usage_note)
        self.main_layout.addWidget(self.deepl_usage_panel)
        self.deepl_usage_panel.setVisible(initial_provider == 'deepl')
        self.deepl_usage_future = None
        self.deepl_usage_generation = 0
        self.deepl_usage_closed = False
        self.deepl_usage_debounce = QTimer(self)
        self.deepl_usage_debounce.setSingleShot(True)
        self.deepl_usage_debounce.setInterval(800)
        self.deepl_usage_debounce.timeout.connect(self.load_deepl_usage)
        self.deepl_usage_poll = QTimer(self)
        self.deepl_usage_poll.setInterval(100)
        self.deepl_usage_poll.timeout.connect(self.finish_deepl_usage)
        self.deepl_usage_refresh = QTimer(self)
        self.deepl_usage_refresh.setInterval(300_000)
        self.deepl_usage_refresh.timeout.connect(self.load_deepl_usage)
        self.deepl_api_key.textChanged.connect(self.invalidate_deepl_usage)
        self.invalidate_deepl_usage()
        self.openai_panel = QWidget()
        openai_form = QFormLayout(self.openai_panel)
        openai_form.setContentsMargins(0, 0, 0, 0)
        self.openai_base_url = QLineEdit(initial_openai_settings['base_url'])
        self.openai_base_url.setPlaceholderText('https://서버주소/v1 또는 전체 /chat/completions 주소')
        self.openai_api_key = QLineEdit(initial_keys['openai'])
        self.openai_api_key.setPlaceholderText('API 키 입력')
        for label, field in [('API 주소', self.openai_base_url), ('API 키', self.openai_api_key)]:
            openai_form.addRow(label, field)
            field.textChanged.connect(lambda: self.api_key_save_timer.start())
        self.openai_model = ModelComboBox()
        self.openai_model.setPlaceholderText('눌러서 모델 선택')
        self.openai_model.setMinimumContentsLength(12)
        self.openai_model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        if initial_openai_settings['model']:
            self.openai_model.addItem(initial_openai_settings['model'], initial_openai_settings['model'])
            self.openai_model.setCurrentIndex(0)
        self.openai_model.currentIndexChanged.connect(lambda: self.api_key_save_timer.start())
        self.openai_model.requested.connect(self.load_models)
        openai_form.addRow('모델 선택', self.openai_model)
        self.models_note = QLabel('저장된 모델을 복원했습니다. 목록을 새로 불러올 수 있습니다.'
                                 if initial_openai_settings['model'] else '주소와 키를 입력한 뒤 모델 선택을 누르세요.')
        self.models_note.setWordWrap(True)
        openai_form.addRow(self.models_note)
        self.models_future = None
        self.models_generation = 0
        self.models_timer = QTimer(self)
        self.models_timer.setInterval(100)
        self.models_timer.timeout.connect(self.finish_model_list)
        self.openai_base_url.textChanged.connect(self.invalidate_model_list)
        self.openai_api_key.textChanged.connect(self.invalidate_model_list)
        self.settings_layout.addWidget(self.openai_panel)
        self.openai_panel.setVisible(initial_provider == 'openai')
        self.api_settings_note = QLabel('API 키 파일을 읽지 못했습니다. 키를 다시 입력하면 자동 저장·적용됩니다. '
                                       '손상된 원본은 저장 시 api-key-backups 폴더에 보관합니다.'
                                       if self.api_settings_error else '')
        self.api_settings_note.setWordWrap(True)
        self.settings_layout.addWidget(self.api_settings_note)
        self.translation_mode.currentIndexChanged.connect(self.update_translation_fields)

    def build_action_buttons(self):
        self.action_panel = QWidget()
        row = QGridLayout(self.action_panel)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self.select_button = QPushButton('영역 선택')
        self.select_button.clicked.connect(self.select_region)
        row.addWidget(self.select_button, 0, 0)
        self.drag_button = QPushButton('드래그 번역')
        self.drag_button.clicked.connect(lambda: self.select_region(single_shot=True))
        row.addWidget(self.drag_button, 0, 1)
        self.toggle_button = QPushButton('번역 시작')
        self.toggle_button.clicked.connect(self.toggle)
        row.addWidget(self.toggle_button, 1, 0)
        self.retry_button = QPushButton('다시 번역')
        self.retry_button.clicked.connect(self.retry_translation)
        row.addWidget(self.retry_button, 1, 1)
        for button in (self.select_button, self.drag_button, self.toggle_button, self.retry_button):
            button.setMinimumHeight(60)
            font = button.font()
            font.setPointSize(14)
            font.setBold(True)
            button.setFont(font)
        self.main_layout.addWidget(self.action_panel)

    def build_ocr_controls(self, initial_ui):
        self.hotkey_button = QPushButton('단축키 설정')
        self.hotkey_button.clicked.connect(self.configure_hotkeys)
        self.settings_layout.addWidget(self.hotkey_button)
        self.manual = QCheckBox('선택 영역 전체가 말풍선 하나 (자동 감지 생략)')
        self.manual.setChecked(initial_ui['single_balloon'])
        self.manual.toggled.connect(self.reset_frame)
        self.settings_layout.addWidget(self.manual)
        self.detection_mode = QComboBox()
        self.detection_mode.addItem('원본 해상도 (기본 · 축소 없이 감지)', None)
        self.detection_mode.addItem('빠른 감지 (작은 글자는 놓칠 수 있음)', 960)
        self.detection_mode.addItem('균형 감지', 1280)
        self.detection_mode.addItem('정밀 감지 (작은 글씨 · 느림)', 1920)
        self.detection_mode.setCurrentIndex(self.detection_mode.findData(initial_ui['detection_size']))
        self.engine.detector_size = initial_ui['detection_size']
        self.detection_mode.currentIndexChanged.connect(self.change_detection_mode)
        self.detection_method = QComboBox()
        for key, label in DETECTION_METHODS.items():
            self.detection_method.addItem(label, key)
        self.detection_method.setCurrentIndex(self.detection_method.findData(initial_ui['detection_method']))
        self.detection_method.setToolTip('Comic Text Detector는 만화 텍스트 블록을 감지합니다. OpenCV (기존 방식)를 선택하면 이전 감지 방식으로 돌아갑니다.')
        self.detection_method.currentIndexChanged.connect(lambda: self.api_key_save_timer.start())
        self.add_setting('영역 감지', self.detection_method, 2, 0)
        self.add_setting('감지 해상도',self.detection_mode,2,1)

    def build_text_style_controls(self, style_error):
        self.text_style_panel = QWidget()
        style_form = QFormLayout(self.text_style_panel)
        style_form.setContentsMargins(0,0,0,0)
        self.translation_font = QFontComboBox()
        self.translation_font.setCurrentFont(QFont(self.overlay.text_style['font_family']))
        style_form.addRow('번역 글꼴', self.translation_font)
        self.translation_font_size = QSpinBox()
        self.translation_font_size.setRange(8,72)
        self.translation_font_size.setSuffix(' px')
        self.translation_font_size.setValue(self.overlay.text_style['font_size'])
        self.translation_font_size.setToolTip('설정한 크기를 기준으로 표시하며, 영역에 들어가지 않으면 자동으로 줄입니다.')
        style_form.addRow('글자 크기', self.translation_font_size)
        self.text_background_opacity = QSpinBox()
        self.text_background_opacity.setRange(0,100)
        self.text_background_opacity.setSuffix(' %')
        self.text_background_opacity.setValue(self.overlay.text_style['background_opacity'])
        self.text_background_opacity.setToolTip('흰색 배경: 0%는 없음, 100%는 완전 불투명. 글자는 항상 선명하게 표시합니다.')
        style_form.addRow('배경 불투명도', self.text_background_opacity)
        self.text_style_note = QLabel(style_error)
        self.text_style_note.setWordWrap(True)
        self.text_style_note.setVisible(bool(style_error))
        style_form.addRow(self.text_style_note)
        self.translation_font.currentFontChanged.connect(self.change_text_style)
        self.translation_font_size.valueChanged.connect(self.change_text_style)
        self.text_background_opacity.valueChanged.connect(self.change_text_style)
        self.settings_layout.addWidget(self.text_style_panel)

    def build_progress_widgets(self, initial_ui):
        self.status = QLabel('준비 중…')
        self.status.setWordWrap(True)
        self.main_layout.addWidget(self.status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 5 if initial_ui['detection_method'] == 'comic' else 4)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('OCR 로딩 대기')
        self.task_progress = (0, self.progress_bar.maximum(), 'OCR 로딩 대기')
        self.release_progress = None
        self.progress_bar.setMinimumHeight(22)
        self.main_layout.addWidget(self.progress_bar)
        self.capture_note = QLabel('')
        self.capture_note.setWordWrap(True)
        self.main_layout.addWidget(self.capture_note)
        note = QLabel('말풍선별로 번역이 끝나는 즉시 표시합니다. 스크롤하면 번역 위치를 추적합니다.\n인식한 대사를 선택한 번역 서비스로 전송합니다 (이용 요금 발생 가능). 이미지는 전송하지 않습니다. 종료하려면 이 창을 닫으세요.')
        note.setWordWrap(True)
        self.settings_layout.addWidget(note)

    def save_current_ui_settings(self, *_):
        if self.restoring_settings:
            return
        settings = {
            'monitor': self.screens.currentText(), 'interface_mode': self.interface_mode,
            'always_on_top': self.always_on_top.isChecked(), 'device': self.device_mode.currentData(),
            'detection_size': self.detection_mode.currentData(),
            'detection_method': self.detection_method.currentData(),
            'single_balloon': self.manual.isChecked(),
        }
        try:
            save_ui_settings(settings)
        except (OSError, ValueError):
            self.ui_settings_note.setText('화면 설정을 저장하지 못했습니다. settings.json의 상태와 권한을 확인하세요.')
            self.ui_settings_note.show()
        else:
            self.ui_settings_note.clear()
            self.ui_settings_note.hide()

    def apply_selected_window_theme(self):
        theme = self.selected_window_theme
        # Child dialog styles must be cleared before changing the parent style.
        self.settings_dialog.setStyleSheet('')
        apply_window_theme(self,self.sakura_background,theme)
        apply_window_theme(self.settings_dialog,self.settings_background,theme)

    def change_window_theme(self, theme):
        self.selected_window_theme = theme
        self.apply_selected_window_theme()
        try:
            save_window_theme(theme)
        except (OSError,ValueError):
            self.theme_note.setText('창 배경을 저장하지 못했습니다. settings.json의 상태와 권한을 확인하세요.')
            self.theme_note.show()
        else:
            self.theme_note.clear()
            self.theme_note.hide()

    def set_interface_mode(self, mode):
        if mode == self.interface_mode:
            return
        self.settings_dialog.hide()
        for scroll in (self.inline_settings, self.dialog_settings):
            if scroll.widget() is self.settings_panel:
                scroll.takeWidget()
        advanced = mode == 'advanced'
        settings_layout = self.settings_panel.layout()
        self.main_layout.removeWidget(self.deepl_usage_panel)
        settings_layout.removeWidget(self.deepl_usage_panel)
        if advanced:
            settings_layout.insertWidget(settings_layout.indexOf(self.deepl_api_key) + 1,
                                         self.deepl_usage_panel)
        else:
            self.main_layout.insertWidget(self.main_layout.indexOf(self.action_panel),
                                          self.deepl_usage_panel)
        self.deepl_usage_panel.setVisible(self.translation_mode.currentData() == 'deepl')
        target = self.inline_settings if advanced else self.dialog_settings
        target.setWidget(self.settings_panel)
        self.settings_panel.setAutoFillBackground(False)
        self.settings_panel.show()
        self.inline_settings.setVisible(advanced)
        self.capture_note.setVisible(advanced)
        self.text_style_panel.setVisible(advanced)
        self.interface_mode = mode
        self.basic_mode_action.setChecked(not advanced)
        self.advanced_mode_action.setChecked(advanced)
        self.main_layout.activate()
        available = self.screen().availableGeometry()
        self.resize(560 if advanced else 510, min(760, available.height() - 80) if advanced else self.minimumSizeHint().height())
        self.save_current_ui_settings()

    def change_text_style(self, *_):
        style = {'font_family': self.translation_font.currentFont().family(),
                 'font_size': self.translation_font_size.value(),
                 'background_opacity': self.text_background_opacity.value()}
        self.overlay.text_style = style
        if self.overlay.rows:
            self.overlay.display(self.overlay.rows, self.overlay.source_size)
        try:
            save_text_style(style)
        except (OSError, ValueError):
            self.text_style_note.setText('글자 표시 설정을 저장하지 못했습니다. settings.json의 상태와 권한을 확인하세요.')
            self.text_style_note.show()
        else:
            self.text_style_note.clear()
            self.text_style_note.hide()

    def open_settings(self):
        if self.inline_settings.widget() is self.settings_panel:
            self.inline_settings.takeWidget()
            self.dialog_settings.setWidget(self.settings_panel)
            self.settings_panel.setAutoFillBackground(False)
            self.inline_settings.hide()
            self.settings_panel.show()
            self.main_layout.activate()
            self.resize(self.width(), self.minimumSizeHint().height())
        available = self.screen().availableGeometry()
        self.settings_dialog.resize(560, min(760, available.height() - 80))
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def restore_inline_settings(self):
        if self.interface_mode == 'advanced' and self.dialog_settings.widget() is self.settings_panel:
            self.dialog_settings.takeWidget()
            self.inline_settings.setWidget(self.settings_panel)
            self.settings_panel.setAutoFillBackground(False)
            self.inline_settings.show()
            self.settings_panel.show()
            self.resize(560, min(760, self.screen().availableGeometry().height() - 80))

    def update_hotkey_note(self, errors=()):
        buttons = {'select': self.select_button, 'drag': self.drag_button,
                   'toggle': self.toggle_button, 'retry': self.retry_button}
        active = set(self.hotkeys.active.values())
        labels = []
        for action, button in buttons.items():
            key = self.hotkey_settings[action] if action in active else '미지정/비활성'
            button.setToolTip(f'{ACTIONS[action]}: {key}')
            labels.append(f'{ACTIONS[action]}: {key}')
        self.hotkey_note.setText('\n'.join(errors) if errors else ' · '.join(labels))

    def activate_hotkey(self, action):
        if self.hotkey_dialog_open or (hasattr(self, 'selector') and self.selector.isVisible()):
            return
        if hasattr(self, 'device_timer') and self.device_timer.isActive():
            self.status.setText('설정 적용이 끝난 뒤 단축키를 사용하세요.')
            return
        actions = {'select': self.select_region,
                   'drag': lambda: self.select_region(single_shot=True),
                   'toggle': self.toggle, 'retry': self.retry_translation}
        actions[action]()

    def configure_hotkeys(self):
        if self.hotkey_dialog_open:
            return
        self.hotkey_dialog_open = True
        self.hotkeys.clear()
        try:
            parent = self.settings_dialog if self.settings_dialog.isVisible() else self
            dialog = HotkeyDialog(parent, self.hotkeys, self.hotkey_settings)
            if dialog.exec():
                self.hotkey_settings = dialog.settings
                errors = []
            else:
                errors = self.hotkeys.apply(self.hotkey_settings)
            self.update_hotkey_note(errors)
        finally:
            self.hotkey_dialog_open = False

    def connect_engine(self):
        self.signals.api_key_required.connect(self.wait_for_api_key)
        self.signals.model_download.connect(self.update_model_download)
        self.signals.progress.connect(self.update_progress)
        self.signals.status.connect(self.status.setText)
        self.signals.failed.connect(self.processing_failed)
        self.signals.ready.connect(self.ready)
        self.signals.result.connect(self.accept_result)
        self.signals.finished.connect(self.frame_finished)

    def update_progress(self, engine, generation, done, total, label):
        if engine is not self.engine or engine.stop_event.is_set():
            return
        if generation != -1 and (not self.running or generation != engine.generation):
            return
        self.set_task_progress(done, total, f'{label} · %v/%m (%p%)' if total > 1 else label)

    def set_task_progress(self, done, total, label):
        self.task_progress = (done, total, label)
        self.render_progress()

    def update_release_progress(self, done, total, label):
        self.release_progress = (done, total, label) if label else None
        self.render_progress()

    def render_progress(self):
        done, total, label = self.release_progress or self.task_progress
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self.progress_bar.setFormat(label)

    def update_model_download(self, engine, info):
        if engine is not self.engine or engine.stop_event.is_set():
            return
        from model_downloads import download_label
        _, received, total = info
        label = download_label(info)
        self.set_task_progress(min(1000, int(received * 1000 / total)) if total else 0,
                               1000 if total else 0, label + (' (%p%)' if total else ''))
        self.status.setText(label)

    def stop_progress(self, label):
        self.set_task_progress(0, 1, label)

    def set_always_on_top(self, enabled):
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        if visible:
            self.show()

    def update_translation_fields(self):
        provider = self.translation_mode.currentData()
        self.api_key.setVisible(provider == 'luna')
        self.deepl_api_key.setVisible(provider == 'deepl')
        self.deepl_usage_panel.setVisible(provider == 'deepl')
        self.invalidate_deepl_usage()
        self.openai_panel.setVisible(provider == 'openai')
        self.worker_ready = False
        self.engine.stop_event.set()
        self.reset_for_settings()
        self.change_device()

    def reset_for_settings(self):
        resume_snapshot = self.single_shot and self.running
        snapshot = self.latest if resume_snapshot else None
        self.running = False
        self.reset_frame()
        self.running = resume_snapshot
        self.latest = snapshot
        self.reference = snapshot.copy() if snapshot is not None else None
        self.toggle_button.setText('취소' if resume_snapshot else '번역 시작')

    def invalidate_deepl_usage(self):
        self.deepl_usage_generation += 1
        self.deepl_usage_debounce.stop()
        self.deepl_usage_refresh.stop()
        has_key = bool(self.deepl_api_key.text().strip())
        self.deepl_usage_note.setText('남은 한도 확인 대기 중…' if has_key else 'DeepL API 키를 입력하세요.')
        if not self.deepl_usage_closed and has_key and self.translation_mode.currentData() == 'deepl':
            self.deepl_usage_debounce.start()
            self.deepl_usage_refresh.start()

    def load_deepl_usage(self):
        if (self.deepl_usage_closed or self.translation_mode.currentData() != 'deepl'
                or self.deepl_usage_future is not None or not self.deepl_api_key.text().strip()):
            return
        self.deepl_usage_debounce.stop()
        self.deepl_usage_request_generation = self.deepl_usage_generation
        self.deepl_usage_note.setText('남은 한도 조회 중…')
        try:
            self.deepl_usage_future = request_deepl_usage(self.deepl_api_key.text().strip())
        except Exception:
            self.deepl_usage_note.setText('사용량 조회를 시작하지 못했습니다. 다시 조회하세요.')
            return
        self.deepl_usage_poll.start()

    def finish_deepl_usage(self):
        if self.deepl_usage_future is None or not self.deepl_usage_future.done():
            return
        future, self.deepl_usage_future = self.deepl_usage_future, None
        self.deepl_usage_poll.stop()
        if self.deepl_usage_request_generation != self.deepl_usage_generation:
            if not self.deepl_usage_closed and self.translation_mode.currentData() == 'deepl':
                self.deepl_usage_debounce.start()
            return
        try:
            usage = future.result()
        except (ValueError, RuntimeError) as exc:
            self.deepl_usage_note.setText(str(exc))
            return
        except Exception:
            self.deepl_usage_note.setText('DeepL 사용량을 확인하지 못했습니다. 다시 조회하세요.')
            return
        remaining = ('남은 한도: 제한 없음' if usage['remaining'] is None else
                     f"남은 번역 가능 문자: {usage['remaining']:,}자")
        self.deepl_usage_note.setText(remaining)

    def load_initial_api_key(self, provider):
        try:
            return load_api_key(provider)
        except (OSError, ValueError):
            self.api_settings_error = True
            log.warning('Could not load API key settings for %s', provider)
            return ''

    def translation_is_current(self):
        return (self.worker_ready and not self.engine.stop_event.is_set()
                and self.engine.device == self.device_mode.currentData()
                and self.engine.detection_method == self.detection_method.currentData()
                and self.engine.source_language == self.source_language.currentData()
                and self.engine.provider == self.translation_mode.currentData()
                and self.engine.api_key == self.selected_api_key()
                and (self.engine.provider != 'openai' or
                     (self.engine.base_url, self.engine.model) == self.selected_openai_settings()))

    def selected_openai_settings(self):
        return self.openai_base_url.text().strip(), self.openai_model.currentData() or ''

    def change_source_language(self):
        try:
            save_source_language(self.source_language.currentData())
            self.language_note.clear()
            self.language_note.hide()
        except (OSError, ValueError):
            self.language_note.setText('원문 언어를 저장하지 못했습니다. settings.json의 상태와 권한을 확인하세요.')
            self.language_note.show()
        self.api_key_save_timer.start()

    def invalidate_model_list(self):
        self.models_generation += 1
        self.openai_model.hidePopup()
        self.openai_model.clear()
        self.models_note.setText('주소 또는 키가 변경되었습니다. 모델 선택을 눌러 다시 불러오세요.')

    def load_models(self):
        if self.models_future is not None:
            return
        self.models_note.setText('모델 목록을 불러오는 중…')
        self.models_request_generation = self.models_generation
        try:
            self.models_future = request_model_list(self.openai_base_url.text().strip(),
                                                   self.openai_api_key.text().strip())
        except Exception:
            self.openai_model.open_requested = False
            self.models_note.setText('모델 목록 조회를 시작하지 못했습니다. 다시 시도하세요.')
            return
        self.models_timer.start()

    def finish_model_list(self):
        if self.models_future is None or not self.models_future.done():
            return
        future, self.models_future = self.models_future, None
        self.models_timer.stop()
        if self.models_request_generation != self.models_generation:
            if self.openai_model.open_requested:
                self.load_models()
            return
        try:
            models = future.result()
        except (ValueError, RuntimeError) as exc:
            self.openai_model.open_requested = False
            self.models_note.setText(str(exc))
            return
        except Exception:
            self.openai_model.open_requested = False
            self.models_note.setText('모델 목록을 불러오지 못했습니다. 주소와 서버 상태를 확인하세요.')
            return
        previous = self.openai_model.currentData()
        open_requested = self.openai_model.open_requested
        self.openai_model.blockSignals(True)
        self.openai_model.clear()
        for model in models:
            self.openai_model.addItem(model, model)
        self.openai_model.setCurrentIndex(self.openai_model.findData(previous) if previous else -1)
        self.openai_model.blockSignals(False)
        self.openai_model.open_requested = open_requested
        self.api_key_save_timer.start()
        if not models:
            self.models_note.setText('서버가 제공한 모델이 없습니다. 키와 서버의 모델 설정을 확인하세요.')
        elif previous and previous not in models:
            self.models_note.setText('기존 모델이 목록에 없습니다. 사용할 모델을 다시 선택하세요.')
        else:
            self.models_note.setText(f'모델 {len(models)}개를 불러왔습니다. 사용할 모델을 선택하면 자동으로 적용됩니다.')
        self.openai_model.show_loaded_models()

    def selected_api_key(self):
        field = {'luna': self.api_key, 'deepl': self.deepl_api_key,
                 'openai': self.openai_api_key}[self.translation_mode.currentData()]
        return field.text().strip()

    def save_api_key(self):
        self.api_key_save_timer.stop()
        try:
            save_api_keys(self.api_key.text(), self.deepl_api_key.text(),
                          openai_key=self.openai_api_key.text(), openai_base_url=self.openai_base_url.text(),
                          openai_model=self.openai_model.currentData() or '',
                          translation_provider=self.translation_mode.currentData())
        except (OSError, ValueError):
            self.status.setText('API 키 자동 저장 실패: 설정 파일의 쓰기 권한을 확인하세요.')
            return False
        self.api_settings_error = False
        self.api_settings_note.clear()
        return True

    def change_device(self):
        if hasattr(self, 'device_timer') and self.device_timer.isActive():
            return
        if not self.save_api_key():
            return
        ocr_changed = (self.engine.detection_method != self.detection_method.currentData()
                       or self.engine.device != self.device_mode.currentData())
        if self.translation_mode.currentData() == 'openai' and not ocr_changed:
            try:
                validate_openai_settings(*self.selected_openai_settings(), self.selected_api_key())
            except ValueError as exc:
                self.status.setText(str(exc))
                return
        if self.translation_is_current() and self.engine.is_alive():
            self.status.setText('이미 적용된 설정입니다.')
            return
        self.worker_ready = False
        self.reset_for_settings()
        self.pending_device = self.device_mode.currentData()
        self.pending_detection_method = self.detection_method.currentData()
        self.pending_source_language = self.source_language.currentData()
        self.pending_api_key = self.selected_api_key()
        self.pending_provider = self.translation_mode.currentData()
        self.pending_base_url, self.pending_model = self.selected_openai_settings()
        self.engine.stop_event.set()
        self.signals.status.disconnect(self.status.setText)
        self.signals.failed.disconnect(self.processing_failed)
        self.signals.ready.disconnect(self.ready)
        self.signals.result.disconnect(self.accept_result)
        self.signals.finished.disconnect(self.frame_finished)
        self.signals.progress.disconnect(self.update_progress)
        self.signals.model_download.disconnect(self.update_model_download)
        self.signals.api_key_required.disconnect(self.wait_for_api_key)
        self.device_mode.setEnabled(False)
        self.detection_method.setEnabled(False)
        self.source_language.setEnabled(False)
        self.api_key.setEnabled(False)
        self.deepl_api_key.setEnabled(False)
        self.translation_mode.setEnabled(False)
        self.openai_panel.setEnabled(False)
        self.status.setText('현재 작업 종료 후 설정을 적용합니다…')
        self.stop_progress('설정 적용 대기')
        self.device_timer = QTimer(self)
        self.device_timer.setInterval(100)
        self.device_timer.timeout.connect(self.finish_device_change)
        self.device_timer.start()

    def finish_device_change(self):
        if self.engine.is_alive():
            return
        self.device_timer.stop()
        models = self.engine.ocr_models if self.engine.device == self.pending_device else None
        self.signals = Signals()
        self.engine = Engine(self.signals, device=self.pending_device,
                             detection_method=self.pending_detection_method,
                             api_key=self.pending_api_key, provider=self.pending_provider,
                             base_url=self.pending_base_url, model=self.pending_model,
                             io_logger=self.io_logger, source_language=self.pending_source_language,
                             ocr_models=models)
        self.engine.generation = self.result_floor
        self.engine.cancel_before = self.result_floor
        self.engine.detector_size = self.detection_mode.currentData()
        self.connect_engine()
        self.device_mode.setEnabled(True)
        self.detection_method.setEnabled(True)
        self.source_language.setEnabled(True)
        self.api_key.setEnabled(True)
        self.deepl_api_key.setEnabled(True)
        self.translation_mode.setEnabled(True)
        self.openai_panel.setEnabled(True)
        self.engine.start()

    def wait_for_api_key(self, engine):
        if (engine is not self.engine or engine.stop_event.is_set()
                or engine.provider != self.translation_mode.currentData()):
            return
        self.worker_ready = False
        self.set_task_progress(1, 1, 'OCR 준비 완료 · API 키 입력 대기')
        self.status.setText('선택한 번역 서비스의 API 키를 입력하세요. 입력 후 자동으로 적용됩니다.')

    def ready(self):
        if (self.engine.stop_event.is_set()
                or self.engine.provider != self.translation_mode.currentData()):
            return
        self.worker_ready = True
        if self.single_shot and self.running:
            if self.latest is None:
                self.finish_snapshot_capture(self.engine.generation)
            else:
                self.submit_snapshot()
            return
        label = 'GPU (CUDA)' if self.engine.device == 'cuda' else 'CPU'
        warning = ' · TrOCR 로딩 실패: 영어는 EasyOCR 사용' if self.engine.ocr_models.handwriting_failed else ''
        self.status.setText(f'{label} · {TRANSLATION_MODES[self.engine.provider]} 준비 완료{warning} · 번역 시작을 누르세요.')

    def processing_failed(self, engine, generation, message):
        if engine is not self.engine or not engine.valid(generation):
            return
        if self.single_shot:
            self.running = False
            self.toggle_button.setText('번역 시작')
            self.show()
        self.status.setText(message)
        self.stop_progress('오류 · 상태 메시지를 확인하세요')

    def change_detection_mode(self, *_):
        self.engine.detector_size = self.detection_mode.currentData()
        self.reset_frame()

    def reset_frame(self, *_):
        if self.worker_ready:
            self.stop_progress('번역 대기')
        self.capture.invalidate()
        if self.single_shot:
            self.running = False
            self.toggle_button.setText('번역 시작')
        self.engine.generation += 1
        self.result_floor = self.engine.generation
        self.engine.cancel_before = self.result_floor
        self.engine.latest_job = None
        self.reference = None
        self.latest = None
        self.submitted = False
        self.overlay.clear()

    def select_region(self, single_shot=False):
        self.region_indicator.hide()
        self.settings_dialog.close()
        self.running = False
        self.toggle_button.setText('번역 시작')
        self.reset_frame()
        self.single_shot = single_shot
        self.region = None
        self.overlay.hide()
        self.selector = Selector(self.screens.currentData())
        self.selector.selected.connect(self.set_region)
        self.selector.cancelled.connect(self.cancel_selection)
        self.selector.show()
        self.selector.activateWindow()

    def cancel_selection(self):
        self.region_indicator.hide()
        self.single_shot = False
        self.status.setText('영역 선택을 취소했습니다.')
        self.show()

    def set_region(self, area):
        self.region_indicator.hide()
        self.region = None
        self.overlay.setGeometry(area)
        self.overlay.show()


        try:
            self.region = native_region(self.overlay)
        except OSError as exc:
            self.running = False
            self.reset_frame()
            self.toggle_button.setText('번역 시작')
            self.overlay.hide()
            self.show()
            log.exception('Selected region lookup failed')
            self.status.setText(f'영역 좌표 조회 실패: {exc} — 영역을 다시 선택하세요.')
            return
        log.info('Selected region=%s', self.region)
        self.region_indicator.show_region(area, duration_ms=3000 if self.single_shot else 0)
        self.update_capture_mode()
        if self.single_shot:
            self.capture_snapshot()
            return
        self.status.setText('영역 선택 완료 · 번역 시작을 누르세요.')
        self.show()

    def retry_translation(self):
        if self.single_shot and self.region is not None:
            self.capture_snapshot()
        else:
            self.reset_frame()

    def capture_snapshot(self):
        self.reset_frame()
        self.running = True
        self.toggle_button.setText('취소')
        self.status.setText('선택 영역 캡처 중…')
        self.overlay.hide()
        generation = self.engine.generation
        QTimer.singleShot(150, lambda: self.finish_snapshot_capture(generation))

    def finish_snapshot_capture(self, generation):
        if (generation != self.engine.generation or not self.single_shot
                or not self.running or self.region is None or self.submitted):
            return
        try:
            pixels = self.capture.grab(self.region)
            if pixels is None:
                QTimer.singleShot(30, lambda: self.finish_snapshot_capture(generation))
                return
            self.latest = pixels
            self.reference = self.latest.copy()
            self.show()
            if self.worker_ready:
                self.submit_snapshot()
            else:
                self.status.setText('영역 캡처 완료 · 번역 설정이 적용되면 자동으로 번역합니다.')
        except Exception as exc:
            self.running = False
            self.toggle_button.setText('번역 시작')
            self.show()
            log.exception('Snapshot capture failed')
            self.status.setText(f'캡처 실패: {exc} — 다시 번역을 눌러 재시도하세요.')
            self.warn_capture_protection(exc)

    def warn_capture_protection(self, error):
        if not isinstance(error, CaptureProtectionError):
            return
        previous = getattr(self, 'capture_warning', None)
        if previous is not None and previous.isVisible():
            return
        if previous is not None:
            previous.deleteLater()
        self.capture_warning = QMessageBox(QMessageBox.Icon.Warning, '화면 캡처 보호 확인',
            str(error) + '\n\nESET을 사용한다면 개요 → 브라우저 화면 보호 → 일시 중지 → 적용 후 '
            '다시 번역하세요. 번역이 끝나면 보호를 다시 켜세요.\n'
            '다른 보안 프로그램을 사용한다면 해당 프로그램의 화면 캡처 보호 설정을 확인하세요.',
            QMessageBox.StandardButton.Ok, self)
        self.capture_warning.open()

    def submit_snapshot(self):
        if not self.translation_is_current():
            self.status.setText('선택한 번역 서비스의 설정을 확인하세요. 변경한 설정은 자동 적용됩니다. 전환 중이면 잠시 기다려주세요.')
            return
        if self.latest is not None and not self.submitted:
            self.submitted = True
            self.status.setText('드래그한 영역 번역 중…')
            self.engine.submit((self.engine.generation, self.latest, self.manual.isChecked()))

    def update_capture_mode(self):
        self.timer.setInterval(150)
        if self.single_shot:
            self.capture_note.setText('Manga Live 아래 창을 읽습니다. 드래그 번역은 화면 변경 후 다시 번역을 누르세요.')
        else:
            self.capture_note.setText('Manga Live 창을 제외하고 아래 화면의 변화를 감지합니다.')

    def toggle(self):
        if self.running:
            self.running = False
            self.reset_frame()
            self.toggle_button.setText('번역 시작')
            self.status.setText('드래그 번역을 취소했습니다.' if self.single_shot else '일시정지')
            return
        if not self.translation_is_current():
            self.status.setText('선택한 번역 서비스의 설정을 확인하세요. 변경한 설정은 자동 적용됩니다. 적용 중이면 잠시 기다려주세요.')
            return
        if self.region is None:
            self.status.setText('영역 선택을 눌러 번역할 화면 영역을 지정하세요.')
            return
        self.single_shot = False
        self.update_capture_mode()
        self.region_indicator.show_region(self.overlay.geometry())
        self.running = not self.running
        self.toggle_button.setText('일시정지' if self.running else '번역 시작')
        self.reset_frame()
        if self.running:
            self.overlay.show()
        self.status.setText('화면 변화 감시 중…' if self.running else '일시정지')

    def tick(self):
        self.resource_monitor.set_active(self.running and self.engine.processing.is_set()
                                         and not self.engine.stop_event.is_set())
        usage = self.resource_monitor.snapshot().text()
        if usage != self.resource_note.text():
            self.resource_note.setText(usage)
        if not self.running or self.region is None or self.single_shot:
            return
        if not self.translation_is_current():
            return
        try:
            pixels = self.capture.grab(self.region)
            if pixels is None:
                return
            previous = self.latest
            self.latest = pixels
            now = time.monotonic()
            if changed(self.reference, pixels):
                self.engine.generation += 1
                self.reference = pixels
                self.last_change = now
                self.submitted = False
                offset = scroll_offset(previous, pixels) if self.overlay.rows else None
                if offset is not None:
                    tracked = move_rows(self.overlay.rows, offset, pixels.shape)
                else:
                    tracked = []
                    for box, text in self.overlay.rows:
                        moved = relocate(box, previous, pixels)
                        if moved is not None:
                            tracked.append((moved, text))
                self.overlay.display(tracked, (pixels.shape[1], pixels.shape[0]))
            self.submit_pending_frame()
        except Exception as exc:
            self.running = False
            self.toggle_button.setText('번역 시작')
            self.reset_frame()
            log.exception('Capture failed')
            self.status.setText(f'캡처 실패: {exc}')
            self.warn_capture_protection(exc)

    def submit_pending_frame(self):
        if not self.translation_is_current():
            return
        if (self.running and self.latest is not None and not self.submitted and
                time.monotonic()-self.last_change >= 0.25):
            self.submitted = True
            self.engine.submit((self.engine.generation, self.latest, self.manual.isChecked()))

    def accept_result(self, generation, rows, pixels):
        if not self.running or generation < self.result_floor or self.latest is None:
            return
        combined = list(self.overlay.rows)
        for box, text in rows:
            moved = relocate(box, pixels, self.latest)
            if moved is not None:
                combined = merge_row(combined, (moved, text))
        if combined:
            self.overlay.display(combined, (self.latest.shape[1], self.latest.shape[0]))
            self.status.setText(f'{len(combined)}개 영역 번역 표시 중 · 나머지 처리 중')

    def frame_finished(self, generation):
        if self.running and generation == self.engine.generation:
            if self.single_shot:
                self.running = False
                self.toggle_button.setText('번역 시작')
                if self.overlay.rows:
                    self.status.setText(f'드래그 번역 완료 · {len(self.overlay.rows)}개 영역 표시 · 다시 번역으로 재실행')
                else:
                    self.status.setText('문자를 인식하지 못했습니다. 원문 언어를 확인하고 영역을 좁히거나 말풍선 하나 모드를 사용해 보세요.')
            else:
                self.status.setText(f'{len(self.overlay.rows)}개 영역 표시 · 화면 변화 대기 중')

    def closeEvent(self, event):
        self.version_updates.close()
        self.resource_monitor.close()
        self.save_current_ui_settings()
        self.settings_dialog.close()
        self.deepl_usage_closed = True
        self.deepl_usage_debounce.stop()
        self.deepl_usage_poll.stop()
        self.deepl_usage_refresh.stop()
        self.deepl_usage_future = None
        self.deepl_usage_generation += 1
        self.models_timer.stop()
        self.models_future = None
        self.models_generation += 1
        self.hotkeys.close()
        if self.api_key_save_timer.isActive():
            self.save_api_key()
        if hasattr(self, 'device_timer'):
            self.device_timer.stop()
        self.timer.stop()
        self.log_cleanup_timer.stop()
        self.engine.stop_event.set()
        self.engine.generation += 1
        self.overlay.close()
        self.region_indicator.close()
        if hasattr(self, 'selector'):
            self.selector.close()
        self.capture.close()
        event.accept()
        QApplication.instance().quit()
