"""Background OCR and translation jobs with cancellation, retries and caches."""
import logging
from runtime_paths import APP_DIR as ROOT
import asyncio
from collections import OrderedDict
from dataclasses import replace
from ocr_image import image_key
import queue
import re
import threading
import time
from PIL import Image
from PyQt6.QtCore import QObject, pyqtSignal
from core import relocate
from translation import create_translation_client, TRANSLATION_MODES
from ocr_backends import OcrBackend, OcrModels, OcrReading
from app_settings import validate_source_language, DETECTION_METHODS, DEFAULT_DETECTION_METHOD
log = logging.getLogger(__name__)
TRANSLATION_RETRIES = 5


class TranslationCancelled(Exception):
    pass


class Signals(QObject):
    status = pyqtSignal(str)
    result = pyqtSignal(int, object, object)
    ready = pyqtSignal()
    failed = pyqtSignal(object, int, str)
    finished = pyqtSignal(int)
    progress = pyqtSignal(object, int, int, int, str)
    model_download = pyqtSignal(object, object)
    api_key_required = pyqtSignal(object)


class Engine(threading.Thread):

    def __init__(self, signals, device='cpu', api_key='', provider='luna',
                 base_url='', model='', io_logger=None, source_language='ja', ocr_models=None,
                 detection_method=DEFAULT_DETECTION_METHOD):
        super().__init__(daemon=True)
        self.signals = signals
        self.device = device
        if detection_method not in DETECTION_METHODS:
            raise ValueError('지원하지 않는 영역 감지 방식입니다.')
        self.detection_method = detection_method
        self.source_language = validate_source_language(source_language)
        self.api_key = api_key
        self.provider = provider
        self.base_url, self.model = base_url, model
        self.io_logger = io_logger
        self.jobs = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.processing = threading.Event()
        self.generation = 0
        self.cache = OrderedDict()
        self.ocr_cache = OrderedDict()
        self.latest_job = None
        self.cancel_before = 0
        self.detector_size = None
        self.ocr_models = ocr_models if ocr_models is not None else OcrModels(device, ROOT)

    async def process_boxes(self, generation, pixels, boxes, mocr, client):
        self.signals.progress.emit(self, generation, 0, len(boxes), '번역')
        for i, box in enumerate(boxes):
            if self.stop_event.is_set() or generation < self.cancel_before:
                break
            active_generation, active_pixels, active_box = generation, pixels, box
            latest = self.latest_job
            if latest is not None and latest[0] > generation:
                active_generation, active_pixels, _ = latest
                active_box = relocate(box, pixels, active_pixels)
                if active_box is None:
                    continue
            self.signals.status.emit(f'글자 인식 중… {i+1}/{len(boxes)}')
            x1, y1, x2, y2 = active_box.crop()
            crop = Image.fromarray(active_pixels[y1:y2, x1:x2])
            key = image_key(crop)
            if key in self.ocr_cache:
                reading = self.ocr_cache[key]
                self.ocr_cache.move_to_end(key)
            else:
                ocr_started = time.monotonic()
                reading = mocr(crop)
                if isinstance(reading, str):
                    reading = OcrReading(reading.strip(), self.source_language)
                log.info('OCR completed: seconds=%.2f chars=%s language=%s', time.monotonic()-ocr_started, len(reading.text), reading.language)
                self.ocr_cache[key] = reading
                if len(self.ocr_cache) > 256:
                    self.ocr_cache.popitem(last=False)
            if self.stop_event.is_set() or generation < self.cancel_before:
                break
            source, source_language = reading.text, reading.language
            pattern = r'[\u3040-\u30ff\u3400-\u9fff]' if source_language == 'ja' else r'[A-Za-z\u3040-\u30ff\u3400-\u9fff]'
            if not re.search(pattern, source):
                self.signals.progress.emit(self, generation, i+1, len(boxes), '번역')
                continue
            if source_language == 'en' or (source_language == 'auto'
                    and re.search(r'[A-Za-z]', source)
                    and not re.search(r'[\u3040-\u30ff\u3400-\u9fff]', source)):
                active_box = replace(active_box, vertical=False)
            self.signals.status.emit(f'{TRANSLATION_MODES[self.provider]} 응답 대기 중… {i+1}/{len(boxes)}')
            try:
                translated = await self.translate(client, source, source_language, generation=generation)
            except TranslationCancelled:
                return
            if self.stop_event.is_set() or generation < self.cancel_before:
                break

            self.signals.result.emit(active_generation, [(active_box, translated)], active_pixels)
            self.signals.progress.emit(self, generation, i+1, len(boxes), '번역')

    def valid(self, generation):
        return not self.stop_event.is_set() and generation == self.generation

    def submit(self, job):
        self.latest_job = job
        try:
            self.jobs.get_nowait()
        except queue.Empty:
            pass
        self.jobs.put_nowait(job)

    async def request_translation(self, client, text, source_language, generation):
        def check_cancelled():
            if self.stop_event.is_set() or (generation is not None and generation < self.cancel_before):
                raise TranslationCancelled()

        for attempt in range(TRANSLATION_RETRIES+1):
            check_cancelled()
            try:
                result = await client.translate(text, src=source_language, dest='ko')
                translated = result.text.strip()
                if not translated:
                    raise RuntimeError('빈 번역 결과')
                return translated
            except Exception as exc:
                check_cancelled()
                if attempt == TRANSLATION_RETRIES:
                    raise
                log.warning('Translation retry: provider=%s retry=%s/%s error_type=%s',
                            self.provider, attempt+1, TRANSLATION_RETRIES, type(exc).__name__)
                self.signals.status.emit(f'번역 API 요청 실패 · 재시도 {attempt+1}/{TRANSLATION_RETRIES}')
                # Wait one second, but let cancel/stop interrupt the retry promptly.
                for _ in range(10):
                    check_cancelled()
                    await asyncio.sleep(.1)

    async def translate(self, client, text, source_language=None, *, generation=None):
        source_language = source_language or self.source_language
        key = text if source_language == self.source_language else (source_language, text)
        cached = key in self.cache
        request_id = self.io_logger.input(text, self.provider, cached) if self.io_logger else None
        try:
            if cached:
                self.cache.move_to_end(key)
                translated = self.cache[key]
            else:
                translated = await self.request_translation(client, text, source_language, generation)
                self.cache[key] = translated
                if len(self.cache) > 512:
                    self.cache.popitem(last=False)
        except Exception as exc:
            if self.io_logger:
                self.io_logger.output(request_id, None, self.provider, cached, type(exc).__name__)
            raise
        if self.io_logger:
            self.io_logger.output(request_id, translated, self.provider, cached)
        return translated

    def run(self):
        try:
            if not self.ocr_models.preload(self.signals.status.emit, self.stop_event.is_set,
                    lambda done, total, label: self.signals.progress.emit(self, -1, done, total, label),
                    detection_method=self.detection_method,
                    download_progress=lambda info: self.signals.model_download.emit(self, info)):
                return
            if self.provider != 'openai' and not self.api_key.strip():
                self.signals.api_key_required.emit(self)
                return
            with asyncio.Runner() as runner:
                runner.run(self.work())
        except Exception as exc:
            log.exception('Engine initialization failed')
            if not self.stop_event.is_set():
                self.signals.failed.emit(self, self.generation, f'초기화 실패: {exc}\nOCR 장치 설정·의존성과 인터넷 연결을 확인한 뒤 다시 시도하세요.')

    async def work(self):
        import torch
        backend = OcrBackend(self.device, ROOT, self.source_language,
                             models=self.ocr_models, detection_method=self.detection_method,
                             stopped=self.stop_event.is_set)
        backend.status = self.signals.status.emit
        async with create_translation_client(self.provider, self.api_key,
                                             base_url=self.base_url, model=self.model) as client:
            self.signals.ready.emit()
            while not self.stop_event.is_set():
                try:
                    generation, pixels, manual = self.jobs.get(timeout=0.2)
                except queue.Empty:
                    continue
                if not self.valid(generation):
                    continue
                started = time.monotonic()
                try:
                    if not self.valid(generation):
                        continue
                    self.processing.set()
                    self.signals.status.emit('글자 영역 감지 중…')
                    self.signals.progress.emit(self, generation, 0, 0, '글자 영역 감지 중…')
                    with torch.inference_mode():
                        detect_started = time.monotonic()
                        boxes = backend.detect(pixels, self.detector_size, manual)
                        log.info('Detection: %.3fs canvas=%s boxes=%s',
                                 time.monotonic()-detect_started, self.detector_size, len(boxes))
                        await self.process_boxes(generation, pixels, boxes, backend.read, client)
                    if self.valid(generation):
                        self.signals.progress.emit(self, generation, 1, 1, '번역 완료' if boxes else '감지 완료 · 글자 없음')
                        self.signals.finished.emit(generation)
                        if not boxes:
                            self.signals.status.emit('글자 영역을 찾지 못했습니다. 말풍선 하나 모드를 사용해 보세요.')
                        log.info('Frame processed: generation=%s regions=%s seconds=%.2f',
                                 generation, len(boxes), time.monotonic()-started)
                except Exception as exc:
                    log.exception('Frame processing failed')
                    if self.valid(generation):
                        self.signals.failed.emit(self, generation, f'처리 실패: {exc} — 다시 번역을 눌러 재시도하세요.')
                finally:
                    self.processing.clear()
