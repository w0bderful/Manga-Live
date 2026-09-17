"""Connect region detectors, language routing and text recognizers."""
import logging
from pathlib import Path
from dataclasses import dataclass, replace
from collections import OrderedDict
import re
import cv2
import numpy as np
from core import Box
from app_settings import validate_source_language, DETECTION_METHODS
from ocr_image import probe_language, image_key, prepare_ocr_image, vertical_probe_image
from ocr_regions import (
    opencv_boxes, verified_opencv_boxes, split_panel_regions, balloon_text_boxes,
    fit_balloon_regions,
)
from ocr_models import OcrModels, validate_device
log = logging.getLogger(__name__)

@dataclass(frozen=True)
class OcrReading:
    text: str
    language: str


class OcrBackend:
    def __init__(self, device, root, source_language='ja', models=None,
                 detection_method='opencv', stopped=lambda: False):
        if detection_method not in DETECTION_METHODS:
            raise ValueError('지원하지 않는 영역 감지 방식입니다.')
        self.detection_method = detection_method
        self.stopped = stopped
        self.comic_detector = getattr(models, 'comic_detector', None) if models is not None else None
        self.source_language = validate_source_language(source_language)
        self.languages = ['en'] if source_language == 'en' else ['ja', 'en']
        self.reader = None
        self.english_reader = None
        self.language_cache = OrderedDict()
        self.mocr = None
        self.handwriting = None
        self.handwriting_failed = False
        self.status = None
        self.device = device
        self.root = Path(root)
        if models is not None:
            if models.device != device:
                raise ValueError('OCR 모델과 처리 장치가 일치하지 않습니다.')
            self.reader = models.readers['en' if source_language == 'en' else 'ja']
            self.english_reader = models.readers['en']
            self.mocr = models.mocr
            self.handwriting = models.handwriting
            self.handwriting_failed = models.handwriting_failed
            return
        import easyocr
        self.reader = easyocr.Reader(self.languages, gpu=(device == 'cuda'),
            model_storage_directory=str(self.root/'.models'/'easyocr'),
            user_network_directory=str(self.root/'.models'/'easyocr'/'user'))
        if source_language == 'auto':
            self.english_reader = easyocr.Reader(['en'], gpu=(device == 'cuda'),
                model_storage_directory=str(self.root/'.models'/'easyocr'),
                user_network_directory=str(self.root/'.models'/'easyocr'/'user'))
        elif source_language == 'en':
            self.english_reader = self.reader
        if source_language != 'en':
            from manga_ocr import MangaOcr
            self.mocr = MangaOcr(force_cpu=(device == 'cpu'))
            self.mocr.model.to(device)
            self.mocr.model.eval()

    def detect(self, pixels, canvas_size=None, manual=False):
        height, width = pixels.shape[:2]
        self.language_cache.clear()
        if manual:
            return [Box(0, 0, width, height)]
        if float(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).std()) < 2:
            return []
        if self.detection_method == 'comic':
            if self.comic_detector is None:
                from comic_detector import ComicTextDetector
                self.comic_detector = ComicTextDetector(self.root, self.status or (lambda _: None),
                                                        self.stopped, device=self.device)
            try:
                regions = self.comic_detector.detect(pixels, canvas_size, self.stopped)
            except Exception as exc:
                if self.stopped():
                    raise
                raise RuntimeError('Comic Text Detector 감지에 실패했습니다. 다시 번역하거나 CPU 모드 또는 '
                                   '영역 감지를 OpenCV (기존 방식)로 변경하세요.') from exc
        else:
            candidates = self.detect_opencv(pixels, canvas_size)
            regions = self.detect_craft(pixels, canvas_size)
            regions = verified_opencv_boxes(candidates, regions)
            regions = fit_balloon_regions(pixels, regions)
            regions = split_panel_regions(pixels, regions)
        if self.source_language == 'auto':
            from PIL import Image
            languages = []
            for box in regions:
                x1,y1,x2,y2 = box.crop()
                languages.append(self.detect_language(Image.fromarray(pixels[y1:y2,x1:x2])))
            result = []
            for box, language in zip(regions, languages):
                if language == 'en':
                    box = replace(box, vertical=False)
                x1,y1,x2,y2 = box.crop()
                self.remember_language(Image.fromarray(pixels[y1:y2,x1:x2]), language)
                result.append(box)
            return result
        return regions

    def detect_opencv(self, pixels, canvas_size):
        try:
            return opencv_boxes(pixels, canvas_size)
        except cv2.error:
            log.warning('OpenCV detection failed; using EasyOCR text regions')
            if self.status:
                self.status('OpenCV 영역 감지 실패 · EasyOCR 감지기로 계속합니다.')
            return []

    def detect_craft(self, pixels, canvas_size):
        height, width = pixels.shape[:2]
        padding = 32
        corners = pixels[[0, 0, -1, -1], [0, -1, 0, -1]]
        background = tuple(int(v) for v in np.median(corners, axis=0))
        padded = cv2.copyMakeBorder(pixels, padding, padding, padding, padding,
                                   cv2.BORDER_CONSTANT, value=background)
        magnification = max(1.0, min(3.0, 256/max(height, width)))
        limit = max(height, width) if canvas_size is None else canvas_size
        limit = max(limit+2*padding, round(max(padded.shape[:2])*magnification)) if magnification > 1 else limit+2*padding
        horizontal, free = self.reader.detect(padded, canvas_size=limit, mag_ratio=magnification,
            min_size=5, text_threshold=0.55, low_text=0.25, link_threshold=0.3, add_margin=0.05,
            width_ths=.05)
        horizontal = [[x1-padding, x2-padding, y1-padding, y2-padding]
                      for x1, x2, y1, y2 in horizontal[0]]
        free = [(np.asarray(poly)-padding).tolist() for poly in free[0]]
        return balloon_text_boxes(horizontal, free, pixels)

    def remember_language(self, crop, language):
        key = image_key(crop)
        self.language_cache[key] = language
        self.language_cache.move_to_end(key)
        if len(self.language_cache) > 256:
            self.language_cache.popitem(last=False)

    def detect_language(self, crop):
        try:
            return self._detect_language(crop)
        except Exception as exc:
            log.warning('Language probe failed; trying Manga OCR (%s)', type(exc).__name__)
            self.remember_language(crop, None)
            return None

    def _detect_language(self, crop):
        key = image_key(crop)
        if key in self.language_cache:
            return self.language_cache[key]
        if self.status:
            self.status('EasyOCR로 원문 언어 판별 중…')
        pixels = np.asarray(crop.convert('RGB'))
        if float(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).std()) < 2:
            self.remember_language(crop, None)
            return None
        corrected = prepare_ocr_image(pixels)
        pixels = corrected
        scale = min(3.0, max(1.0, 64/min(crop.size)))
        if scale > 1:
            pixels = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        pixels = cv2.copyMakeBorder(pixels,16,16,16,16,cv2.BORDER_REPLICATE)
        lines = self.reader.readtext(pixels, detail=1, paragraph=False,
            min_size=5, text_threshold=.45, low_text=.25, link_threshold=.3)
        language = probe_language(lines)
        if language is None:
            # Reflow before adding the detection border. Replicated ink at the crop
            # edge can otherwise join all columns and hide valid vertical Japanese.
            # A second, inset probe excludes balloon/panel edges for language only;
            # final recognition always keeps the original complete crop.
            for inset in dict.fromkeys((0, min(6, (min(crop.size)-8)//2))):
                if inset < 0:
                    continue
                inner = corrected[inset:corrected.shape[0]-inset,
                                  inset:corrected.shape[1]-inset] if inset else corrected
                probe_scale = min(3.0, max(1.0, 192/min(inner.shape[:2])))
                if probe_scale > 1:
                    inner = cv2.resize(inner, None, fx=probe_scale, fy=probe_scale,
                                       interpolation=cv2.INTER_CUBIC)
                vertical = vertical_probe_image(inner)
                if vertical is not None:
                    text = self.reader.recognize(vertical, detail=1, paragraph=False)
                    if probe_language(text) == 'ja':
                        language = 'ja'
                        break
        if language is None:
            english = self.english_reader.readtext(pixels, detail=1, paragraph=False,
                min_size=5, text_threshold=.45, low_text=.25, link_threshold=.3)
            # Require reliable Latin evidence; the English-only model can invent words on Japanese.
            reliable = [row for row in english if row[2] >= .7 and len(re.findall(r'[A-Za-z]',row[1])) >= 2]
            if probe_language(reliable) == 'en':
                language = 'en'
        self.remember_language(crop, language)
        return language

    def recognize(self, crop):
        return self.read(crop).text

    def read(self, crop):
        language = self.detect_language(crop) if self.source_language == 'auto' else self.source_language
        if language is None:
            # Blank regions are not failed language probes and must not hallucinate text.
            if float(cv2.cvtColor(np.asarray(crop.convert('RGB')), cv2.COLOR_RGB2GRAY).std()) < 2:
                return OcrReading('', 'auto')
            language = 'ja'
            if self.status:
                self.status('언어 판별 실패 · Manga OCR로 인식 시도 중…')
        elif self.source_language == 'auto' and self.status:
            self.status('일본어 감지 · Manga OCR 인식 중…' if language == 'ja' else '영어 감지 · 글씨체 확인 중…')
        pixels = prepare_ocr_image(np.asarray(crop.convert('RGB')))
        if language == 'en':
            reader = self.english_reader
            corners = pixels[[0, 0, -1, -1], [0, -1, 0, -1]]
            background = tuple(int(v) for v in np.median(corners, axis=0))
            pixels = cv2.copyMakeBorder(pixels, 16, 16, 16, 16,
                                       cv2.BORDER_CONSTANT, value=background)
            return OcrReading(self.read_english(pixels, reader), language)
        from PIL import Image
        return OcrReading(self.mocr(Image.fromarray(pixels)).strip(), language)

    def read_english(self, pixels, reader):
        from easyocr.utils import get_paragraph
        from handwriting_ocr import crop_text_line, usable_handwriting, classify_text_style
        if self.status:
            self.status('영어 글씨체 확인 중…')
        horizontal, free = reader.detect(pixels, min_size=5, text_threshold=.55,
            low_text=.25, link_threshold=.3, add_margin=.05)
        polygons = [[[x1,y1],[x2,y1],[x2,y2],[x1,y2]] for x1,x2,y1,y2 in horizontal[0]] + list(free[0])
        lines = []
        for polygon in polygons:
            line = crop_text_line(pixels, polygon)
            if line is None:
                continue
            style = classify_text_style(line)
            if self.status:
                label = {'printed': '인쇄체 추정 · EasyOCR',
                         'handwritten': '손글씨·장식체 추정 · TrOCR',
                         'unknown': '글씨체 불확실 · TrOCR'}[style]
                self.status('TrOCR를 사용할 수 없어 EasyOCR 결과로 계속합니다.'
                            if self.handwriting_failed else f'영어 {label} 인식 중…')
            if style == 'printed':
                # Require the recognizer's confidence as well as regular shapes.
                # A handwritten/decorative line can also have a regular baseline.
                try:
                    rows = reader.recognize(np.asarray(line), detail=1, paragraph=False)
                    text = ' '.join(row[1].strip() for row in rows if row[1].strip())
                    reliable = bool(text) and all(np.isfinite(row[2]) and row[2] >= .8 for row in rows)
                except Exception as exc:
                    log.warning('Printed English recognition failed; trying TrOCR (%s)', type(exc).__name__)
                    text, reliable = '', False
                if not reliable:
                    if self.status and not self.handwriting_failed:
                        self.status('영어 인쇄체 인식 불확실 · TrOCR로 다시 인식 중…')
                    candidate = self.read_handwriting(line)
                    if usable_handwriting(candidate):
                        text = candidate
            else:
                candidate = self.read_handwriting(line)
                if usable_handwriting(candidate):
                    text = candidate
                else:
                    if self.status and not self.handwriting_failed:
                        self.status('영어 TrOCR 인식 불확실 · EasyOCR로 다시 인식 중…')
                    text = ' '.join(reader.recognize(np.asarray(line), detail=0, paragraph=False)).strip()
            if text:
                lines.append((polygon, text, 1.0))
        return ' '.join(row[1] for row in get_paragraph(lines, mode='ltr')).strip()

    def read_handwriting(self, line):
        if self.handwriting_failed:
            return ''
        try:
            from handwriting_ocr import HandwritingOcr
            if self.handwriting is None:
                if self.status:
                    self.status('영어 TrOCR 모델 로딩 중… 첫 사용 시 다운로드합니다.')
                self.handwriting = HandwritingOcr(self.device, self.root)
            return self.handwriting(line).strip()
        except Exception as exc:
            self.handwriting_failed = True
            self.handwriting = None
            log.warning('Handwriting enhancement disabled; retaining EasyOCR readings (%s)', type(exc).__name__)
            if self.status:
                self.status('TrOCR를 사용할 수 없어 EasyOCR 결과로 계속합니다. 재시도는 OCR 설정을 전환하거나 재실행하세요.')
            return ''
