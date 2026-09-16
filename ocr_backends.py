from pathlib import Path
import logging
import cv2
import numpy as np
from core import Box, text_boxes
from app_settings import validate_source_language

MODES = {
    'manga_ocr': '기본 OCR (언어별 인식)',
    'easyocr': 'EasyOCR (영어 손글씨 자동 보완)',
    'opencv': 'OpenCV 감지 (언어별 인식)',
}
MODE_DESCRIPTIONS = {
    'manga_ocr': 'EasyOCR로 영역을 감지합니다. 일본어 선택 시 Manga OCR, 자동 감지·영어 선택 시 EasyOCR로 읽고 불확실한 영어는 TrOCR로 보완합니다.',
    'easyocr': 'EasyOCR로 영역 감지와 글자 인식을 수행합니다. 자동 감지·영어 선택 시 불확실한 영어를 TrOCR로 보완합니다.',
    'opencv': 'OpenCV 후보를 EasyOCR 글자 감지로 검증해 그림을 걸러내고 합쳐진 영역을 분리합니다. 일본어 선택 시 Manga OCR, 자동 감지·영어 선택 시 EasyOCR로 읽고 불확실한 영어는 TrOCR로 보완합니다.',
}
log = logging.getLogger(__name__)


def validate_device(device):
    import torch
    if device not in ('cpu', 'cuda'):
        raise ValueError('지원하지 않는 실행 모드입니다.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU 모드를 사용할 수 없습니다. NVIDIA GPU·드라이버와 CUDA용 PyTorch가 필요합니다. CPU 모드를 선택하세요.')
    return device


def opencv_boxes(pixels, canvas_size=None):
    height, width = pixels.shape[:2]
    scale = 1.0 if canvas_size is None else min(1.0, canvas_size/max(height, width))
    small = pixels if scale == 1.0 else cv2.resize(
        pixels, (max(1, round(width*scale)), max(1, round(height*scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    accepted = np.zeros(count, dtype=bool)
    max_glyph = max(80, min(256, round(max(gray.shape)*0.12)))
    for index, (x, y, w, h, area) in enumerate(stats[1:count], start=1):
        if (2 <= h <= max_glyph and 2 <= w <= max_glyph and area >= 3
                and 0.04 <= area/(w*h) <= 0.95 and max(w, h) <= 12*min(w, h)):
            accepted[index] = True
    # Copy only accepted components, not unrelated picture strokes inside their bounds.
    glyphs = np.where(accepted[labels], 255, 0).astype(np.uint8)

    grouped = cv2.dilate(glyphs, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horizontal = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ink = np.count_nonzero(glyphs[y:y+h, x:x+w])
        if w >= 6 and h >= 6 and ink >= 12:
            horizontal.append([x/scale, (x+w)/scale, y/scale, (y+h)/scale])
    return text_boxes(horizontal, [], width, height)


def verified_opencv_boxes(candidates, text_regions):
    """Keep text detector separation; expand only with a closely matching CV candidate."""
    result = []
    for text in text_regions:
        best, best_score = None, 0.75
        for candidate in candidates:
            overlap_w = max(0, min(text.x+text.w, candidate.x+candidate.w)-max(text.x, candidate.x))
            overlap_h = max(0, min(text.y+text.h, candidate.y+candidate.h)-max(text.y, candidate.y))
            overlap = overlap_w*overlap_h
            union = text.w*text.h + candidate.w*candidate.h - overlap
            score = overlap/union if union else 0
            if score > best_score:
                best, best_score = candidate, score
        if best is None:
            result.append(text)
        else:
            x, y = min(text.x, best.x), min(text.y, best.y)
            result.append(Box(x, y, max(text.x+text.w, best.x+best.w)-x,
                              max(text.y+text.h, best.y+best.h)-y, text.vertical))
    return result


class OcrBackend:
    def __init__(self, mode, device, root, source_language='ja'):
        if mode not in MODES:
            raise ValueError('지원하지 않는 OCR 모드입니다.')
        self.source_language = validate_source_language(source_language)
        self.use_easyocr = mode == 'easyocr' or source_language != 'ja'
        self.languages = ['en'] if source_language == 'en' else ['ja', 'en']
        self.mode = mode
        self.reader = None
        self.mocr = None
        self.handwriting = None
        self.handwriting_failed = False
        self.status = None
        self.device = device
        self.root = Path(root)
        if mode != 'opencv' or self.use_easyocr:
            import easyocr
            self.reader = easyocr.Reader(self.languages, gpu=(device == 'cuda'),
                recognizer=self.use_easyocr, model_storage_directory=str(self.root/'.models'/'easyocr'),
                user_network_directory=str(self.root/'.models'/'easyocr'/'user'))
        if not self.use_easyocr:
            from manga_ocr import MangaOcr
            self.mocr = MangaOcr(force_cpu=(device == 'cpu'))
            self.mocr.model.to(device)
            self.mocr.model.eval()

    def detect(self, pixels, canvas_size=None, manual=False):
        height, width = pixels.shape[:2]
        if manual:
            return [Box(0, 0, width, height)]
        if self.mode == 'opencv':
            boxes = opencv_boxes(pixels, canvas_size)
            if float(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).std()) < 2:
                return []
            if self.reader is None:
                import easyocr
                self.reader = easyocr.Reader(self.languages, gpu=(self.device == 'cuda'),
                    recognizer=False, model_storage_directory=str(self.root/'.models'/'easyocr'),
                    user_network_directory=str(self.root/'.models'/'easyocr'/'user'))
            recovered = self.detect_craft(pixels, canvas_size)
            verified = verified_opencv_boxes(boxes, recovered)
            log.info('OpenCV verified detection: candidates=%s text=%s final=%s',
                     len(boxes), len(recovered), len(verified))
            return verified
        return self.detect_craft(pixels, canvas_size)

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
            min_size=5, text_threshold=0.55, low_text=0.25, link_threshold=0.3, add_margin=0.05)
        horizontal = [[x1-padding, x2-padding, y1-padding, y2-padding]
                      for x1, x2, y1, y2 in horizontal[0]]
        free = [(np.asarray(poly)-padding).tolist() for poly in free[0]]
        return text_boxes(horizontal, free, width, height)

    def recognize(self, crop):
        if self.use_easyocr:
            pixels = np.asarray(crop.convert('RGB'))
            corners = pixels[[0, 0, -1, -1], [0, -1, 0, -1]]
            background = tuple(int(v) for v in np.median(corners, axis=0))
            pixels = cv2.copyMakeBorder(pixels, 16, 16, 16, 16,
                                       cv2.BORDER_CONSTANT, value=background)
            if self.source_language in ('en', 'auto'):
                from easyocr.utils import get_paragraph
                lines = self.reader.readtext(pixels, detail=1, paragraph=False,
                    min_size=5, text_threshold=0.55, low_text=0.25,
                    link_threshold=0.3, add_margin=0.05)
                corrected = []
                for polygon, original, confidence in lines:
                    original = self.correct_handwriting(pixels, polygon, original, confidence)
                    corrected.append((polygon, original, confidence))
                return ' '.join(row[1] for row in get_paragraph(corrected, mode='ltr')).strip()
            # A merged balloon can contain several lines. recognize() without
            # line boxes squeezes that whole balloon into a single text line.
            parts = self.reader.readtext(pixels, detail=0, paragraph=True,
                min_size=5, text_threshold=0.55, low_text=0.25,
                link_threshold=0.3, add_margin=0.05)
            return ' '.join(parts).strip()
        return self.mocr(crop).strip()

    def correct_handwriting(self, pixels, polygon, original, confidence):
        if self.handwriting_failed:
            return original
        try:
            from handwriting_ocr import needs_handwriting, choose_reading, crop_text_line, HandwritingOcr
            if not needs_handwriting(original, confidence):
                return original
            line = crop_text_line(pixels, polygon)
            if line is None:
                return original
            if self.handwriting is None:
                if self.status:
                    self.status('영어 손글씨 보완 모델 로딩 중… 첫 사용 시 다운로드합니다.')
                self.handwriting = HandwritingOcr(self.device, self.root)
            return choose_reading(original, self.handwriting(line))
        except Exception as exc:
            self.handwriting_failed = True
            self.handwriting = None
            log.warning('Handwriting enhancement disabled; retaining EasyOCR readings (%s)', type(exc).__name__)
            if self.status:
                self.status('손글씨 보완을 사용할 수 없어 기본 OCR 결과로 계속합니다. 보완 재시도는 OCR 설정을 전환하거나 재실행하세요.')
            return original
