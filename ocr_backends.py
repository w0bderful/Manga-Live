from pathlib import Path
import logging
import cv2
import numpy as np
from core import Box, text_boxes
from app_settings import validate_source_language

MODES = {
    'manga_ocr': 'Manga OCR (EasyOCR 감지 + Manga OCR 인식)',
    'easyocr': 'EasyOCR (감지 + 인식)',
    'opencv': 'OpenCV (감지 부족 시 EasyOCR 보완 + Manga OCR 인식)',
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
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    glyphs = np.zeros_like(binary)
    max_glyph = max(80, min(256, round(max(gray.shape)*0.12)))
    for x, y, w, h, area in stats[1:count]:
        if (2 <= h <= max_glyph and 2 <= w <= max_glyph and area >= 3
                and 0.04 <= area/(w*h) <= 0.95 and max(w, h) <= 12*min(w, h)):
            glyphs[y:y+h, x:x+w] |= binary[y:y+h, x:x+w]

    grouped = cv2.dilate(glyphs, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horizontal = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ink = np.count_nonzero(glyphs[y:y+h, x:x+w])
        if w >= 6 and h >= 6 and ink >= 12:
            horizontal.append([x/scale, (x+w)/scale, y/scale, (y+h)/scale])
    return text_boxes(horizontal, [], width, height)


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
            if len(boxes) > 1:
                return boxes
            if float(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).std()) < 2:
                return boxes
            if self.reader is None:
                import easyocr
                self.reader = easyocr.Reader(self.languages, gpu=(self.device == 'cuda'),
                    recognizer=False, model_storage_directory=str(self.root/'.models'/'easyocr'),
                    user_network_directory=str(self.root/'.models'/'easyocr'/'user'))
            recovered = self.detect_craft(pixels, canvas_size)
            log.info('OpenCV supplemental detection: primary=%s craft=%s', len(boxes), len(recovered))
            return recovered or boxes
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
            parts = self.reader.recognize(np.asarray(crop.convert('RGB')), detail=0, paragraph=True)
            return ' '.join(parts).strip()
        return self.mocr(crop).strip()
