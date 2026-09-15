from pathlib import Path
import cv2
import numpy as np
from core import Box, text_boxes

MODES = {
    'manga_ocr': 'Manga OCR (EasyOCR 감지 + Manga OCR 인식)',
    'easyocr': 'EasyOCR (감지 + 인식)',
    'opencv': 'OpenCV (빠른 영역 감지 + Manga OCR 인식)',
}


def opencv_boxes(pixels, canvas_size=960):
    height, width = pixels.shape[:2]
    scale = min(1.0, canvas_size/max(height, width))
    small = cv2.resize(pixels, (max(1, round(width*scale)), max(1, round(height*scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    glyphs = np.zeros_like(binary)
    for x, y, w, h, area in stats[1:count]:
        if 3 <= h <= 80 and 2 <= w <= 80 and 5 <= area <= 2500 and 0.06 <= area/(w*h) <= 0.95:
            glyphs[y:y+h, x:x+w] |= binary[y:y+h, x:x+w]

    grouped = cv2.dilate(glyphs, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horizontal = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ink = np.count_nonzero(glyphs[y:y+h, x:x+w])
        if w >= 12 and h >= 12 and ink >= 30 and w*h < gray.size*0.6:
            horizontal.append([x/scale, (x+w)/scale, y/scale, (y+h)/scale])
    return text_boxes(horizontal, [], width, height)


class OcrBackend:
    def __init__(self, mode, device, root):
        if mode not in MODES:
            raise ValueError('지원하지 않는 OCR 모드입니다.')
        self.mode = mode
        self.reader = None
        self.mocr = None
        root = Path(root)
        if mode != 'opencv':
            import easyocr
            self.reader = easyocr.Reader(['ja', 'en'], gpu=(device == 'cuda'),
                recognizer=(mode == 'easyocr'), model_storage_directory=str(root/'.models'/'easyocr'),
                user_network_directory=str(root/'.models'/'easyocr'/'user'))
        if mode != 'easyocr':
            from manga_ocr import MangaOcr
            self.mocr = MangaOcr(force_cpu=(device == 'cpu'))
            self.mocr.model.to(device)
            self.mocr.model.eval()

    def detect(self, pixels, canvas_size=960, manual=False):
        height, width = pixels.shape[:2]
        if manual:
            return [Box(0, 0, width, height)]
        if self.mode == 'opencv':
            return opencv_boxes(pixels, canvas_size)
        horizontal, free = self.reader.detect(pixels, canvas_size=canvas_size,
                                              min_size=12, add_margin=0.05)
        return text_boxes(horizontal[0], free[0], width, height)

    def recognize(self, crop):
        if self.mode == 'easyocr':
            parts = self.reader.recognize(np.asarray(crop.convert('RGB')), detail=0, paragraph=True)
            return ' '.join(parts).strip()
        return self.mocr(crop).strip()
