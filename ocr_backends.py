from pathlib import Path
from dataclasses import dataclass, replace
from collections import OrderedDict
import hashlib
import re
import logging
import cv2
import numpy as np
from core import Box, text_boxes
from app_settings import validate_source_language

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrReading:
    text: str
    language: str


def probe_language(lines):
    """Use EasyOCR's script and confidence, not the translation service, to route OCR."""
    scores = {'ja': 0.0, 'en': 0.0}
    for _, text, confidence in lines:
        confidence = float(confidence)
        if not np.isfinite(confidence) or confidence < .03:
            continue
        japanese = len(re.findall(r'[\u3040-\u30ff\u3400-\u9fff]', text))
        english = len(re.findall(r'[A-Za-z]', text))
        scores['ja'] += japanese * confidence
        scores['en'] += english * confidence
    if scores['ja'] >= .1 and scores['ja'] >= scores['en'] * .4:
        return 'ja'
    if scores['en'] >= .5 and scores['en'] > scores['ja'] * 2.5:
        return 'en'
    return None


def image_key(crop):
    return crop.size, hashlib.sha256(crop.tobytes()).digest()


def prepare_ocr_image(pixels):
    """Correct recognition input without changing coordinates or the captured image."""
    try:
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        if min(gray.shape) < 3 or float(gray.std()) < 2:
            return pixels.copy()
        low, high = np.percentile(gray, (.5, 99.5))
        span = high-low
        # Avoid distorting already-clear strokes or amplifying nearly-flat noise.
        if span < 12 or span >= 220:
            return pixels.copy()
        gain = min(2.0, 255.0/span)
        gray = np.clip(gray.astype(np.float32)*gain + 255-high*gain, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(gray, (3, 3), .6)
        gray = cv2.addWeighted(gray, 1.15, blurred, -.15, 0)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    except cv2.error:
        log.warning('OpenCV image correction failed; using original OCR input')
        return pixels.copy()


def vertical_probe_image(pixels):
    """Reflow separated upright glyphs for EasyOCR's horizontal recognizer."""
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if np.count_nonzero(ink) > ink.size * .5:
        ink = 255 - ink

    def spans(mask, gap):
        closed = cv2.morphologyEx(mask.astype(np.uint8)[None, :], cv2.MORPH_CLOSE,
                                 np.ones((1, gap), np.uint8))[0]
        edges = np.diff(np.pad(closed, (1, 1)).astype(int))
        return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

    glyphs = []
    for left, right in reversed(spans(np.count_nonzero(ink, axis=0) >= 2, 5)):
        width = right-left
        rows = spans(np.count_nonzero(ink[:,left:right], axis=1) >= 2, max(3, round(width*.15)))
        if width < 5 or len(rows) < 2 or rows[-1][1]-rows[0][0] < width*1.8:
            continue
        for top, bottom in rows:
            if 3 <= bottom-top <= width*1.8:
                glyphs.append(pixels[top:bottom,left:right])
    if len(glyphs) < 2:
        return None
    # Keep each glyph upright, centered in a square cell, then read horizontally.
    cell = max(max(glyph.shape[:2]) for glyph in glyphs) + 8
    if cell * len(glyphs) > 8192:
        return None
    strip = np.full((cell+16, cell*len(glyphs)+16, 3), 255, np.uint8)
    for index, glyph in enumerate(glyphs):
        h, w = glyph.shape[:2]
        x, y = 8+index*cell+(cell-w)//2, 8+(cell-h)//2
        strip[y:y+h,x:x+w] = glyph
    return strip


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


def split_panel_regions(pixels, regions):
    """Keep a detected region from spanning a straight, dark comic panel border."""
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    result = []
    for box in regions:
        x1, y1, x2, y2 = box.crop()
        crop = gray[y1:y2, x1:x2]
        cuts = []
        # A panel divider crosses almost the whole text region; individual glyphs do not.
        if box.h >= 40:
            columns = np.flatnonzero(np.mean(crop < 80, axis=0) >= .9)
            # Treat the two strokes of a narrow panel gutter as one divider.
            for run in np.split(columns, np.flatnonzero(np.diff(columns) > 5)+1):
                if (len(run) and run[-1]-run[0]+1 <= max(8, box.w*.06)
                        and run[0] >= 12 and box.w-run[-1]-1 >= 12):
                    cuts.append((int(run[0]), int(run[-1])+1))
        left = 0
        for start, end in cuts:
            if start-left < 12:
                continue
            result.append(Box(x1+left, y1, start-left, box.h, box.vertical))
            left = end
        result.append(Box(x1+left, y1, box.w-left, box.h, box.vertical))
    return result


def balloon_text_boxes(horizontal, free, pixels):
    """Group nearby text lines without joining across dark balloon/panel outlines."""
    height, width = pixels.shape[:2]
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    ink = cv2.morphologyEx((gray < 180).astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((3,3),np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1-ink, 8)
    regions = {}

    def interior(box):
        if box not in regions:
            x1,y1,x2,y2 = box.crop()
            histogram = np.bincount(labels[y1:y2,x1:x2].ravel(),minlength=count)
            histogram[0] = 0
            label = int(histogram.argmax())
            regions[box] = label if (histogram[label] > box.w*box.h*.35
                and stats[label,cv2.CC_STAT_AREA] >= 200) else None
        return regions[box]

    def same_enclosed_region(a,b):
        label = interior(a)
        if label is None or label != interior(b):
            return False
        x,y,w,h,area = stats[label]
        return (0 < x and 0 < y and x+w < width and y+h < height
                and area < width*height*.025)

    def can_merge(a, b):
        first,second = interior(a),interior(b)
        if first is not None and second is not None and first != second:
            return False
        if same_enclosed_region(a,b):
            return True
        xo = min(a.x+a.w, b.x+b.w)-max(a.x,b.x)
        yo = min(a.y+a.h, b.y+b.h)-max(a.y,b.y)
        overlap = max(0,xo)*max(0,yo)
        if overlap >= .35*min(a.w*a.h,b.w*b.h):
            return True
        row = (a.w >= a.h and b.w >= b.h and xo > .55*min(a.w,b.w)
               and -yo < .65*min(a.h,b.h))
        col = (a.h > a.w and b.h > b.w and yo > .55*min(a.h,b.h)
               and abs(a.y-b.y) < .3*min(a.h,b.h)
               and -xo < .65*min(a.w,b.w))
        inline = (a.w >= a.h and b.w >= b.h and yo > .7*min(a.h,b.h)
                  and -xo < .65*min(a.h,b.h))
        stack = (xo > .6*min(a.w,b.w) and max(a.w,b.w) < 2*min(a.w,b.w)
                 and -yo < .5*min(a.w,b.w))
        if stack and yo > 0:
            return True
        if not (row or col or stack or inline):
            return False
        if col or inline:
            left,right = sorted((a,b),key=lambda box:box.x)
            x1,x2 = max(left.x,left.x+left.w-6),min(right.x+right.w,right.x+6)
            y1,y2 = max(a.y,b.y)+3,min(a.y+a.h,b.y+b.h)-3
            gap = gray[y1:y2,x1:x2]
            barrier = gap.size and np.mean(np.any(gap<120,axis=1)) > .55
        else:
            top,bottom = sorted((a,b),key=lambda box:box.y)
            y1,y2 = max(top.y,top.y+top.h-6),min(bottom.y+bottom.h,bottom.y+6)
            x1,x2 = max(a.x,b.x)+3,min(a.x+a.w,b.x+b.w)-3
            gap = gray[y1:y2,x1:x2]
            barrier = gap.size and np.mean(np.any(gap<120,axis=0)) > .55
        return not barrier

    result = text_boxes(horizontal,[],width,height,can_merge=can_merge)
    # The bounding rectangle of an angled note includes empty corners. It must
    # not act as a bridge between the neighboring upright dialogue columns.
    for angled in text_boxes([],free,width,height,merge=False):
        for index,box in enumerate(result):
            xo = max(0,min(box.x+box.w,angled.x+angled.w)-max(box.x,angled.x))
            yo = max(0,min(box.y+box.h,angled.y+angled.h)-max(box.y,angled.y))
            if (same_enclosed_region(box,angled)
                    or xo*yo >= .7*angled.w*angled.h and can_merge(box,angled)):
                x,y = min(box.x,angled.x),min(box.y,angled.y)
                result[index] = Box(x,y,max(box.x+box.w,angled.x+angled.w)-x,
                                    max(box.y+box.h,angled.y+angled.h)-y,box.vertical)
                break
        else:
            result.append(angled)
    return sorted(result,key=lambda box:(box.y,-box.x))


def fit_balloon_regions(pixels, regions):
    """Trim small detection overhangs to enclosed white interiors; remove duplicates."""
    height,width = pixels.shape[:2]
    gray = cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
    ink = cv2.morphologyEx((gray<180).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,_ = cv2.connectedComponentsWithStats(1-ink,8)
    fitted = []
    for box in regions:
        x1,y1,x2,y2 = box.crop()
        histogram = np.bincount(labels[y1:y2,x1:x2].ravel(),minlength=count)
        histogram[0] = 0
        label = int(histogram.argmax())
        x,y,w,h,area = (int(value) for value in stats[label])
        if (histogram[label] > box.w*box.h*.35 and 200 <= area < width*height*.025
                and 0 < x and 0 < y and x+w < width and y+h < height):
            left,top,right,bottom = max(x1,x),max(y1,y),min(x2,x+w),min(y2,y+h)
            if (right-left)*(bottom-top) >= box.w*box.h*.75:
                box = Box(left,top,right-left,bottom-top,box.vertical)
        fitted.append(box)
    # Small detector fragments already inside a larger text crop must not get a
    # second translation on top of the same sentence.
    result = []
    for box in sorted(fitted,key=lambda b:b.w*b.h,reverse=True):
        if any(max(0,min(box.x+box.w,b.x+b.w)-max(box.x,b.x))
               *max(0,min(box.y+box.h,b.y+b.h)-max(box.y,b.y)) >= box.w*box.h*.9
               for b in result):
            continue
        result.append(box)
    return sorted(result,key=fitted.index)


class OcrModels:
    """Models owned by one controller, reused by its sequential engine threads."""
    def __init__(self, device, root):
        self.device = device
        self.root = Path(root)
        self.readers = {}
        self.mocr = None
        self.handwriting = None
        self.handwriting_failed = False

    def preload(self, status, stopped, progress=lambda *args: None):
        progress(0, 4, 'OCR 로딩')
        validate_device(self.device)
        for step, (language, languages) in enumerate([('ja', ['ja', 'en']), ('en', ['en'])], 1):
            if stopped():
                return False
            if language not in self.readers:
                status(f'EasyOCR {"일본어" if language == "ja" else "영어"} 모델 미리 로딩 중…')
                import easyocr
                self.readers[language] = easyocr.Reader(languages, gpu=self.device == 'cuda',
                    model_storage_directory=str(self.root/'.models'/'easyocr'),
                    user_network_directory=str(self.root/'.models'/'easyocr'/'user'))
            progress(step, 4, 'OCR 로딩')
        if stopped():
            return False
        if self.mocr is None:
            status('Manga OCR 모델 미리 로딩 중…')
            from manga_ocr import MangaOcr
            model = MangaOcr(force_cpu=self.device == 'cpu')
            model.model.to(self.device)
            model.model.eval()
            self.mocr = model
        progress(3, 4, 'OCR 로딩')
        if stopped():
            return False
        if self.handwriting is None:
            status('영어 TrOCR 모델 미리 로딩 중…')
            try:
                from handwriting_ocr import HandwritingOcr
                self.handwriting = HandwritingOcr(self.device, self.root)
                self.handwriting_failed = False
            except Exception as exc:
                self.handwriting_failed = True
                log.warning('Handwriting preload failed (%s)', type(exc).__name__)
        progress(4, 4, 'OCR 로딩 완료' if not self.handwriting_failed else 'OCR 준비 · 영어는 EasyOCR 사용')
        return not stopped()


class OcrBackend:
    def __init__(self, device, root, source_language='ja', models=None):
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
            self.status('일본어 감지 · Manga OCR 인식 중…' if language == 'ja' else '영어 감지 · TrOCR → EasyOCR 인식 중…')
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
        from handwriting_ocr import crop_text_line, usable_handwriting
        horizontal, free = reader.detect(pixels, min_size=5, text_threshold=.55,
            low_text=.25, link_threshold=.3, add_margin=.05)
        polygons = [[[x1,y1],[x2,y1],[x2,y2],[x1,y2]] for x1,x2,y1,y2 in horizontal[0]] + list(free[0])
        lines = []
        for polygon in polygons:
            line = crop_text_line(pixels, polygon)
            if line is None:
                continue
            # Only uncertain/failed TrOCR readings invoke the EasyOCR recognizer.
            candidate = self.read_handwriting(line)
            if usable_handwriting(candidate):
                text = candidate
            else:
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
