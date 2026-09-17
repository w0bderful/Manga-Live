"""Coordinate-preserving OCR preprocessing and language-probe helpers."""
import logging
import hashlib
import re
import cv2
import numpy as np
log = logging.getLogger(__name__)


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
