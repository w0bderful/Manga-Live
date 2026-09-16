"""Local recognition of English handwriting, one detected text line at a time."""
from pathlib import Path
import re

MODEL_ID = 'microsoft/trocr-small-handwritten'


def classify_text_style(image):
    """Conservative line-shape heuristic, not font identification or a classifier model."""
    import cv2
    import numpy as np

    try:
        gray = np.asarray(image.convert('L'))
        if min(gray.shape) < 8 or float(gray.std()) < 8:
            return 'unknown'
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if np.count_nonzero(ink) > ink.size / 2:
            ink = cv2.bitwise_not(ink)
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        parts = stats[1:]
        parts = parts[(parts[:, 4] >= 5) & (parts[:, 3] >= 4)]
        if len(parts) < 6:
            return 'unknown'
        reference = float(np.percentile(parts[:, 3], 90))
        parts = parts[(parts[:, 3] >= .5 * reference) & (parts[:, 2] <= 2 * reference)]
        if not 6 <= len(parts) <= 160 or reference < 10:
            return 'unknown'
        height = float(np.median(parts[:, 3]))
        x = parts[:, 0] + parts[:, 2] / 2
        bottom = parts[:, 1] + parts[:, 3]
        dx = x[:, None] - x
        dy = bottom[:, None] - bottom
        separated = np.abs(dx) > 2 * height
        if not np.any(separated):
            return 'unknown'
        slope = float(np.median(dy[separated] / dx[separated]))
        if abs(slope) > .35:
            return 'unknown'
        baseline = bottom - slope * x
        aligned = float(np.max(np.mean(np.abs(baseline[:, None] - baseline) <= max(1, .08 * height), axis=1)))
        variation = float(np.ptp(np.percentile(parts[:, 3], [10, 90])) / height)
        if aligned >= .85 and variation <= .5:
            return 'printed'
        if aligned < .65 and variation > .3:
            return 'handwritten'
        return 'unknown'
    except (cv2.error, ValueError, TypeError):
        return 'unknown'


def crop_text_line(pixels, polygon):
    import cv2
    import numpy as np
    from PIL import Image

    points = np.asarray(polygon, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    width = int(round(max(np.linalg.norm(points[1]-points[0]), np.linalg.norm(points[2]-points[3]))))
    height = int(round(max(np.linalg.norm(points[3]-points[0]), np.linalg.norm(points[2]-points[1]))))
    if width < 2 or height < 2:
        return None
    target = np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    transform = cv2.getPerspectiveTransform(points, target)
    line = cv2.warpPerspective(pixels, transform, (width, height),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return Image.fromarray(line)


def word_quality(text):
    from wordfreq import zipf_frequency

    words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?", text)
    if not words:
        return 0.0
    score = sum(min(6.0, zipf_frequency(word.lower(), 'en'))
                if not re.search(r'\d', word) else 0.0 for word in words) / len(words)
    noise = len(re.findall(r"[^A-Za-z0-9\s.,!?;:'’\"()\-]", text))
    return score - min(3.0, noise)


def usable_handwriting(text):
    if not text or re.search(r'[\u3040-\u30ff\u3400-\u9fff]', text):
        return False
    # Keep measurements and counts; dictionary scores only cover ordinary words.
    if re.fullmatch(r'\d+(?:[.,]\d+)?\s*(?:cm|mm|km|m|kg|g|ml|l|%)?', text, re.I):
        return True
    return bool(re.search(r'[A-Za-z]', text)) and word_quality(text) >= 2.5


class HandwritingOcr:
    def __init__(self, device, root):
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from transformers import AutoImageProcessor, VisionEncoderDecoderModel

        cache = str(Path(root) / '.models' / 'huggingface' / 'hub')
        def load(local):
            options = dict(cache_dir=cache, local_files_only=local)
            processor = AutoImageProcessor.from_pretrained(MODEL_ID, **options)
            vocabulary = hf_hub_download(MODEL_ID, 'sentencepiece.bpe.model', **options)
            model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, **options)
            return processor, vocabulary, model

        try:
            self.processor, vocabulary, model = load(True)
        except OSError:
            try:
                self.processor, vocabulary, model = load(False)
            except OSError as exc:
                raise RuntimeError('영어 손글씨 OCR 모델을 불러오지 못했습니다. 인터넷 연결과 .models 폴더를 확인하세요.') from exc
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=vocabulary)
        self.model = model.to(device).eval()
        self.device = device

    def __call__(self, image):
        import torch

        with torch.inference_mode():
            pixels = self.processor(images=image.convert('RGB'), return_tensors='pt').pixel_values.to(self.device)
            tokens = self.model.generate(pixels, max_new_tokens=96)[0].tolist()
        # This checkpoint uses XLM-R IDs: four special tokens, and an offset
        # of one for ordinary SentencePiece IDs. No tokenizer conversion needed.
        pieces = [token - 1 for token in tokens if 4 <= token <= self.tokenizer.get_piece_size()]
        return self.tokenizer.decode(pieces).strip()
