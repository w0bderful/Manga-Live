"""Local recognition of English handwriting, one detected text line at a time."""
from pathlib import Path
import re

MODEL_ID = 'microsoft/trocr-small-handwritten'


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
