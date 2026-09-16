"""Local recognition of English handwriting, one detected text line at a time."""
from pathlib import Path
import re
from difflib import SequenceMatcher

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


def needs_handwriting(text, confidence):
    from wordfreq import zipf_frequency

    if re.search(r'[\u3040-\u30ff\u3400-\u9fff]', text) or not text.strip():
        return False
    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text)
    if not words:
        return False
    rare = any(len(word) >= 3 and zipf_frequency(word.lower(), 'en') < 2.5 for word in words)
    mixed = bool(re.search(r'[A-Za-z]\d|\d[A-Za-z]', text))
    return confidence < 0.8 or rare or mixed or bool(re.search(r'[{}<>\\|]', text))


def choose_reading(original, candidate):
    if not candidate or word_quality(candidate) < word_quality(original) + 0.6:
        return original
    def numbers(text, other):
        result = []
        for match in re.finditer(r'\d+(?:[.,]\d+)*', text):
            attached = ((match.start() > 0 and text[match.start()-1].isalpha())
                        or (match.end() < len(text) and text[match.end()].isalpha()))
            unit = re.match(r'(?:cm|mm|km|m|kg|g|ml|l)\b', text[match.end():], re.IGNORECASE)
            split_word = re.match(r'\s+([A-Za-z]{2,})\b', text[match.end():])
            letter = {'0':'o', '1':'il', '2':'z', '5':'s', '8':'b', '9':'g'}.get(match.group(), '')
            joined_letter = bool(split_word and letter and any(
                len(word) == len(split_word[1])+1 and word[0].lower() in letter
                and SequenceMatcher(None, split_word[1].lower(), word[1:].lower()).ratio() >= .75
                for word in re.findall(r'[A-Za-z]+', other)))
            # A single digit attached to a word may be a letter error (he1lo, 9rown).
            # Measurements, counts and multi-digit values must not be rewritten.
            if (not attached and not joined_letter) or unit or len(match.group()) > 1:
                result.append(match.group())
        return result
    if numbers(original, candidate) != numbers(candidate, original):
        return original
    before = re.sub(r'[^a-z]', '', original.lower())
    after = re.sub(r'[^a-z]', '', candidate.lower())
    if not before or not after or len(after) > max(8, len(before) * 2) or len(after) < len(before) * .5:
        return original
    original_words = re.findall(r'[A-Za-z]+', original)
    candidate_words = re.findall(r'[A-Za-z]+', candidate)
    if len(original_words) == len(candidate_words) and any(
            len(old) >= 3 and len(new) > len(old) * 1.4
            for old, new in zip(original_words, candidate_words)):
        return original
    if before and SequenceMatcher(None, before, after).ratio() < .4:
        return original
    return candidate


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
