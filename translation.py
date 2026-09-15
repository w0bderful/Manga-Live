from pathlib import Path
import re
from types import SimpleNamespace

MODEL_ID = 'facebook/nllb-200-distilled-600M'
CACHE_DIR = Path(__file__).resolve().parent / '.models' / 'huggingface' / 'hub'


def validate_device(device):
    import torch
    if device not in ('cpu', 'cuda'):
        raise ValueError('지원하지 않는 실행 모드입니다.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU 모드를 사용할 수 없습니다. NVIDIA GPU·드라이버와 CUDA용 PyTorch가 필요합니다. CPU 모드를 선택하세요.')
    return device


def load_cached(loader, **kwargs):
    try:
        return loader.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR), local_files_only=True, **kwargs)
    except OSError:
        return loader.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR), **kwargs)


class TranslationClient:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.tokenizer = None

    async def __aenter__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        validate_device(self.device)
        self.tokenizer = load_cached(AutoTokenizer, src_lang='jpn_Jpan')
        self.model = load_cached(AutoModelForSeq2SeqLM)
        self.model.to(self.device)
        self.model.eval()
        self.target_id = self.tokenizer.convert_tokens_to_ids('kor_Hang')
        if self.target_id is None or self.target_id == self.tokenizer.unk_token_id:
            raise RuntimeError('NLLB 한국어 토큰을 찾을 수 없습니다.')
        return self

    async def __aexit__(self, *args):
        self.model = None
        self.tokenizer = None

    async def translate(self, text, src='ja', dest='ko'):
        import torch
        if (src, dest) != ('ja', 'ko'):
            raise ValueError('현재 번역 방향은 일본어 → 한국어입니다.')
        if self.model is None:
            raise RuntimeError('NLLB 모델을 먼저 로딩해야 합니다.')
        if not text.strip():
            return SimpleNamespace(text='')
        inputs = self.tokenizer(text, return_tensors='pt', truncation=False)
        if inputs['input_ids'].shape[-1] > 512:
            raise ValueError('번역할 글이 너무 깁니다. 말풍선 단위로 영역을 줄여주세요.')

        sentences = [part.strip() for part in re.findall(r'[^。！？!?\n]+[。！？!?]*|[。！？!?]+', text)
                     if part.strip()]
        translated = []
        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors='pt', truncation=False)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.inference_mode():
                output = self.model.generate(**inputs, forced_bos_token_id=self.target_id,
                                             max_new_tokens=256, max_length=None,
                                             num_beams=2, do_sample=False)
            result = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
            if not result:
                raise RuntimeError('NLLB가 빈 번역을 반환했습니다.')
            translated.append(result)
        return SimpleNamespace(text=' '.join(translated))
