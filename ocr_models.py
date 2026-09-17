"""Device validation and reusable OCR model preparation."""
import logging
from pathlib import Path
from model_downloads import DownloadReporter, easyocr_downloads, huggingface_downloads
log = logging.getLogger(__name__)


def validate_device(device):
    import torch
    if device not in ('cpu', 'cuda'):
        raise ValueError('지원하지 않는 실행 모드입니다.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU 모드를 사용할 수 없습니다. NVIDIA GPU·드라이버와 CUDA용 PyTorch가 필요합니다. CPU 모드를 선택하세요.')
    return device


class OcrModels:
    """Models owned by one controller, reused by its sequential engine threads."""
    def __init__(self, device, root):
        self.device = device
        self.root = Path(root)
        self.readers = {}
        self.mocr = None
        self.handwriting = None
        self.handwriting_failed = False
        self.comic_detector = None

    def preload(self, status, stopped, progress=lambda *args: None, detection_method='opencv', download_progress=None):
        total = 5 if detection_method == 'comic' else 4
        progress(0, total, 'OCR 로딩')
        validate_device(self.device)
        for step, (language, languages) in enumerate([('ja', ['ja', 'en']), ('en', ['en'])], 1):
            if stopped():
                return False
            if language not in self.readers:
                status(f'EasyOCR {"일본어" if language == "ja" else "영어"} 모델 미리 로딩 중…')
                import easyocr
                reporter = DownloadReporter(f'EasyOCR {"일본어" if language == "ja" else "영어"}', download_progress, stopped)
                with easyocr_downloads(easyocr, reporter):
                    self.readers[language] = easyocr.Reader(languages, gpu=self.device == 'cuda',
                        model_storage_directory=str(self.root/'.models'/'easyocr'),
                        user_network_directory=str(self.root/'.models'/'easyocr'/'user'))
            progress(step, total, 'OCR 로딩')
        if stopped():
            return False
        if self.mocr is None:
            status('Manga OCR 모델 미리 로딩 중…')
            from manga_ocr import MangaOcr
            with huggingface_downloads(DownloadReporter('Manga OCR', download_progress, stopped)):
                model = MangaOcr(force_cpu=self.device == 'cpu')
            model.model.to(self.device)
            model.model.eval()
            self.mocr = model
        progress(3, total, 'OCR 로딩')
        if stopped():
            return False
        if self.handwriting is None:
            status('영어 TrOCR 모델 미리 로딩 중…')
            try:
                from handwriting_ocr import HandwritingOcr
                with huggingface_downloads(DownloadReporter('TrOCR', download_progress, stopped)):
                    self.handwriting = HandwritingOcr(self.device, self.root)
                self.handwriting_failed = False
            except Exception as exc:
                self.handwriting_failed = True
                log.warning('Handwriting preload failed (%s)', type(exc).__name__)
        progress(4, total, 'OCR 로딩')
        if detection_method == 'comic' and self.comic_detector is None:
            if stopped():
                return False
            try:
                from comic_detector import ComicTextDetector
                self.comic_detector = ComicTextDetector(self.root, status, stopped, device=self.device,
                    download_progress=download_progress)
            except Exception as exc:
                if stopped():
                    return False
                raise RuntimeError('Comic Text Detector 준비에 실패했습니다. setup.bat·인터넷 연결·모델 저장 권한을 '
                                   '확인하세요. GPU 오류라면 CPU 모드를 선택하거나 영역 감지를 '
                                   'OpenCV (기존 방식)로 변경하세요.') from exc
        progress(total, total, 'OCR 로딩 완료' if not self.handwriting_failed else 'OCR 준비 · 영어는 EasyOCR 사용')
        return not stopped()
