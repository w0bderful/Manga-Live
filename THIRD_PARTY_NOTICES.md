# Comic Text Detector

Manga Live는 [dmMaze/comic-text-detector](https://github.com/dmMaze/comic-text-detector)의 ONNX 모델을 사용합니다. 프로젝트의 라이선스는 GPL-3.0이며 [원문](licenses/comic-text-detector-GPL-3.0.txt)을 함께 보관합니다.

- 모델: `comictextdetector.pt.onnx`
- 배포: [manga-image-translator beta-0.2.1](https://github.com/zyddnys/manga-image-translator/releases/tag/beta-0.2.1)
- SHA-256: `1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f`
- 모델은 첫 사용 시 `.models/comic-text-detector/`에 다운로드하며 Git에는 포함하지 않습니다.
- `comic_detector.py`는 이 프로그램의 좌표·영역 형식에 맞춘 ONNX 실행 및 후처리 어댑터입니다. 글자 제거·말풍선 외곽 분할은 수행하지 않습니다.

GPU 실행은 Microsoft의 [ONNX Runtime](https://github.com/microsoft/onnxruntime)(MIT 라이선스)의 CUDA 실행 기능을 사용합니다. CPU 실행은 OpenCV DNN을 사용합니다.
