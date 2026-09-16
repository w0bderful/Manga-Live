# 이미지 자료

- `manga-live.png`: 내장 imagegen으로 생성한 Manga Live 아이콘 원본.
- `manga-live.ico`: 동일 이미지를 Windows용 16·24·32·48·64·128·256px ICO로 변환한 파일. 프로그램 창과 작업 표시줄에 사용한다.
- `screenshot.png`: 실제 Windows에서 프로그램 창과 번역 오버레이를 촬영한 화면. 자체 제작한 샘플 페이지와 미리 준비한 번역문을 사용했으며 OCR·번역 API 품질 검증 자료는 아니다. 개인 설정과 API 키는 사용하지 않았다.

## 아이콘 생성 프롬프트

```text
Use case: logo-brand. Asset type: square Windows app icon and favicon for Manga Live, a Japanese-to-Korean manga screen translator. Primary request: a polished, highly legible icon combining a white manga speech bubble and two bold cyan scanning corners on a deep charcoal rounded-square tile. Simple original graphic, flat crisp shapes, very few elements, strong silhouette recognizable at 16 pixels. One short dark vertical column of three broad typographic strokes inside the bubble suggests manga text, no actual letters or words. Centered mark fills most of the square with modest safe margin. Transparent canvas outside rounded tile. No mockup, no perspective, no drop shadow outside tile, no watermark. Output a single square icon.
```
