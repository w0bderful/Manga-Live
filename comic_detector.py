"""Comic Text Detector ONNX adapter; OpenCV CPU or ONNX Runtime CUDA inference.

Model: dmMaze/comic-text-detector (see THIRD_PARTY_NOTICES.md).
This adapter consumes its block and line-map outputs without loading pickle code.
"""
import hashlib
from pathlib import Path
import urllib.request

import cv2
import numpy as np

from core import Box

MODEL_NAME = 'comictextdetector.pt.onnx'
MODEL_URL = ('https://github.com/zyddnys/manga-image-translator/releases/download/'
             'beta-0.2.1/' + MODEL_NAME)
MODEL_SIZE = 94669756
MODEL_SHA256 = '1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f'
INPUT_SIZE = 1024


def check_stopped(stopped):
    if stopped():
        raise InterruptedError('Comic Text Detector 준비·감지가 취소되었습니다.')


def valid_model(path):
    if not path.is_file() or path.stat().st_size != MODEL_SIZE:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == MODEL_SHA256


def model_path(root, status, stopped):
    path = Path(root) / '.models' / 'comic-text-detector' / MODEL_NAME
    check_stopped(stopped)
    if valid_model(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.onnx.part')
    status('Comic Text Detector 모델 다운로드 중…')
    try:
        request = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'Manga-Live'})
        with urllib.request.urlopen(request, timeout=15) as response, temporary.open('wb') as stream:
            received = 0
            while True:
                check_stopped(stopped)
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > MODEL_SIZE:
                    raise ValueError('모델 파일 크기가 예상과 다릅니다.')
                stream.write(chunk)
                status(f'Comic Text Detector 다운로드 · {received * 100 // MODEL_SIZE}%')
        check_stopped(stopped)
        if not valid_model(temporary):
            raise ValueError('모델 파일 검증에 실패했습니다. 다시 시도하세요.')
        temporary.replace(path)
        return path
    finally:
        temporary.unlink(missing_ok=True)


def overlap(a, b):
    return max(0, min(a.x+a.w, b.x+b.w)-max(a.x, b.x)) * max(
        0, min(a.y+a.h, b.y+b.h)-max(a.y, b.y))


def suppress(regions):
    """Reconcile partial duplicates from the overview and overlapping tiles."""
    result = []
    for box in sorted(regions, key=lambda b: b.w*b.h, reverse=True):
        if any(overlap(box, other) >= .75 * box.w*box.h for other in result):
            continue
        while True:
            duplicate = next((other for other in result
                if box.vertical == other.vertical and
                overlap(box,other) >= .45*min(box.w*box.h,other.w*other.h)),None)
            if duplicate is None:
                break
            result.remove(duplicate)
            x,y = min(box.x,duplicate.x),min(box.y,duplicate.y)
            box = Box(x,y,max(box.x+box.w,duplicate.x+duplicate.w)-x,
                      max(box.y+box.h,duplicate.y+duplicate.h)-y,box.vertical)
        result.append(box)
    return sorted(result, key=lambda b: (b.y, -b.x))


def line_groups(lines, pixels=None):
    """Group original line rectangles, never a growing block spanning balloons."""
    groups = []
    gray = cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY) if pixels is not None else None
    def linked(a, b):
        if a.vertical != b.vertical:
            return overlap(a, b) >= .5*min(a.w*a.h, b.w*b.h)
        xo = min(a.x+a.w,b.x+b.w)-max(a.x,b.x)
        yo = min(a.y+a.h,b.y+b.h)-max(a.y,b.y)
        if a.vertical:
            connected = (yo > .3*min(a.h,b.h) and -xo <= 2*max(a.w,b.w)
                and (xo>0 or abs(a.y-b.y)<=max(2*max(a.w,b.w),.4*min(a.h,b.h))))
        else:
            connected = ((xo > .25*min(a.w,b.w) and -yo <= 1.5*min(a.h,b.h))
                or (yo > .5*min(a.h,b.h) and -xo <= 2*min(a.h,b.h)))
        if not connected or gray is None:
            return connected
        if xo<0 and yo>0:
            left,right=sorted((a,b),key=lambda r:r.x)
            gap=gray[max(a.y,b.y):min(a.y+a.h,b.y+b.h),left.x+left.w:right.x]
            # Line-map cores stop inside glyph strokes. Ignore the outside
            # thirds, which may still contain the neighboring letters.
            margin=gap.shape[1]//3
            gap=gap[:,margin:gap.shape[1]-margin]
            if gap.size and np.mean(np.any(gap<120,axis=1))>.5:
                return False
        elif yo<0 and xo>0:
            top,bottom=sorted((a,b),key=lambda r:r.y)
            gap=gray[top.y+top.h:bottom.y,max(a.x,b.x):min(a.x+a.w,b.x+b.w)]
            margin=gap.shape[0]//3
            gap=gap[margin:gap.shape[0]-margin,:]
            if gap.size and np.mean(np.any(gap<120,axis=0))>.5:
                return False
        return True
    for i,line in enumerate(lines):
        matches = [group for group in groups if any(linked(line,lines[j]) for j in group)]
        combined = [i]
        for group in matches:
            combined.extend(group)
            groups.remove(group)
        groups.append(combined)
    return groups


def decode_outputs(outputs, width, height, pixels=None):
    blocks = next((a for a in outputs if a.ndim == 3 and a.shape[-1] >= 6), None)
    line_map = next((a for a in outputs if a.ndim == 4 and a.shape[1] == 2), None)
    if blocks is None or line_map is None:
        raise ValueError('Comic Text Detector 모델 출력 형식이 올바르지 않습니다.')
    probability = line_map[0, 0, :height, :width]
    contours, _ = cv2.findContours((probability > .3).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines, cores = [], []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 2 or cv2.contourArea(contour) < 8:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [contour - (x, y)], -1, 1, -1)
        if cv2.mean(probability[y:y+h, x:x+w], mask=mask)[0] < .6:
            continue
        # The probability map covers the inner stroke region, so restore margins.
        pad = max(2, round(min(w, h)*.4))
        left, top = max(0, x-pad), max(0, y-pad)
        right, bottom = min(width, x+w+pad), min(height, y+h+pad)
        lines.append(Box(left, top, right-left, bottom-top, h > w*1.2))
        cores.append(Box(x,y,w,h,h > w*1.2))

    rows = blocks[0]
    rows = rows[np.isfinite(rows).all(axis=1)]
    scores = rows[:, 4] * rows[:, 5:].max(axis=1)
    candidates, confidences = [], []
    for row, score in zip(rows, scores):
        if score < .4:
            continue
        cx, cy, w, h = row[:4]
        left, top = max(0, int(np.floor(cx-w/2))), max(0, int(np.floor(cy-h/2)))
        right, bottom = min(width, int(np.ceil(cx+w/2))), min(height, int(np.ceil(cy+h/2)))
        if right-left < 4 or bottom-top < 4:
            continue
        candidates.append([left, top, right-left, bottom-top])
        confidences.append(float(score))
    kept = cv2.dnn.NMSBoxes(candidates, confidences, .4, .35) if candidates else []
    result = []
    assigned = set()
    groups = line_groups(cores,pixels)
    group_for = {index:group for group in groups for index in group}
    for index in np.asarray(kept).reshape(-1):
        x, y, w, h = candidates[int(index)]
        box = Box(x, y, w, h)
        members = [(i, line) for i, line in enumerate(lines)
                   if overlap(box, line) >= .5*line.w*line.h]
        if members:
            members = [(i,line) for i,line in members if i not in assigned]
            if not members:
                continue
        vertical = None
        if members:
            vertical = sum(line.w*line.h * (1 if line.vertical else -1)
                           for _, line in members) > 0
            assigned.update(i for i, _ in members)
            matched_groups = []
            for i,_ in members:
                if group_for[i] not in matched_groups:
                    matched_groups.append(group_for[i])
            if len(matched_groups) > 1 and vertical:
                # A model block may cover several adjacent balloons. Use line
                # spacing at the original stroke scale before adding margins.
                for group in matched_groups:
                    parts = [lines[index] for index in group]
                    assigned.update(group)
                    left,top = min(p.x for p in parts),min(p.y for p in parts)
                    right,bottom = max(p.x+p.w for p in parts),max(p.y+p.h for p in parts)
                    direction = sum(p.w*p.h*(1 if p.vertical else -1) for p in parts)>0
                    result.append(Box(left,top,right-left,bottom-top,direction))
                continue
            # Sparse horizontal handwriting may have words absent from the line
            # map. Keep its full model block instead of cropping those words out.
            indices = [i for group in matched_groups for i in group]
            members = [(i,lines[i]) for i in indices]
            assigned.update(indices)
            vertical = sum(line.w*line.h*(1 if line.vertical else -1)
                           for _,line in members)>0
            # Include the complete detected lines when a block clips a character.
            x = min(x, *(line.x for _, line in members))
            y = min(y, *(line.y for _, line in members))
            right = max(box.x+box.w, *(line.x+line.w for _, line in members))
            bottom = max(box.y+box.h, *(line.y+line.h for _, line in members))
            w, h = right-x, bottom-y
        result.append(Box(x, y, w, h, vertical))
    # Recover standalone text lines missed by the block head, without CRAFT.
    for group in groups:
        parts = [lines[i] for i in group if i not in assigned]
        if parts:
            x,y = min(p.x for p in parts),min(p.y for p in parts)
            result.append(Box(x,y,max(p.x+p.w for p in parts)-x,
                max(p.y+p.h for p in parts)-y,
                sum(p.w*p.h*(1 if p.vertical else -1) for p in parts)>0))
    # Keep breathing room around the outer strokes for downstream line detection
    # and recognition, just as the existing detector does.
    padded = []
    for box in result:
        x, y = max(0, box.x-4), max(0, box.y-4)
        right, bottom = min(width, box.x+box.w+4), min(height, box.y+box.h+4)
        padded.append(Box(x, y, right-x, bottom-y, box.vertical))
    return suppress(padded)


def cuda_session(path):
    # Importing torch first exposes its CUDA/cuDNN DLLs on Windows. Its CUDA
    # major version must match the onnxruntime-gpu package in requirements.txt.
    try:
        import torch
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError('CTD GPU 실행에 필요한 패키지가 없습니다. setup.bat을 실행하세요.') from exc
    if not torch.cuda.is_available():
        raise RuntimeError('CTD GPU를 사용할 수 없습니다. NVIDIA 드라이버를 확인하거나 CPU 모드를 선택하세요.')
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('CTD CUDA 실행 기능이 없습니다. setup.bat으로 onnxruntime-gpu를 설치하세요.')
    try:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(str(path), sess_options=options, providers=[
            ('CUDAExecutionProvider', {'device_id': torch.cuda.current_device(),
                'cudnn_conv_algo_search': 'HEURISTIC', 'cudnn_conv_use_max_workspace': '0',
                'arena_extend_strategy': 'kSameAsRequested', 'use_tf32': '0'})])
        # ORT may quietly construct a CPU-only session when CUDA DLL loading fails.
        if not session.get_providers() or session.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('CUDA 실행 세션을 만들지 못했습니다.')
        session.disable_fallback()
        return session
    except Exception as exc:
        raise RuntimeError('CTD GPU 초기화에 실패했습니다. setup.bat·NVIDIA 드라이버를 확인하거나 '
                           'OCR 처리 장치를 CPU 모드로 변경하세요.') from exc


class ComicTextDetector:
    def __init__(self, root, status=lambda _: None, stopped=lambda: False, *, device='cpu'):
        if device not in ('cpu', 'cuda'):
            raise ValueError('지원하지 않는 CTD 실행 장치입니다.')
        self.device = device
        self.session = None
        path = model_path(root, status, stopped)
        check_stopped(stopped)
        status(f'Comic Text Detector {"GPU" if device == "cuda" else "CPU"} 모델 로딩 중…')
        if device == 'cuda':
            self.session = cuda_session(path)
            self.input_name = self.session.get_inputs()[0].name
            self.outputs = [output.name for output in self.session.get_outputs()]
        else:
            self.net = cv2.dnn.readNetFromONNX(str(path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            # OpenCV's default target is CPU (the packaged build has no CUDA DNN).
            self.outputs = self.net.getUnconnectedOutLayersNames()

    def detect_tile(self, pixels):
        height, width = pixels.shape[:2]
        padded = cv2.copyMakeBorder(pixels, 0, INPUT_SIZE-height, 0, INPUT_SIZE-width,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # Input is RGB already. Small crops stay at native size to avoid making
        # handwritten strokes several times larger than the training examples.
        blob = cv2.dnn.blobFromImage(padded, scalefactor=1/255.0)
        if self.session is not None:
            outputs = self.session.run(self.outputs, {self.input_name: blob})
        else:
            self.net.setInput(blob)
            outputs = self.net.forward(self.outputs)
        return decode_outputs(outputs, width, height, pixels)

    def detect(self, pixels, canvas_size=None, stopped=lambda: False):
        height, width = pixels.shape[:2]
        if height == 0 or width == 0:
            return []
        scale = min(1.0, canvas_size/max(height, width)) if canvas_size else 1.0
        working = cv2.resize(pixels, (max(1, round(width*scale)), max(1, round(height*scale)))) if scale < 1 else pixels
        sh, sw = working.shape[:2]
        result = []
        def mapped(box, x=0, y=0, ratio_x=1., ratio_y=1.):
            left = max(0, int(np.floor((box.x*ratio_x+x)*width/sw)))
            top = max(0, int(np.floor((box.y*ratio_y+y)*height/sh)))
            right = min(width, int(np.ceil(((box.x+box.w)*ratio_x+x)*width/sw)))
            bottom = min(height, int(np.ceil(((box.y+box.h)*ratio_y+y)*height/sh)))
            return Box(left, top, right-left, bottom-top, box.vertical)
        if max(sh, sw) > INPUT_SIZE:
            # A whole-page pass supplies complete blocks crossing tile seams.
            ratio = INPUT_SIZE/max(sh, sw)
            overview = cv2.resize(working, (max(1, round(sw*ratio)), max(1, round(sh*ratio))))
            check_stopped(stopped)
            result.extend(mapped(b, ratio_x=sw/overview.shape[1], ratio_y=sh/overview.shape[0])
                          for b in self.detect_tile(overview))
        def positions(length):
            end = max(0, length-INPUT_SIZE)
            return sorted(set([*range(0, end, 768), end]))
        for y in positions(sh):
            for x in positions(sw):
                check_stopped(stopped)
                result.extend(mapped(b, x, y) for b in self.detect_tile(working[y:y+INPUT_SIZE, x:x+INPUT_SIZE]))
        return suppress(result)
