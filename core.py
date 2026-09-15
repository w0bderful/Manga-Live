from dataclasses import dataclass
import numpy as np
import cv2


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    def crop(self):
        return self.x, self.y, self.x + self.w, self.y + self.h


def relocate(box, before, after):

    if before is None or after is None:
        return None
    x, y, x2, y2 = box.crop()
    if x < 0 or y < 0 or x2 > before.shape[1] or y2 > before.shape[0]:
        return None
    patch = before[y:y2, x:x2]
    if patch.size == 0 or box.w > after.shape[1] or box.h > after.shape[0]:
        return None
    if y2 <= after.shape[0] and x2 <= after.shape[1] and not changed(patch, after[y:y2, x:x2]):
        return box
    template = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    if template.std() < 8:
        return None
    left, right = max(0, x-40), min(after.shape[1], x2+40)
    search = cv2.cvtColor(after[:, left:right], cv2.COLOR_RGB2GRAY)
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (nx, ny) = cv2.minMaxLoc(scores)
    if score < 0.94:
        return None
    scores[max(0, ny-box.h//2):ny+box.h//2+1,
           max(0, nx-box.w//2):nx+box.w//2+1] = -1
    if float(scores.max()) > score-0.04:
        return None
    return Box(left+nx, ny, box.w, box.h)


def merge_row(rows, incoming):
    box, _ = incoming
    def overlaps(other):
        xo = max(0, min(box.x+box.w, other.x+other.w)-max(box.x, other.x))
        yo = max(0, min(box.y+box.h, other.y+other.h)-max(box.y, other.y))
        return xo*yo > 0.4*min(box.w*box.h, other.w*other.h)
    return [row for row in rows if not overlaps(row[0])] + [incoming]


def scroll_offset(before, after, ignored=()):

    if before is None or after is None or before.shape != after.shape:
        return None
    height, width = before.shape[:2]
    mask = np.full((height, width), 255, np.uint8)
    for b in ignored:
        mask[max(0, b.y-14):max(0, b.y+b.h+14), max(0, b.x-14):max(0, b.x+b.w+14)] = 0
    old = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    new = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    shift, confidence = cv2.phaseCorrelate(old.astype(np.float32), new.astype(np.float32))
    if confidence > 0.15 and np.isfinite(shift).all():
        px, py = np.rint(shift).astype(int)
        if abs(px) <= 40 and 1 <= abs(py) < height*0.8:
            transform = np.float32([[1, 0, px], [0, 1, py]])
            shifted = cv2.warpAffine(before, transform, (width, height))
            visible = (cv2.warpAffine(mask, transform, (width, height)) & mask) > 0
            error = np.max(np.abs(shifted.astype(np.int16)-after.astype(np.int16)), axis=2)
            if visible.sum() >= 100 and np.mean(error[visible] > 35) <= 0.2:
                return int(px), int(py)
    points = cv2.goodFeaturesToTrack(old, 240, 0.03, 10, mask=mask)
    if points is None or len(points) < 6:
        return None
    found, status, _ = cv2.calcOpticalFlowPyrLK(old, new, points, None, winSize=(31, 31), maxLevel=4)
    if found is None:
        return None
    source, target = points.reshape(-1, 2), found.reshape(-1, 2)
    valid = status.ravel().astype(bool)
    for index, (x, y) in enumerate(target):
        if not (0 <= x < width and 0 <= y < height):
            valid[index] = False
        elif mask[int(y), int(x)] == 0:
            valid[index] = False
    movement = (target-source)[valid]
    if len(movement) < 6:
        return None
    median = np.median(movement, axis=0)
    inliers = np.linalg.norm(movement-median, axis=1) < 2.5
    if inliers.sum() < 6 or inliers.mean() < 0.65:
        return None
    dx, dy = np.rint(np.median(movement[inliers], axis=0)).astype(int)
    if abs(dx) > 40 or abs(dy) < 1 or abs(dy) >= height*0.8:
        return None
    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(before, transform, (width, height))
    visible = (cv2.warpAffine(mask, transform, (width, height)) & mask) > 0
    if visible.sum() < 100:
        return None
    error = np.max(np.abs(shifted.astype(np.int16)-after.astype(np.int16)), axis=2)
    if np.mean(error[visible] > 35) > 0.2:
        return None
    return int(dx), int(dy)


def move_rows(rows, offset, shape):
    height, width = shape[:2]
    dx, dy = offset
    moved = [(Box(b.x+dx, b.y+dy, b.w, b.h), text) for b, text in rows]
    return [(b, text) for b, text in moved
            if b.x < width and b.y < height and b.x+b.w > 0 and b.y+b.h > 0]


def restore_occluded(before, captured, boxes, offset):

    height, width = captured.shape[:2]
    dx, dy = offset
    shifted = cv2.warpAffine(before, np.float32([[1, 0, dx], [0, 1, dy]]),
                             (width, height), borderValue=(255, 255, 255))
    clean = captured.copy()
    for b in boxes:
        ys = slice(max(0, b.y-14), max(0, b.y+b.h+14))
        xs = slice(max(0, b.x-14), max(0, b.x+b.w+14))
        clean[ys, xs] = shifted[ys, xs]
    return clean


def changed(before, after, threshold=0.008, ignored=()):
    if before is None or before.shape != after.shape:
        return True
    delta = np.abs(before.astype(np.int16) - after.astype(np.int16))
    different = np.max(delta, axis=2) > 25
    visible = np.ones(different.shape, dtype=bool)
    for box in ignored:
        visible[max(0, box.y-12):max(0, box.y+box.h+12),
                max(0, box.x-12):max(0, box.x+box.w+12)] = False
    return bool(visible.any() and float(np.mean(different[visible])) > threshold)


def text_boxes(horizontal, free, width, height):
    raw = [(b[0], b[2], b[1], b[3]) for b in horizontal]
    for poly in free:
        points = np.asarray(poly)
        raw.append((*points.min(axis=0), *points.max(axis=0)))
    boxes = []
    for x1, y1, x2, y2 in raw:
        x1, y1 = max(0, int(x1)-3), max(0, int(y1)-3)
        x2, y2 = min(width, int(x2)+4), min(height, int(y2)+4)
        if x2-x1 >= 8 and y2-y1 >= 8:
            boxes.append(Box(x1, y1, x2-x1, y2-y1))
    while True:
        merged = False
        for i, a in enumerate(boxes):
            for j in range(i+1, len(boxes)):
                b = boxes[j]
                xo = min(a.x+a.w, b.x+b.w)-max(a.x, b.x)
                yo = min(a.y+a.h, b.y+b.h)-max(a.y, b.y)
                row = (a.w >= a.h and b.w >= b.h and
                       xo > 0.55*min(a.w, b.w) and -yo < 0.65*min(a.h, b.h))
                col = (a.h > a.w and b.h > b.w and
                       yo > 0.55*min(a.h, b.h) and -xo < 0.65*min(a.w, b.w))
                overlap = xo > 0 and yo > 0
                if row or col or overlap:
                    x, y = min(a.x, b.x), min(a.y, b.y)
                    boxes[i] = Box(x, y, max(a.x+a.w, b.x+b.w)-x,
                                   max(a.y+a.h, b.y+b.h)-y)
                    boxes.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            return sorted(boxes, key=lambda b: (b.y, -b.x))
