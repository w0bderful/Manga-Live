"""OpenCV candidate regions and text/balloon geometry refinement."""
import cv2
import numpy as np
from core import Box, text_boxes


def opencv_boxes(pixels, canvas_size=None):
    height, width = pixels.shape[:2]
    scale = 1.0 if canvas_size is None else min(1.0, canvas_size/max(height, width))
    small = pixels if scale == 1.0 else cv2.resize(
        pixels, (max(1, round(width*scale)), max(1, round(height*scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    accepted = np.zeros(count, dtype=bool)
    max_glyph = max(80, min(256, round(max(gray.shape)*0.12)))
    for index, (x, y, w, h, area) in enumerate(stats[1:count], start=1):
        if (2 <= h <= max_glyph and 2 <= w <= max_glyph and area >= 3
                and 0.04 <= area/(w*h) <= 0.95 and max(w, h) <= 12*min(w, h)):
            accepted[index] = True
    # Copy only accepted components, not unrelated picture strokes inside their bounds.
    glyphs = np.where(accepted[labels], 255, 0).astype(np.uint8)

    grouped = cv2.dilate(glyphs, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    horizontal = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ink = np.count_nonzero(glyphs[y:y+h, x:x+w])
        if w >= 6 and h >= 6 and ink >= 12:
            horizontal.append([x/scale, (x+w)/scale, y/scale, (y+h)/scale])
    return text_boxes(horizontal, [], width, height)


def verified_opencv_boxes(candidates, text_regions):
    """Keep text detector separation; expand only with a closely matching CV candidate."""
    result = []
    for text in text_regions:
        best, best_score = None, 0.75
        for candidate in candidates:
            overlap_w = max(0, min(text.x+text.w, candidate.x+candidate.w)-max(text.x, candidate.x))
            overlap_h = max(0, min(text.y+text.h, candidate.y+candidate.h)-max(text.y, candidate.y))
            overlap = overlap_w*overlap_h
            union = text.w*text.h + candidate.w*candidate.h - overlap
            score = overlap/union if union else 0
            if score > best_score:
                best, best_score = candidate, score
        if best is None:
            result.append(text)
        else:
            x, y = min(text.x, best.x), min(text.y, best.y)
            result.append(Box(x, y, max(text.x+text.w, best.x+best.w)-x,
                              max(text.y+text.h, best.y+best.h)-y, text.vertical))
    return result


def split_panel_regions(pixels, regions):
    """Keep a detected region from spanning a straight, dark comic panel border."""
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    result = []
    for box in regions:
        x1, y1, x2, y2 = box.crop()
        crop = gray[y1:y2, x1:x2]
        cuts = []
        # A panel divider crosses almost the whole text region; individual glyphs do not.
        if box.h >= 40:
            columns = np.flatnonzero(np.mean(crop < 80, axis=0) >= .9)
            # Treat the two strokes of a narrow panel gutter as one divider.
            for run in np.split(columns, np.flatnonzero(np.diff(columns) > 5)+1):
                if (len(run) and run[-1]-run[0]+1 <= max(8, box.w*.06)
                        and run[0] >= 12 and box.w-run[-1]-1 >= 12):
                    cuts.append((int(run[0]), int(run[-1])+1))
        left = 0
        for start, end in cuts:
            if start-left < 12:
                continue
            result.append(Box(x1+left, y1, start-left, box.h, box.vertical))
            left = end
        result.append(Box(x1+left, y1, box.w-left, box.h, box.vertical))
    return result


def balloon_text_boxes(horizontal, free, pixels):
    """Group nearby text lines without joining across dark balloon/panel outlines."""
    height, width = pixels.shape[:2]
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    ink = cv2.morphologyEx((gray < 180).astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((3,3),np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1-ink, 8)
    regions = {}

    def interior(box):
        if box not in regions:
            x1,y1,x2,y2 = box.crop()
            histogram = np.bincount(labels[y1:y2,x1:x2].ravel(),minlength=count)
            histogram[0] = 0
            label = int(histogram.argmax())
            regions[box] = label if (histogram[label] > box.w*box.h*.35
                and stats[label,cv2.CC_STAT_AREA] >= 200) else None
        return regions[box]

    def same_enclosed_region(a,b):
        label = interior(a)
        if label is None or label != interior(b):
            return False
        x,y,w,h,area = stats[label]
        return (0 < x and 0 < y and x+w < width and y+h < height
                and area < width*height*.025)

    def can_merge(a, b):
        first,second = interior(a),interior(b)
        if first is not None and second is not None and first != second:
            return False
        if same_enclosed_region(a,b):
            return True
        xo = min(a.x+a.w, b.x+b.w)-max(a.x,b.x)
        yo = min(a.y+a.h, b.y+b.h)-max(a.y,b.y)
        overlap = max(0,xo)*max(0,yo)
        if overlap >= .35*min(a.w*a.h,b.w*b.h):
            return True
        row = (a.w >= a.h and b.w >= b.h and xo > .55*min(a.w,b.w)
               and -yo < .65*min(a.h,b.h))
        col = (a.h > a.w and b.h > b.w and yo > .55*min(a.h,b.h)
               and abs(a.y-b.y) < .3*min(a.h,b.h)
               and -xo < .65*min(a.w,b.w))
        inline = (a.w >= a.h and b.w >= b.h and yo > .7*min(a.h,b.h)
                  and -xo < .65*min(a.h,b.h))
        stack = (xo > .6*min(a.w,b.w) and max(a.w,b.w) < 2*min(a.w,b.w)
                 and -yo < .5*min(a.w,b.w))
        if stack and yo > 0:
            return True
        if not (row or col or stack or inline):
            return False
        if col or inline:
            left,right = sorted((a,b),key=lambda box:box.x)
            x1,x2 = max(left.x,left.x+left.w-6),min(right.x+right.w,right.x+6)
            y1,y2 = max(a.y,b.y)+3,min(a.y+a.h,b.y+b.h)-3
            gap = gray[y1:y2,x1:x2]
            barrier = gap.size and np.mean(np.any(gap<120,axis=1)) > .55
        else:
            top,bottom = sorted((a,b),key=lambda box:box.y)
            y1,y2 = max(top.y,top.y+top.h-6),min(bottom.y+bottom.h,bottom.y+6)
            x1,x2 = max(a.x,b.x)+3,min(a.x+a.w,b.x+b.w)-3
            gap = gray[y1:y2,x1:x2]
            barrier = gap.size and np.mean(np.any(gap<120,axis=0)) > .55
        return not barrier

    result = text_boxes(horizontal,[],width,height,can_merge=can_merge)
    # The bounding rectangle of an angled note includes empty corners. It must
    # not act as a bridge between the neighboring upright dialogue columns.
    for angled in text_boxes([],free,width,height,merge=False):
        for index,box in enumerate(result):
            xo = max(0,min(box.x+box.w,angled.x+angled.w)-max(box.x,angled.x))
            yo = max(0,min(box.y+box.h,angled.y+angled.h)-max(box.y,angled.y))
            if (same_enclosed_region(box,angled)
                    or xo*yo >= .7*angled.w*angled.h and can_merge(box,angled)):
                x,y = min(box.x,angled.x),min(box.y,angled.y)
                result[index] = Box(x,y,max(box.x+box.w,angled.x+angled.w)-x,
                                    max(box.y+box.h,angled.y+angled.h)-y,box.vertical)
                break
        else:
            result.append(angled)
    return sorted(result,key=lambda box:(box.y,-box.x))


def fit_balloon_regions(pixels, regions):
    """Trim small detection overhangs to enclosed white interiors; remove duplicates."""
    height,width = pixels.shape[:2]
    gray = cv2.cvtColor(pixels,cv2.COLOR_RGB2GRAY)
    ink = cv2.morphologyEx((gray<180).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,_ = cv2.connectedComponentsWithStats(1-ink,8)
    fitted = []
    for box in regions:
        x1,y1,x2,y2 = box.crop()
        histogram = np.bincount(labels[y1:y2,x1:x2].ravel(),minlength=count)
        histogram[0] = 0
        label = int(histogram.argmax())
        x,y,w,h,area = (int(value) for value in stats[label])
        if (histogram[label] > box.w*box.h*.35 and 200 <= area < width*height*.025
                and 0 < x and 0 < y and x+w < width and y+h < height):
            left,top,right,bottom = max(x1,x),max(y1,y),min(x2,x+w),min(y2,y+h)
            if (right-left)*(bottom-top) >= box.w*box.h*.75:
                box = Box(left,top,right-left,bottom-top,box.vertical)
        fitted.append(box)
    # Small detector fragments already inside a larger text crop must not get a
    # second translation on top of the same sentence.
    result = []
    for box in sorted(fitted,key=lambda b:b.w*b.h,reverse=True):
        if any(max(0,min(box.x+box.w,b.x+b.w)-max(box.x,b.x))
               *max(0,min(box.y+box.h,b.y+b.h)-max(box.y,b.y)) >= box.w*box.h*.9
               for b in result):
            continue
        result.append(box)
    return sorted(result,key=fitted.index)
