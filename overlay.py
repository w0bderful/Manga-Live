"""Korean text layout and transparent translated-region rendering."""
import unicodedata
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (
    QColor, QFont, QFontMetricsF, QPainter,
    QPainterPath, QPen, QRegion, QImage,
    QBitmap, QTextLayout, QTextOption, QTextCharFormat,
)
from PyQt6.QtWidgets import QWidget
from overlay_settings import DEFAULT_TEXT_STYLE


def vertical_text_layout(text, rect, family='Malgun Gothic', max_size=23, bold=False):
    characters = list(unicodedata.normalize('NFC', ' '.join(text.split())))
    font = QFont(family)
    font.setBold(bold)
    if not characters or rect.width() <= 0 or rect.height() <= 0:
        return font, []
    for size in range(max_size, 0, -1):
        font.setPixelSize(size)
        metrics = QFontMetricsF(font)
        cell_width = max(metrics.height(), *(max(metrics.horizontalAdvance(char),
                                                 metrics.boundingRect(char).width())
                                             for char in characters))
        cell_height = metrics.height()
        rows = int(rect.height() // cell_height)
        columns = int(rect.width() // cell_width)
        if rows and columns and rows * columns >= len(characters):
            break
    else:
        return font, []
    used_columns = (len(characters) + rows - 1) // rows
    right = rect.center().x() + used_columns * cell_width / 2
    top = rect.center().y() - min(rows, len(characters)) * cell_height / 2
    cells = []
    for index, char in enumerate(characters):
        column, row = divmod(index, rows)
        cell = QRectF(right - (column + 1) * cell_width, top + row * cell_height,
                      cell_width, cell_height)
        cells.append((char, cell))
    return font, cells


def horizontal_text_layout(text, rect, family='Malgun Gothic', max_size=23, bold=False, outlined=False):
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\u2028')
    font = QFont(family)
    font.setBold(bold)
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    option.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    for size in range(max_size, 0, -1):
        font.setPixelSize(size)
        layout = QTextLayout(text, font)
        layout.setTextOption(option)
        if outlined:
            char_format = QTextCharFormat()
            char_format.setForeground(QColor('white'))
            char_format.setTextOutline(QPen(QColor('#151515'), min(2.5, size*.12),
                                           Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            span = QTextLayout.FormatRange()
            span.start, span.length, span.format = 0, len(text.encode('utf-16-le'))//2, char_format
            layout.setFormats([span])
        layout.beginLayout()
        height = 0.0
        fits = True
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max(0.0, rect.width()))
            line.setPosition(QPointF(0, height))
            height += line.height()
            fits = fits and line.naturalTextWidth() <= rect.width() + 1e-6
        layout.endLayout()
        if fits and height <= rect.height():
            return layout, QPointF(rect.x(), rect.y() + (rect.height()-height)/2)
    return layout, QPointF(rect.x(), rect.y())


def overlay_rectangles(rows, sx=1., sy=1.):
    """Reserve disjoint display space without changing the OCR source boxes."""
    rects = [QRectF(box.x*sx, box.y*sy, box.w*sx, box.h*sy)
             if text.strip() else QRectF() for box, text in rows]
    # Stable ordering keeps placement independent of translation completion order.
    order = sorted(range(len(rows)), key=lambda i: (
        rects[i].y(), rects[i].x(), rects[i].width(), rects[i].height(), rows[i][1]))
    for position, i in enumerate(order):
        for j in order[position+1:]:
            a, b = rects[i], rects[j]
            if not a.intersects(b):
                continue
            options = []
            for horizontal in (True, False):
                start = (lambda r: r.left()) if horizontal else (lambda r: r.top())
                end = (lambda r: r.right()) if horizontal else (lambda r: r.bottom())
                low, high = sorted((i, j), key=lambda k: start(rects[k])+end(rects[k]))
                first, second = rects[low], rects[high]
                # Divide the shared strip; both new rectangles remain inside
                # their original regions. Shrinking cannot create new overlaps.
                cut = (max(start(first), start(second))+min(end(first), end(second)))/2
                gap = min(2., (cut-start(first))/2, (end(second)-cut)/2)
                first_size = cut-gap/2-start(first)
                second_size = end(second)-cut-gap/2
                retained = (first_size/(end(first)-start(first)),
                            second_size/(end(second)-start(second)))
                options.append((min(retained), sum(retained), horizontal, low, high, cut, gap))
            _, _, horizontal, low, high, cut, gap = max(options, key=lambda item: item[:2])
            if horizontal:
                rects[low].setRight(cut-gap/2)
                rects[high].setLeft(cut+gap/2)
            else:
                rects[low].setBottom(cut-gap/2)
                rects[high].setTop(cut+gap/2)
    return rects


class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                            Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput |
                            Qt.WindowType.WindowDoesNotAcceptFocus)


        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.text_style = dict(DEFAULT_TEXT_STYLE)
        self.rows = []
        self.source_size = (1, 1)
        self.rendered = QImage()
        self.setMask(QRegion(-2, -2, 1, 1))

    def clear(self):
        self.rows = []
        self.rendered = QImage()
        self.setMask(QRegion(-2, -2, 1, 1))
        self.update()

    def display(self, rows, source_size):
        self.rows = rows
        self.source_size = source_size
        layer = QImage(self.size(), QImage.Format.Format_RGBA8888)
        layer.fill(0)
        painter = QPainter(layer)
        background_region = QRegion()
        sx, sy = self.width()/source_size[0], self.height()/source_size[1]
        for rect, (_, text) in zip(overlay_rectangles(rows, sx, sy), rows):
            if text.strip():
                background_region |= QRegion(rect.toAlignedRect())
        opacity = self.text_style['background_opacity']
        if opacity:
            painter.save()
            painter.setClipRegion(background_region)
            painter.fillRect(layer.rect(), QColor(255,255,255,round(255*opacity/100)))
            painter.restore()
        self.paint_text(painter)
        painter.end()


        self.rendered = layer
        mask = background_region if opacity else QRegion(QBitmap.fromImage(self.rendered.createAlphaMask()))
        self.setMask(mask if not mask.isEmpty() else QRegion(-2, -2, 1, 1))
        self.show()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.rendered)

    def paint_text(self, painter):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        sx, sy = self.width()/self.source_size[0], self.height()/self.source_size[1]
        for rect, (box, text) in zip(overlay_rectangles(self.rows, sx, sy), self.rows):
            padding = min(4, rect.width() / 4, rect.height() / 4)
            inner = rect.adjusted(padding, padding, -padding, -padding)
            if inner.width() <= 0 or inner.height() <= 0:
                continue

            vertical = box.vertical if box.vertical is not None else box.h > box.w * 1.25
            family, max_size = self.text_style['font_family'], self.text_style['font_size']
            if vertical:
                font, cells = vertical_text_layout(text, inner, family, max_size, True)
                painter.save()
                painter.setClipRect(rect)
                painter.setFont(font)
                painter.setPen(QPen(QColor('#151515'), min(2.5, font.pixelSize()*.12),
                                    Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                painter.setBrush(QColor('white'))
                for char, cell in cells:
                    path = QPainterPath()
                    path.addText(QPointF(0,0), font, char)
                    bounds = path.boundingRect()
                    path.translate(cell.center().x()-bounds.center().x(), cell.center().y()-bounds.center().y())
                    painter.drawPath(path)
                    painter.save()
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawPath(path)
                    painter.restore()
                painter.restore()
                continue

            layout, position = horizontal_text_layout(text, inner, family, max_size, True, outlined=True)
            painter.save()
            painter.setClipRect(rect)
            layout.draw(painter, position)
            fill = QTextCharFormat()
            fill.setForeground(QColor('white'))
            fill.setTextOutline(QPen(Qt.PenStyle.NoPen))
            span = QTextLayout.FormatRange()
            span.start, span.length, span.format = 0, len(layout.text().encode('utf-16-le'))//2, fill
            layout.draw(painter, position, [span])
            painter.restore()
