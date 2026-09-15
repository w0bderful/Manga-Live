import ctypes
from ctypes import wintypes as w
from concurrent.futures import Future
import os
import queue
import threading
import time

import mss
from mss.exception import ScreenShotError
import numpy as np


def intersection(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    return (x, y, right, bottom) if right > x and bottom > y else None


class BitmapInfo(ctypes.Structure):
    _fields_ = [('size', w.DWORD), ('width', w.LONG), ('height', w.LONG),
                ('planes', w.WORD), ('bit_count', w.WORD), ('compression', w.DWORD),
                ('image_size', w.DWORD), ('xppm', w.LONG), ('yppm', w.LONG),
                ('colors', w.DWORD), ('important', w.DWORD)]


class NativeWindows:
    def __init__(self):
        self.user = u = ctypes.WinDLL('user32', use_last_error=True)
        self.gdi = g = ctypes.WinDLL('gdi32', use_last_error=True)
        self.dwm = ctypes.WinDLL('dwmapi')
        self.callback_type = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        u.EnumWindows.argtypes = [self.callback_type, w.LPARAM]
        u.IsWindowVisible.argtypes = u.IsIconic.argtypes = [w.HWND]
        u.GetWindowRect.argtypes = [w.HWND, ctypes.POINTER(w.RECT)]
        u.GetWindowThreadProcessId.argtypes = [w.HWND, ctypes.POINTER(w.DWORD)]
        u.PrintWindow.argtypes = [w.HWND, w.HDC, w.UINT]
        u.GetDC.argtypes = [w.HWND]
        u.GetDC.restype = w.HDC
        u.ReleaseDC.argtypes = [w.HWND, w.HDC]
        g.CreateCompatibleDC.argtypes = [w.HDC]
        g.CreateCompatibleDC.restype = w.HDC
        g.CreateDIBSection.argtypes = [w.HDC, ctypes.POINTER(BitmapInfo), w.UINT,
                                      ctypes.POINTER(ctypes.c_void_p), w.HANDLE, w.DWORD]
        g.CreateDIBSection.restype = w.HBITMAP
        g.SelectObject.argtypes = [w.HDC, w.HANDLE]
        g.SelectObject.restype = w.HANDLE
        g.DeleteObject.argtypes = [w.HANDLE]
        g.DeleteDC.argtypes = [w.HDC]
        g.CreateRectRgn.argtypes = [ctypes.c_int]*4
        g.CreateRectRgn.restype = w.HRGN
        g.GetRgnBox.argtypes = [w.HRGN, ctypes.POINTER(w.RECT)]
        u.GetWindowRgn.argtypes = [w.HWND, w.HRGN]
        self.dwm.DwmGetWindowAttribute.argtypes = [w.HWND, w.DWORD, ctypes.c_void_p, w.DWORD]

    def layers(self, bounds):
        layers = []
        @self.callback_type
        def visit(hwnd, _):
            if not self.user.IsWindowVisible(hwnd) or self.user.IsIconic(hwnd):
                return True
            cloaked = w.DWORD()
            self.dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
            if cloaked.value:
                return True
            rect, pid = w.RECT(), w.DWORD()
            if not self.user.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True
            box = (rect.left, rect.top, rect.right, rect.bottom)
            self.user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            own = pid.value == os.getpid()
            if own:
                region = self.gdi.CreateRectRgn(0, 0, 0, 0)
                try:
                    if self.user.GetWindowRgn(hwnd, region):
                        extent = w.RECT()
                        self.gdi.GetRgnBox(region, ctypes.byref(extent))
                        box = (rect.left+extent.left, rect.top+extent.top,
                               rect.left+extent.right, rect.top+extent.bottom)
                finally:
                    self.gdi.DeleteObject(region)
            if intersection(bounds, box):
                layers.append((hwnd, box, own))
            return True
        self.user.EnumWindows(visit, 0)
        return layers

    def render(self, hwnd, box):
        width, height = box[2]-box[0], box[3]-box[1]
        info = BitmapInfo(ctypes.sizeof(BitmapInfo), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
        dc = self.user.GetDC(None)
        mem = self.gdi.CreateCompatibleDC(dc)
        bits = ctypes.c_void_p()
        bitmap = self.gdi.CreateDIBSection(dc, ctypes.byref(info), 0, ctypes.byref(bits), None, 0)
        old = None
        try:
            if not mem or not bitmap or not bits.value:
                raise RuntimeError('아래 창 캡처용 메모리를 준비할 수 없습니다.')
            old = self.gdi.SelectObject(mem, bitmap)
            if not self.user.PrintWindow(hwnd, mem, 2):
                raise RuntimeError('아래 창이 이미지 캡처에 응답하지 않습니다.')
            raw = (ctypes.c_ubyte * (width*height*4)).from_address(bits.value)
            return np.ctypeslib.as_array(raw).reshape(height, width, 4)[:, :, 2::-1].copy()
        finally:
            if old:
                self.gdi.SelectObject(mem, old)
            if bitmap:
                self.gdi.DeleteObject(bitmap)
            if mem:
                self.gdi.DeleteDC(mem)
            if dc:
                self.user.ReleaseDC(None, dc)


def restore_under_windows(pixels, bounds, layers, render):
    missing = np.zeros(pixels.shape[:2], dtype=bool)
    def slices(rect, origin):
        return (slice(rect[1]-origin[1], rect[3]-origin[1]),
                slice(rect[0]-origin[0], rect[2]-origin[0]))
    for _, box, own in layers:
        overlap = intersection(bounds, box)
        if own and overlap:
            missing[slices(overlap, bounds)] = True
    if not missing.any():
        return pixels
    for hwnd, box, own in layers:
        overlap = intersection(bounds, box)
        if own or overlap is None:
            continue
        target = slices(overlap, bounds)
        mask = missing[target]
        if not mask.any():
            continue
        source = render(hwnd, box)
        if source.shape[:2] != (box[3]-box[1], box[2]-box[0]):
            raise RuntimeError('캡처 중 아래 창 크기가 변경되었습니다. 다시 시도하세요.')
        pixels[target][mask] = source[slices(overlap, box)][mask]
        missing[target] = False
        if not missing.any():
            return pixels
    raise RuntimeError('Manga Live 아래에서 읽을 수 있는 창을 찾지 못했습니다.')


class CaptureWithoutApp:
    def __init__(self):
        self.jobs = queue.Queue(maxsize=1)
        self.pending = None
        self.closed = False
        self.version = 0
        self.thread = threading.Thread(target=self.run, daemon=True, name='window-capture')
        self.thread.start()

    def run(self):
        user = ctypes.WinDLL('user32')
        user.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        native = NativeWindows()
        with mss.MSS() as screen:
            while True:
                job = self.jobs.get()
                if job is None or self.closed:
                    return
                region, future = job
                try:
                    x, y, width, height = (region[k] for k in ('left', 'top', 'width', 'height'))
                    bounds = (x, y, x+width, y+height)
                    layers = native.layers(bounds)
                    if any(own and intersection(bounds, box) == bounds for _, box, own in layers):
                        pixels = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        try:
                            pixels = np.asarray(screen.grab(region))[:, :, 2::-1].copy()
                        except ScreenShotError:
                            pixels = np.zeros((height, width, 3), dtype=np.uint8)
                            layers.insert(0, (0, bounds, True))
                    future.set_result(restore_under_windows(pixels, bounds, layers, native.render))
                except Exception as exc:
                    future.set_exception(exc)
                if self.closed:
                    return

    def grab(self, region):
        if self.closed:
            return None
        if self.pending is not None:
            previous, future, started, version = self.pending
            if not future.done():
                if time.monotonic()-started > 5:
                    raise RuntimeError('아래 창의 캡처 응답이 지연됩니다. 대상 앱 상태를 확인하세요.')
                return None
            self.pending = None
            if previous == region and version == self.version:
                return future.result()
        future = Future()
        self.pending = (dict(region), future, time.monotonic(), self.version)
        self.jobs.put((dict(region), future))
        return None

    def invalidate(self):
        self.version += 1

    def close(self):
        self.closed = True
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
