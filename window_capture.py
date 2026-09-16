import ctypes
from ctypes import wintypes as w
import multiprocessing
import os
import time

import mss
from mss.exception import ScreenShotError
import numpy as np


class CaptureProtectionError(RuntimeError):
    pass


def check_blank_capture(pixels):
    if pixels.size and int(pixels.max()) <= 8 and int(pixels.max()) - int(pixels.min()) <= 2:
        raise CaptureProtectionError(
            '캡처 결과가 검은 화면입니다. 브라우저 화면 보호로 차단되었을 가능성이 있습니다. '
            '선택 영역이 실제로 검은 화면인지도 확인하세요.')


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
    def __init__(self, owner_pid=None):
        self.owner_pid = os.getpid() if owner_pid is None else owner_pid
        self.user = u = ctypes.WinDLL('user32', use_last_error=True)
        self.gdi = g = ctypes.WinDLL('gdi32', use_last_error=True)
        self.dwm = ctypes.WinDLL('dwmapi')
        self.callback_type = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        u.EnumWindows.argtypes = [self.callback_type, w.LPARAM]
        u.IsWindowVisible.argtypes = u.IsIconic.argtypes = [w.HWND]
        u.GetWindowRect.argtypes = [w.HWND, ctypes.POINTER(w.RECT)]
        u.GetWindowThreadProcessId.argtypes = [w.HWND, ctypes.POINTER(w.DWORD)]
        u.GetWindowDisplayAffinity.argtypes = [w.HWND, ctypes.POINTER(w.DWORD)]
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
            own = pid.value == self.owner_pid
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

    def check_protection(self, bounds, layers):
        uncovered = np.ones((bounds[3]-bounds[1], bounds[2]-bounds[0]), dtype=bool)
        for hwnd, box, own in layers:
            overlap = intersection(bounds, box)
            if own or overlap is None:
                continue
            x1, y1, x2, y2 = overlap
            visible = uncovered[y1-bounds[1]:y2-bounds[1], x1-bounds[0]:x2-bounds[0]]
            if not visible.any():
                continue
            affinity = w.DWORD()
            if self.user.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity)) and affinity.value & 1:
                raise CaptureProtectionError('선택 영역에 Windows 화면 캡처 보호가 적용된 창이 있습니다.')
            visible[:] = False
            if not uncovered.any():
                break

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


def capture_worker(connection, owner_pid):
    try:
        user = ctypes.WinDLL('user32')
        user.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))
        native = NativeWindows(owner_pid)
        with mss.MSS() as screen:
            while True:
                region = connection.recv()
                if region is None:
                    return
                try:
                    x, y, width, height = (region[k] for k in ('left', 'top', 'width', 'height'))
                    bounds = (x, y, x+width, y+height)
                    layers = native.layers(bounds)
                    native.check_protection(bounds, layers)
                    if any(own and intersection(bounds, box) == bounds for _, box, own in layers):
                        pixels = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        try:
                            pixels = np.asarray(screen.grab(region))[:, :, 2::-1].copy()
                        except ScreenShotError:
                            pixels = np.zeros((height, width, 3), dtype=np.uint8)
                            layers.insert(0, (0, bounds, True))
                    pixels = restore_under_windows(pixels, bounds, layers, native.render)
                    check_blank_capture(pixels)
                    connection.send((True, pixels))
                except CaptureProtectionError as exc:
                    connection.send((False, {'type': 'capture_protection', 'message': str(exc)}))
                except Exception as exc:
                    connection.send((False, str(exc)))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send((False, f'캡처 초기화 실패: {exc}'))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class CaptureWithoutApp:
    def __init__(self):
        self.pending = None
        self.closed = False
        self.version = 0
        self.process = None
        self.connection = None

    def start_worker(self):
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe()
        process = context.Process(target=capture_worker, args=(child, os.getpid()),
                                  daemon=True, name='window-capture')
        try:
            process.start()
        except Exception:
            parent.close()
            raise
        finally:
            child.close()
        self.connection, self.process = parent, process

    def stop_worker(self):
        self.pending = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            # PrintWindow can block indefinitely; a separate process can be stopped safely.
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=0.2)
            if not self.process.is_alive():
                self.process.close()
            self.process = None

    def grab(self, region):
        if self.closed:
            return None
        try:
            if self.pending is not None:
                previous, started, version = self.pending
                if previous != region or version != self.version:
                    self.stop_worker()
                elif self.connection.poll():
                    success, result = self.connection.recv()
                    self.pending = None
                    if not success:
                        if isinstance(result, dict) and result.get('type') == 'capture_protection':
                            raise CaptureProtectionError(result['message'])
                        raise RuntimeError(result)
                    return result
                elif not self.process.is_alive():
                    raise RuntimeError('캡처 작업이 종료되었습니다. 다시 번역을 눌러 재시도하세요.')
                elif time.monotonic()-started > 5:
                    raise RuntimeError('아래 창의 캡처 응답이 지연됩니다. 대상 앱을 확인하고 다시 번역을 누르세요.')
                else:
                    return None
            if self.process is None:
                self.start_worker()
            self.connection.send(dict(region))
            self.pending = (dict(region), time.monotonic(), self.version)
            return None
        except Exception:
            self.stop_worker()
            raise

    def invalidate(self):
        self.version += 1
        if self.pending is not None:
            self.stop_worker()

    def close(self):
        self.closed = True
        self.stop_worker()
