"""Process-only CPU and Windows GPU counters, collected off the UI thread."""
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import math
import os
import threading
import time


@dataclass(frozen=True)
class ResourceUsage:
    cpu: float | None = None
    gpu: float | None = None
    vram: float | None = None

    def text(self):
        def percent(value):
            return '—' if value is None else f'{value:.1f}%'
        memory = '—' if self.vram is None else (
            f'{self.vram / 1024**3:.2f} GiB' if self.vram >= 1024**3 else f'{self.vram / 1024**2:.0f} MiB')
        return f'CPU {percent(self.cpu)}  ·  GPU {percent(self.gpu)}  ·  VRAM {memory}'


class _CounterNumber(ctypes.Union):
    _fields_ = [('doubleValue', ctypes.c_double), ('largeValue', ctypes.c_longlong)]


class _CounterValue(ctypes.Structure):
    _anonymous_ = ('number',)
    _fields_ = [('status', wintypes.DWORD), ('number', _CounterNumber)]


class _CounterItem(ctypes.Structure):
    _fields_ = [('name', wintypes.LPWSTR), ('value', _CounterValue)]


def process_counter_values(items, pid):
    prefix = f'pid_{pid}_'
    return [float(value) for name, status, value in items
            if name.startswith(prefix) and status in (0, 1) and math.isfinite(value) and value >= 0]


class WindowsGpuCounters:
    """WDDM counters include CUDA work and dedicated memory for our PID only."""
    def __init__(self):
        self.query = wintypes.HANDLE()
        self.counters = {}
        self.pid = os.getpid()
        self.pdh = ctypes.WinDLL('pdh')
        signatures = {
            'PdhOpenQueryW': [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)],
            'PdhAddEnglishCounterW': [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)],
            'PdhCollectQueryData': [wintypes.HANDLE],
            'PdhGetFormattedCounterArrayW': [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                            ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_CounterItem)],
            'PdhCloseQuery': [wintypes.HANDLE],
        }
        for name, args in signatures.items():
            function = getattr(self.pdh, name)
            function.argtypes = args
            function.restype = wintypes.DWORD
        if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)):
            raise OSError('GPU 성능 카운터를 열지 못했습니다.')
        try:
            for name, path in [('gpu', r'\GPU Engine(*)\Utilization Percentage'),
                               ('vram', r'\GPU Process Memory(*)\Dedicated Usage')]:
                counter = wintypes.HANDLE()
                if self.pdh.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(counter)) == 0:
                    self.counters[name] = counter
            self.pdh.PdhCollectQueryData(self.query)
        except Exception:
            self.close()
            raise

    def values(self, name):
        counter = self.counters.get(name)
        if counter is None:
            return []
        # Instances can appear between sizing and reading the array.
        for _ in range(3):
            size, count = wintypes.DWORD(), wintypes.DWORD()
            status = self.pdh.PdhGetFormattedCounterArrayW(counter, 0x200, ctypes.byref(size), ctypes.byref(count), None)
            if status != 0x800007D2 or not 0 < size.value <= 32 * 1024**2:  # PDH_MORE_DATA
                return []
            buffer = ctypes.create_string_buffer(size.value)
            array = ctypes.cast(buffer, ctypes.POINTER(_CounterItem))
            status = self.pdh.PdhGetFormattedCounterArrayW(counter, 0x200, ctypes.byref(size), ctypes.byref(count), array)
            if status == 0:
                return process_counter_values(
                    ((array[i].name or '', array[i].value.status, array[i].value.doubleValue)
                     for i in range(count.value)), self.pid)
            if status != 0x800007D2:
                return []
        return []

    def sample(self):
        if self.pdh.PdhCollectQueryData(self.query):
            return None, None
        gpu, vram = self.values('gpu'), self.values('vram')
        # Like Task Manager: busiest engine for this process; memory across adapters.
        return (min(100.0, max(gpu)) if gpu else None, sum(vram) if vram else None)

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = wintypes.HANDLE()


class ProcessSampler:
    def __init__(self):
        self.previous = (time.monotonic(), time.process_time())
        self.cpu_count = max(1, os.cpu_count() or 1)
        try:
            self.gpu = WindowsGpuCounters()
        except (OSError, AttributeError):
            self.gpu = None

    def sample(self):
        now, used = time.monotonic(), time.process_time()
        elapsed, consumed = now-self.previous[0], used-self.previous[1]
        self.previous = (now, used)
        cpu = max(0.0, min(100.0, consumed / elapsed / self.cpu_count * 100)) if elapsed > 0 else None
        try:
            gpu, vram = self.gpu.sample() if self.gpu else (None, None)
        except (OSError, ValueError):
            gpu, vram = None, None
        return ResourceUsage(cpu, gpu, vram)

    def close(self):
        if self.gpu:
            self.gpu.close()


class ResourceMonitor:
    def __init__(self, sampler_factory=ProcessSampler):
        self.sampler_factory = sampler_factory
        self.condition = threading.Condition()
        self.interval = 3.0
        self.stopped = False
        self.latest = ResourceUsage()
        self.thread = threading.Thread(target=self.run, daemon=True, name='resource-usage')

    def start(self):
        self.thread.start()

    def set_active(self, active):
        interval = 1.0 if active else 3.0
        with self.condition:
            if interval != self.interval:
                self.interval = interval
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return self.latest

    def run(self):
        sampler = None
        try:
            sampler = self.sampler_factory()
            last = time.monotonic()
            while True:
                with self.condition:
                    while not self.stopped:
                        delay = self.interval - (time.monotonic()-last)
                        if delay <= 0:
                            break
                        self.condition.wait(delay)
                    if self.stopped:
                        return
                try:
                    result = sampler.sample()
                except Exception:
                    result = ResourceUsage()
                with self.condition:
                    self.latest = result
                last = time.monotonic()
        except Exception:
            with self.condition:
                self.latest = ResourceUsage()
        finally:
            if sampler is not None:
                sampler.close()

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
