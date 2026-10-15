"""Lightweight wall timers. CUDA boundary synchronization is opt-in only."""
from contextlib import contextmanager
import os
import inspect
import sys
import time


def model_state(model):
    """Inspect actual tensor devices, without moving weights or launching CUDA work."""
    if model is None:
        return {"state": "unloaded", "devices": []}
    try:
        devices = {str(p.device) for p in model.parameters()}
        devices.update(str(b.device) for b in model.buffers())
    except (AttributeError, TypeError):
        return {"state": "unknown", "devices": []}
    if devices and all(d.startswith("cuda") for d in devices):
        state = "gpu_resident"
    elif devices == {"cpu"}:
        state = "cpu_loaded"
    else:
        state = "mixed" if devices else "unknown"
    return {"state": state, "devices": sorted(devices)}


class StageTimings:
    def __init__(self):
        self.values = {}
        self.calls = {}
        self.sync_enabled = os.environ.get("SOUNDTRACE_PROFILE_SYNC", "0") == "1"
        self.sync_seconds = 0.0
        self.sync_calls = 0

    def sync(self):
        torch = sys.modules.get("torch")
        if self.sync_enabled and torch is not None and torch.cuda.is_initialized():
            started = time.perf_counter()
            torch.cuda.synchronize()
            self.sync_seconds += time.perf_counter() - started
            self.sync_calls += 1

    @contextmanager
    def measure(self, name, gpu=False):
        if gpu:
            self.sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            if gpu:
                self.sync()
            self.values[name] = self.values.get(name, 0.0) + time.perf_counter() - started
            self.calls[name] = self.calls.get(name, 0) + 1

    def seconds(self):
        return {k: round(v, 6) for k, v in self.values.items()}

    def info(self):
        return {"clock": "perf_counter", "gpu_timing": "synchronized_wall" if self.sync_enabled else "wall_no_forced_sync",
                "cuda_sync_seconds": round(self.sync_seconds, 6), "cuda_sync_calls": self.sync_calls,
                "stage_calls": dict(self.calls)}


@contextmanager
def observe_call(target, name, timer, stage, gpu=False):
    """Time the original method, preserving arguments, return values and exceptions."""
    original = getattr(target, name)
    own = name in vars(target)
    descriptor = vars(target).get(name)
    callable_descriptor = inspect.getattr_static(target, name)

    def wrapped(*args, **kwargs):
        with timer.measure(stage, gpu=gpu):
            return original(*args, **kwargs)

    # Class/static methods are already bound by getattr(); ordinary class
    # methods still need Python's instance binding (e.g. nn.Module.cuda).
    is_bound_class_callable = isinstance(target, type) and isinstance(callable_descriptor, (classmethod, staticmethod))
    setattr(target, name, staticmethod(wrapped) if is_bound_class_callable else wrapped)
    try:
        yield
    finally:
        if own:
            setattr(target, name, descriptor)
        else:
            delattr(target, name)
