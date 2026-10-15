"""CPU parking and memory guard for the existing Main/FunASR owners."""
import os
import ctypes
from performance import model_state

GIB = 1024 ** 3
FAST_COMPONENTS = ("model", "vad_model", "punc_model", "spk_model", "cb_model")


def tensor_bytes(models):
    seen = set()
    size = 0
    for model in models:
        if model is None:
            continue
        for tensor in list(model.parameters()) + list(model.buffers()):
            if id(tensor) not in seen:
                seen.add(id(tensor))
                size += tensor.numel() * tensor.element_size()
    return size


def cpu_move_groups(module, limit=512 * 1024 ** 2):
    """Partition large module trees; parent cleanup comes after its moved children.

    Fresh-byte estimates exclude those children on the parent pass. A single
    oversized leaf cannot be split without changing its tensor layout.
    """
    children = list(module.children())
    total = tensor_bytes([module])
    if total <= limit or not children:
        return [(module, total)]
    groups = []
    for child in children:
        groups.extend(cpu_move_groups(child, limit))
    own = list(module.parameters(recurse=False)) + list(module.buffers(recurse=False))
    groups.append((module, sum(t.numel() * t.element_size() for t in own)))
    return groups


def fast_state(owner):
    return {name: model_state(getattr(owner, name, None)) for name in FAST_COMPONENTS}


def move_fast(owner, device):
    """Never infer while parked; keep FunASR's original runtime/base kwargs intact."""
    before = fast_state(owner)
    for name in FAST_COMPONENTS:
        module = getattr(owner, name, None)
        if module is not None and hasattr(module, "to"):
            observed = before[name]["devices"]
            if observed and observed == [str(device)]:
                continue
            module.to(device)
    after = fast_state(owner)
    if str(device) == "cpu" and any(
        d != "cpu" for item in after.values() for d in item["devices"]
    ):
        raise RuntimeError("快速模型未完整释放设备权重，不能启动另一条 GPU 路线。")
    return {"before": before, "after": after}


def memory_budget(required_bytes=0):
    """Fail closed if RAM cannot be observed; this is a guard, not an allocation guarantee."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        configured = os.environ.get("SOUNDTRACE_CACHE_MIN_FREE_GB")
        reserve = max(4 * GIB, int(memory.total * .15)) if configured is None else max(0, float(configured)) * GIB
        enabled = os.environ.get("SOUNDTRACE_MODEL_REUSE", "1") != "0"
        commit_free = None
        if os.name == "nt":
            class PerformanceInfo(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in (
                    "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
                    "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
                    (name, ctypes.c_ulong) for name in ("HandleCount", "ProcessCount", "ThreadCount")]
            info = PerformanceInfo()
            info.cb = ctypes.sizeof(info)
            if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
                raise OSError("Cannot observe Windows commit headroom")
            commit_free = (info.CommitLimit - info.CommitTotal) * info.PageSize
        return {"available_bytes": memory.available, "total_bytes": memory.total,
                "commit_available_bytes": commit_free,
                "reserve_bytes": int(reserve), "commit_reserve_bytes": GIB, "required_bytes": int(required_bytes),
                "can_retain": enabled and memory.available >= reserve + required_bytes
                              and (commit_free is None or commit_free >= GIB + required_bytes),
                "reuse_enabled": enabled}
    except (ImportError, ValueError, OSError) as exc:
        return {"can_retain": False, "reason": str(exc)}
