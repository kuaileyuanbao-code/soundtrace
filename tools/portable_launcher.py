"""Relocatable Windows launcher; the build places this file at the package root."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def instance_id(root=ROOT):
    return hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:20]


def prepare(root=ROOT):
    """Rebase only the copied environments, including after moving the folder."""
    base = root / "runtime/python"
    if not (base / "python.exe").is_file():
        raise FileNotFoundError("运行包缺少 runtime/python/python.exe，请完整解压全部文件。")
    for relative in (".venv", "runtime/envs/main", ".venv_moss"):
        env = root / relative
        if not env.exists():
            continue
        if not env.resolve().is_relative_to(root.resolve()):
            raise RuntimeError("运行环境必须位于当前声迹目录内。")
        if not (env / "Scripts/python.exe").is_file():
            raise FileNotFoundError(f"运行环境不完整：{relative}")
        (env / "pyvenv.cfg").write_text(
            f"home = {base}\ninclude-system-site-packages = false\n"
            f"version = 3.11.15\nexecutable = {base / 'python.exe'}\n", encoding="utf-8")
    environ = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH"):
        environ.pop(name, None)
    environ.update(
        PYTHONUTF8="1", PYTHONNOUSERSITE="1",
        SOUNDTRACE_INSTANCE_ID=instance_id(root),
        SOUNDTRACE_MAIN_PYTHON=str(root / "runtime/envs/main/Scripts/python.exe"),
        SOUNDTRACE_FAST_ASR_MODEL=str(root / "project/user_data/model_data/SenseVoiceSmall"),
        SOUNDTRACE_FAST_VAD_MODEL=str(root / "project/user_data/model_data/fsmn-vad"),
        SOUNDTRACE_FAST_SPEAKER_MODEL=str(root / "project/user_data/model_data/funasr-campp"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MODELSCOPE_OFFLINE="1",
        HF_HOME=str(root / "runtime/cache/huggingface"),
        MODELSCOPE_CACHE=str(root / "runtime/cache/modelscope"),
        MS_CACHE_HOME=str(root / "runtime/cache/modelscope"),
    )
    return environ


def health(port):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/api/health", timeout=2) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def start(port, browser):
    environ = prepare()
    current = health(port)
    if current and current.get("instance_id") == instance_id():
        if browser:
            open_browser(port)
        return
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"端口 {port} 正被其他程序使用。请先关闭另一个声迹窗口对应的服务，或使用 --port 指定其他端口。")
    logs = ROOT / "app/logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "server.log").open("ab") as output:
        process = subprocess.Popen(
            [str(ROOT / ".venv/Scripts/python.exe"), "-X", "utf8", "-m", "uvicorn",
             "server:app", "--app-dir", str(ROOT / "app"), "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=environ, stdout=output, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
    state = ROOT / "runtime" / f"server-{port}.json"
    state.write_text(json.dumps({"pid": process.pid, "instance_id": instance_id(), "port": port}), encoding="utf-8")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        current = health(port)
        if current and current.get("instance_id") == instance_id():
            print(f"声迹已启动：http://127.0.0.1:{port}/", flush=True)
            if browser:
                open_browser(port)
            return
        if process.poll() is not None:
            raise RuntimeError("服务启动失败，请查看 app/logs/server.log。")
        time.sleep(.5)
    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
    raise TimeoutError("服务启动超时，请查看 app/logs/server.log。")


def stop(port):
    state = ROOT / "runtime" / f"server-{port}.json"
    if not state.is_file():
        print("当前目录没有运行中的声迹服务记录。")
        return
    saved = json.loads(state.read_text(encoding="utf-8"))
    current = health(port)
    if not current or current.get("instance_id") != instance_id() or saved.get("instance_id") != instance_id():
        raise RuntimeError("当前端口未匹配本目录的声迹服务，已保留其他程序。")
    # Verify that the saved redirector PID still belongs to this environment.
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.QueryFullProcessImageNameW.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, int(saved["pid"]))
        if not handle:
            raise RuntimeError("无法核对声迹进程，请通过任务管理器检查。")
        try:
            size = ctypes.c_ulong(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                raise RuntimeError("无法读取声迹进程路径。")
            if Path(buffer.value).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
                raise RuntimeError("进程路径与本目录不一致，已保留该进程。")
        finally:
            kernel.CloseHandle(handle)
    subprocess.run(["taskkill", "/PID", str(int(saved["pid"])), "/T", "/F"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
    state.unlink(missing_ok=True)
    print("声迹服务已关闭。")


def open_browser(port):
    url = f"http://127.0.0.1:{port}/"
    for variable in ("PROGRAMFILES(X86)", "PROGRAMFILES"):
        edge = Path(os.environ.get(variable, "")) / "Microsoft/Edge/Application/msedge.exe"
        if edge.is_file():
            subprocess.Popen([str(edge), f"--app={url}", "--start-maximized"], creationflags=NO_WINDOW)
            return
    os.startfile(url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "start", "stop"), nargs="?", default="start")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("端口范围为 1024–65535")
    if args.action == "prepare":
        prepare()
        print("本目录运行环境已就绪。")
    elif args.action == "start":
        start(args.port, not args.no_browser)
    else:
        stop(args.port)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        if os.name == "nt" and Path(sys.executable).stem.lower() == "pythonw":
            ctypes.windll.user32.MessageBoxW(0, str(exc), "声迹", 0x10)
        raise SystemExit(1)
