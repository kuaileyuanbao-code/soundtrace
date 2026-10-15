from __future__ import annotations

import ctypes
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = APP_DIR / "server.py"
LOGS_DIR = APP_DIR / "logs"
LAUNCHER_LOG = LOGS_DIR / "launcher.log"
SERVER_LOG = LOGS_DIR / "server.log"
SERVER_ERROR_LOG = LOGS_DIR / "server-error.log"
HEALTH_URL = "http://127.0.0.1:8767/api/health"
APP_URL = "http://127.0.0.1:8767/"
CREATE_NO_WINDOW = 0x08000000


def log(message: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with LAUNCHER_LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now():%Y-%m-%d %H:%M:%S.%f} {message}\n")


def service_ready() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return bool(payload.get("ok"))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def show_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(0, message, "声迹启动失败", 0x10)


def start_server() -> None:
    if not PYTHON.exists():
        raise FileNotFoundError(f"没有找到本地 Python 环境：{PYTHON}")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with SERVER_LOG.open("ab") as stdout, SERVER_ERROR_LOG.open("ab") as stderr:
        process = subprocess.Popen(
            [str(PYTHON), str(SERVER)],
            cwd=str(PROJECT_ROOT),
            stdout=stdout,
            stderr=stderr,
            creationflags=CREATE_NO_WINDOW,
        )
    log(f"server process requested; pid={process.pid}")
    for _ in range(180):
        if service_ready():
            log("service became ready")
            return
        if process.poll() is not None:
            raise RuntimeError(f"本地算法服务提前退出，退出码 {process.returncode}。")
        time.sleep(0.5)
    raise TimeoutError("本地算法服务在 90 秒内没有启动，请查看 app\\logs\\server-error.log。")


def open_app_window() -> None:
    candidates = [
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
    ]
    edge = next((path for path in candidates if path.is_file()), None)
    if edge:
        subprocess.Popen(
            [str(edge), f"--app={APP_URL}", "--start-maximized"],
            creationflags=CREATE_NO_WINDOW,
        )
        log(f"Edge app window requested: {edge}")
    else:
        os.startfile(APP_URL)
        log("default browser window requested")


def main() -> None:
    log(f"pythonw launcher started; pid={os.getpid()}")
    if service_ready():
        log("service already ready")
    else:
        start_server()
    open_app_window()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc!r}")
        show_error(str(exc))
