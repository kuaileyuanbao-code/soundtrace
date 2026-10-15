from __future__ import annotations

import ctypes
import json
import re
import subprocess
import urllib.request


HEALTH_URL = "http://127.0.0.1:8767/api/health"
CREATE_NO_WINDOW = 0x08000000


def is_soundtrace() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("service") == "声迹本地转写服务"
    except Exception:
        return False


def listening_pids() -> set[int]:
    completed = subprocess.run(
        ["netstat", "-ano", "-p", "TCP"],
        capture_output=True,
        text=True,
        errors="replace",
        creationflags=CREATE_NO_WINDOW,
    )
    pattern = re.compile(r"^\s*TCP\s+127\.0\.0\.1:8767\s+\S+\s+LISTENING\s+(\d+)\s*$", re.MULTILINE)
    return {int(pid) for pid in pattern.findall(completed.stdout)}


def main() -> None:
    stopped = False
    if is_soundtrace():
        for pid in listening_pids():
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW,
            )
            stopped = True
    message = "声迹本地服务已关闭，显卡资源已经释放。" if stopped else "声迹当前没有在运行。"
    ctypes.windll.user32.MessageBoxW(0, message, "声迹", 0x40)


if __name__ == "__main__":
    main()
