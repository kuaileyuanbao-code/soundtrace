"""Build a Windows directory distribution from the verified local environments.

No package installation, original-environment mutation, or private audio copy.
MOVER and Review model/environment files are excluded from this distribution.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", "__pycache__", ".cache"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(source, target):
    if not source.is_dir():
        raise FileNotFoundError(source)
    result = subprocess.run(
        ["robocopy", str(source), str(target), "/E", "/COPY:DAT", "/DCOPY:DAT", "/R:1", "/W:1",
         "/MT:8", "/XJ", "/XD", *sorted(EXCLUDED_DIRS), "/XF", "*.pyc", "*.pyo",
         "/NFL", "/NDL", "/NJH", "/NJS", "/NP"], capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode >= 8:
        raise RuntimeError(f"Copy failed ({result.returncode}): {source}\n{result.stdout[-2000:]!r}")
    print("Copied " + str(target), flush=True)


def source_files():
    return subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")[:-1]


def write_text(path, text, *, batch=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8") if batch else text.encode("utf-8"))


def build(output, python_base, license_dir):
    output = output.resolve()
    expected_parent = (ROOT / "delivery/dist").resolve()
    if output.parent != expected_parent or output.exists():
        raise ValueError("Output must be a new direct child of delivery/dist; existing directories are preserved")
    if shutil.disk_usage(expected_parent.parent).free < 35 * 1024**3:
        raise RuntimeError("至少需要 35GiB 可用空间以准备运行目录与压缩文件。")
    for required in ["NVIDIA-Open-Model-License.html", "FunASR-MODEL-LICENSE.txt"]:
        if not (license_dir / required).is_file():
            raise FileNotFoundError(license_dir / required)
    output.mkdir(parents=True)
    selected = source_files()
    for name in selected:
        # Historic regression source copies and developer-only checks stay in the repository.
        if name.startswith("app/validation/") or Path(name).name.startswith("test_"):
            continue
        source = ROOT / name
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"Unexpected source entry: {name}")
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(ROOT / "tools/portable_launcher.py", output / "portable.py")
    shutil.copy2(ROOT / "tools/verify_windows_package.py", output / "verify_package.py")
    for name in ("build_windows_delivery.py", "portable_launcher.py", "verify_windows_package.py"):
        destination = output / "delivery/build_tools" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "tools" / name, destination)
    copy_tree(python_base, output / "runtime/python")
    copy_tree(ROOT / ".venv", output / ".venv")
    copy_tree(ROOT / "tmp/audit_repro_full_20260911/.venv_augmented", output / "runtime/envs/main")
    copy_tree(ROOT / "vendor/3D-Speaker/speakerlab", output / "vendor/3D-Speaker/speakerlab")
    shutil.copy2(ROOT / "vendor/3D-Speaker/LICENSE", output / "vendor/3D-Speaker/LICENSE")
    # The original Fast route resolves these exact snapshots from ModelScope.
    models = ROOT / "project/user_data/model_data"
    sources = {
        "Qwen3-ASR-1.7B": models / "Qwen3-ASR-1.7B",
        "Qwen3-ForcedAligner-0.6B": models / "Qwen3-ForcedAligner-0.6B",
        "diar_streaming_sortformer_4spk-v2.1": models / "diar_streaming_sortformer_4spk-v2.1",
        "3dspeaker/iic/speech_campplus_sv_zh_en_16k-common_advanced": models / "3dspeaker/iic/speech_campplus_sv_zh_en_16k-common_advanced",
        "SenseVoiceSmall": Path.home() / ".cache/modelscope/iic/SenseVoiceSmall",
        "fsmn-vad": Path.home() / ".cache/modelscope/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "funasr-campp": Path.home() / ".cache/modelscope/iic/speech_campplus_sv_zh-cn_16k-common",
    }
    for name, source in sources.items():
        copy_tree(source, output / "project/user_data/model_data" / name)
    copy_tree(license_dir, output / "licenses/models")
    apache = ROOT / "vendor/3D-Speaker/LICENSE"
    shutil.copy2(apache, output / "licenses/Apache-2.0.txt")
    sortformer = output / "project/user_data/model_data/diar_streaming_sortformer_4spk-v2.1"
    shutil.copy2(license_dir / "NVIDIA-Open-Model-License.html", sortformer / "LICENSE.html")
    write_text(sortformer / "Notice.txt", "Licensed by NVIDIA Corporation under the NVIDIA Open Model License\n")
    for name in ("Qwen3-ASR-1.7B", "Qwen3-ForcedAligner-0.6B", "fsmn-vad", "funasr-campp",
                 "3dspeaker/iic/speech_campplus_sv_zh_en_16k-common_advanced"):
        shutil.copy2(apache, output / "project/user_data/model_data" / name / "LICENSE.txt")
    shutil.copy2(license_dir / "FunASR-MODEL-LICENSE.txt", output / "project/user_data/model_data/SenseVoiceSmall/MODEL_LICENSE.txt")
    # Make the first interpreter independent immediately; venv homes are rebased at every launch.
    subprocess.run([str(output / "runtime/python/python.exe"), "-I", str(output / "portable.py"), "prepare"],
                   check=True, cwd=output, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    write_text(output / "打开声迹.bat", '@echo off\nstart "" "%~dp0runtime\\python\\pythonw.exe" "%~dp0portable.py" start\n', batch=True)
    write_text(output / "关闭声迹.bat", '@echo off\n"%~dp0runtime\\python\\python.exe" "%~dp0portable.py" stop\nif errorlevel 1 pause\n', batch=True)
    write_text(output / "诊断启动.bat", '@echo off\n"%~dp0runtime\\python\\python.exe" "%~dp0portable.py" start\nif errorlevel 1 pause\n', batch=True)
    write_text(output / "README.md", """# 声迹 Windows 本地运行包

本包提供纯文字转写 Text、快速多人识别 Fast 和标准多人识别 Main，包含 Python、两套运行环境及相应模型。精细复核 Review 保留扩展入口，其组件未随本包配置。

## 开始使用

1. 完整解压至本机磁盘，建议为目录预留 30GB 空间。
2. 双击“打开声迹.bat”，在弹出的浏览器窗口中导入音频或使用麦克风。
3. 选择模式并开始识别，完成后可编辑、保存历史及导出结果。
4. 使用完毕双击“关闭声迹.bat”释放服务和显卡资源。

首次识别需要加载模型；界面会显示准备与处理进度。后续任务可复用已加载模型。移动整个目录后再次使用同一启动文件即可重新配置包内环境。

环境要求与操作说明见 DEPLOYMENT.md，组件来源及许可见 THIRD_PARTY.md。模型识别在本机执行，运行时使用包内权重。
""")
    write_text(output / "DEPLOYMENT.md", """# 声迹部署与环境说明

## 运行环境

Windows 10/11 x64、兼容 CUDA 12.8 的 NVIDIA 驱动，建议至少 8GB 显存、32GB 系统内存。验证设备为 RTX 5070 Laptop GPU。包内使用 CPython 3.11.15 与 PyTorch 2.11.0+cu128，服务环境和 Main 环境保持隔离。

包内包含 Python、依赖及模型，启动过程使用本地文件；驱动和 Edge/Chrome 由操作系统环境提供。浏览器麦克风录音需要授予网站及 Windows 麦克风权限。

## 目录与启动

- runtime/python：随包 Python；启动器重定位 .venv 与 runtime/envs/main。
- .venv：服务、Fast 与声纹处理环境。
- runtime/envs/main：Text/Main 环境。
- project/user_data/model_data：七组模型或配置目录。
- app：页面、服务与处理代码。
- licenses：模型及第三方许可证；Python 和各依赖目录保留原许可证文件。

双击“打开声迹.bat”启动；默认地址为 http://127.0.0.1:8767/。启动失败时运行“诊断启动.bat”，查看 app/logs/server.log。端口被其他应用占用时，先关闭对应服务；命令行可使用 runtime/python/python.exe portable.py start --port 18768 指定其他端口。

运行目录需要可写权限。上传的音频、结果和日志保存到 app/jobs、app/voiceprints、app/logs；浏览器保存最近 20 条识别结果。复制或分享整个运行目录前，请自行移出这些使用中生成的数据。

## 模式与资源

Text 输出连续正文；Fast/Main 输出文字、人物和时间。Review 的扩展依赖未配置，选择该模式时界面明确提示。默认模型、识别参数及融合调用代码保留原设计。

首次模型加载和切换耗时随设备及内存情况变化。主要识别流程使用单队列处理 GPU 任务，并保留 CPU 缓存与内存保护机制。

## 文件核对

PACKAGE_MANIFEST.json 记录包内文件的大小和 SHA256。解压后可用随包 Python 执行 verify_package.py 校验。VALIDATION.md 列出本次真实验证及测试环境。
""")
    write_text(output / "THIRD_PARTY.md", """# 组件与模型来源

本包保留 Python、安装依赖与模型目录中的原许可证和版权记录，模型许可另存于 licenses/models。精确 Python 依赖见 ENVIRONMENTS.json，文件散列见 PACKAGE_MANIFEST.json。

| 组件 | 用途 | 来源及许可 |
| --- | --- | --- |
| CPython 3.11.15 | 包内运行时 | python.org / Astral python-build-standalone；原 LICENSE.txt 随包 |
| Qwen3-ASR-1.7B | Text/Main 文字识别 | https://huggingface.co/Qwen/Qwen3-ASR-1.7B ，Apache-2.0 |
| Qwen3-ForcedAligner-0.6B | Main 时间对齐 | https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B ，Apache-2.0 |
| Sortformer 4spk v2.1 | Main 人物活动 | https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1 ，NVIDIA Open Model License |
| 3D-Speaker / advanced CAM++ | Main 声纹 | https://github.com/modelscope/3D-Speaker ，Apache-2.0 |
| SenseVoiceSmall | Fast 文字识别 | https://modelscope.cn/models/iic/SenseVoiceSmall ，保留本地模型卡及 FunASR 模型协议 |
| FSMN-VAD | Fast 语音活动 | https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch ，Apache-2.0 |
| CAM++ | Fast 声纹 | https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common ，Apache-2.0 |

SenseVoice、FSMN-VAD、CAM++ 来自已验证的本地 ModelScope 快照，其文件散列随包固定。运行参数保持原值，部署代码新增包内路径配置、运行环境重定位和实例识别。

已有算法赛的 ifly_tcpwer 模块提供说话人处理基础，Main 主体架构及相关说话人模块来自刘泽睿的方案；声迹在此基础上进行四路线组织、界面交互、编辑导出、模型生命周期管理和系统验证。原有贡献记录见 delivery/DEVELOPMENT_HISTORY.md。

源码中的 Review 调用接口保留；本包不含 MOVER 实现、其原生构建、MOSS/FireRed 模型或 MOSS 环境。MOVER 项目来源：https://github.com/nttcslab-sp/mover 。
""")
    envs = {}
    for key, rel in (("service", ".venv"), ("main", "runtime/envs/main")):
        metadata = output / rel / "Lib/site-packages"
        envs[key] = sorted(p.name for p in metadata.glob("*.dist-info"))
    write_text(output / "ENVIRONMENTS.json", json.dumps(envs, ensure_ascii=False, indent=2))
    build_info = {"created_at": datetime.now().astimezone().isoformat(),
                  "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "working_changes": subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines(),
                  "enabled_modes": ["text", "fast", "main"], "review_components_included": False,
                  "private_user_data_included": False}
    write_text(output / "BUILD.json", json.dumps(build_info, ensure_ascii=False, indent=2))
    print("BUILD READY " + str(output), flush=True)


def finalize(output):
    """Manifest a clean candidate; user-created data are never deleted by this tool."""
    if any((output / "app" / name).exists() for name in ("jobs", "voiceprints", "logs")):
        raise RuntimeError("Candidate contains runtime data; move test data to the validation directory before finalizing")
    for site in output.glob("**/sitecustomize.py"):
        if "SOUNDTRACE_TEST_GUARD" in site.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError("Temporary validation guard remains: " + str(site))
    blocked = [p for p in output.rglob("*") if p.is_symlink()
               or bool(getattr(p.lstat(), "st_file_attributes", 0) & 0x400)]
    if blocked:
        raise RuntimeError(f"External links are not permitted: {blocked[:3]}")
    for relative in ("research/third_party/mover", ".venv_moss", "models/MOSS-Transcribe-Diarize"):
        if (output / relative).exists():
            raise RuntimeError("Unexpected Review implementation/model: " + relative)
    paths = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "PACKAGE_MANIFEST.json")
    def record(path):
        return {"path": path.relative_to(output).as_posix(), "size": path.stat().st_size,
                "sha256": sha256(path), "mutable_at_launch": path.name == "pyvenv.cfg"}
    files = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for entry in pool.map(record, paths):
            files.append(entry)
            if len(files) % 10000 == 0:
                print(f"Hashed {len(files)}/{len(paths)} files", flush=True)
    write_text(output / "PACKAGE_MANIFEST.json", json.dumps({"files": files, "bytes": sum(x["size"] for x in files)}, ensure_ascii=False, indent=2))
    print(f"Manifest: {len(files)} files; {sum(x['size'] for x in files)/1024**3:.3f} GiB", flush=True)


def archive(output):
    target = output.with_suffix(".zip")
    if target.exists():
        raise FileExistsError(target)
    sevenzip = shutil.which("7z") or next((str(p) for p in (
        Path("C:/Program Files/7-Zip/7z.exe"), Path("C:/Program Files (x86)/7-Zip/7z.exe")) if p.is_file()), None)
    if sevenzip:
        subprocess.run([sevenzip, "a", "-tzip", "-mx=1", "-mmt=4", "-bso0", "-bsp0",
                        str(target), output.name], cwd=output.parent, check=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        subprocess.run([sevenzip, "t", "-bso0", "-bsp0", str(target)], check=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    z.write(path, output.name + "/" + path.relative_to(output).as_posix())
        with zipfile.ZipFile(target) as z:
            if z.testzip() is not None:
                raise RuntimeError("ZIP integrity verification failed")
    write_text(target.with_suffix(".zip.sha256"), sha256(target) + "  " + target.name + "\n")
    print(f"ARCHIVE {target} {target.stat().st_size/1024**3:.3f} GiB", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "finalize", "archive"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-base", type=Path)
    parser.add_argument("--license-dir", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        if not args.python_base or not args.license_dir:
            parser.error("build requires --python-base and --license-dir")
        build(args.output, args.python_base, args.license_dir)
    elif args.action == "finalize":
        finalize(args.output.resolve())
    else:
        archive(args.output.resolve())


if __name__ == "__main__":
    main()
