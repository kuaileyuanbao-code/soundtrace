# 声迹 Soundtrace

声迹是一款面向 Windows 的本地语音转写与对话整理软件，支持文件导入、浏览器录音、文字编辑、历史记录和 TXT/JSON 导出。识别使用本机 NVIDIA GPU，音频与模型在本地处理。

## 识别路线

| 模式 | 处理路线 | 输出 |
| --- | --- | --- |
| Text 纯文字转写 | Qwen3-ASR-1.7B | 连续正文 |
| Fast 快速多人识别 | SenseVoiceSmall / FunASR、FSMN-VAD、CAM++ | 文字、人物、时间 |
| Main 标准多人识别 | Qwen3-ASR、ForcedAligner、Sortformer、CAM++ | 文字、人物、时间 |
| Review 多模型精细复核 | Main、MOSS、FireRed、MOVER | 多路复核与融合结果 |

2026-10-20 的 Windows 目录式运行包提供 **Text、Fast、Main**，随包包含 Python、两套环境和七组模型。Review 保留扩展入口，未配置组件时明确提示。开发工作区保留四路线版本。

## Windows 使用

完整解压运行包，双击“打开声迹.bat”，选择模式并导入音频；使用完毕双击“关闭声迹.bat”。运行包可整体移动，启动器自动调整包内环境路径。

验证设备为 Windows、RTX 5070 Laptop GPU（8GB 显存）、32GB 内存；目标机需兼容 CUDA 12.8 的 NVIDIA 驱动及 Edge/Chrome。首次识别包含模型加载，后续任务复用已加载模型。

运行包在同机独立目录、中文和空格路径、空白配置及模型缓存下完成真实 GPU 验证，六次任务结果与开发版本一致。详细环境、计时口径与验证范围见 [Windows 交付说明](delivery/WINDOWS_DELIVERY.md)。另一台电脑的实际部署待验证。

## 源码与测试

本仓库保存应用源码、回归测试、部署工具和组件来源。模型、Python 环境与用户音频通过独立运行包管理。

```powershell
python tools/source_baseline.py --check
.venv/Scripts/python.exe -X utf8 -m unittest discover -s app -p 'test_*.py' -v
.venv/Scripts/python.exe -X utf8 -m unittest discover -s tools -p 'test_windows_delivery.py' -v
node app/test_modes_frontend.cjs
```

应用测试需要开发依赖，部分测试需要 Node.js；GPU 验证记录与自动化回归分别记录。

- [源码清单、检查与导出](delivery/SOURCE_BASELINE.md)
- [真实开发历史、团队贡献及 AI 辅助范围](delivery/DEVELOPMENT_HISTORY.md)
- [组件来源、模型与补丁](delivery/THIRD_PARTY.md)
- [Review 配置与交付范围](delivery/REVIEW_STATUS.md)
- [应用与历史实验说明](app/README.md)

参赛源码及真实提交历史已在 GitHub 公开，评委可直接查看、下载或克隆本仓库。
