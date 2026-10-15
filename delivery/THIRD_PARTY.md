# 第三方源码、模型与本地修改

2026-10-15 来源盘点；不是全软件供应链的最终授权结论。精确源码版本见 `third_party_sources.json`，此仓库不包含第三方 checkout、模型或运行时。

| 组件 | 用途 | 本次记录与处理 |
| --- | --- | --- |
| FireRedASR2S | Review 的 AED 识别 | 固定上游 commit，保留当前两个文件的按需导入补丁及 Apache-2.0 许可副本；不保留模型 |
| MOSS-Transcribe-Diarize | Review 的 MOSS 转写及人物 | 固定 SDK 上游 commit；当前源码未修改，权重另外准备 |
| 3D-Speaker | Main 的 CAM++ | 固定 commit；本机另有不属于当前 CAM++ 路径的 VAD 示例修改，单列记录，不作为声迹必需补丁附带 |
| MeetEval | Review 的 SegLST/fusion 输入 | 固定上游 commit；未修改 |
| MOVER | Review 多路融合 | 仅记录来源、commit、许可 SHA 与阻碍，不复制实现、原生扩展或声称可分发 |
| SpyDER 0.4.1 | 当前 MOVER 的原生依赖 | 来源版本与本地构建状态记录；不附 CPython/MinGW 二进制，仍需正确 ABI 和 DLL |

模型来源与预期许可（代码许可和模型许可分别核对）：

- Qwen3-ASR-1.7B、Qwen3-ForcedAligner-0.6B：Qwen 官方模型，先前核对 Apache-2.0。
- NVIDIA diar_streaming_sortformer_4spk-v2.1：NVIDIA Open Model License，应携带其协议与所需 NOTICE。
- CAM++ advanced / Fast CAM++ / FSMN-VAD：分别对应 ModelScope 模型，版本和模型许可需以实际快照归档，不混用两份 CAM++。
- SenseVoiceSmall：需核对实际模型快照的 FunASR 模型条款，不能仅因 Python 代码 Apache 就把权重一律标为 Apache。
- MOSS、FireRed AED：先前核对模型卡为 Apache-2.0，发布时仍需锁定完整快照并附许可证。

本轮没有为大模型文件重新做全量 SHA 或下载锁定，所以模型 id 不是已完成的可复现模型 lock。Python 环境观察也不等于所有 wheel 与原生运行时已通过分发审查。

## FireRed 补丁

来源：`https://github.com/FireRedTeam/FireRedASR2S.git`，commit `4e7d9aaf4482a47cec1724807026b9b151926eb5`。

补丁：`patches/FireRedASR2S-aed-lazy-imports.patch`。声迹于 2026-10-14 修改了：

- `fireredasr2s/__init__.py`：完整系统类按访问导入；
- `fireredasr2s/fireredasr2/asr.py`：LLM 组件仅在 LLM 路径导入。

未更换 AED 模型、权重或推理参数。补丁与三份历史回归参照中涉及上游 FireRed 的代码适用 `licenses/FireRedASR2S-LICENSE.txt`。保留上游版权和许可，修改来源须随补丁说明一并提供。清单还保存修改后文件的字节 SHA，补丁应用后的 Git blob 内容与本机按规范化换行比较一致；原工作文件的 CRLF/LF 记录以清单为准。

在锁定的上游源码目录中可以先用 `git apply --check` 检查，再由后续依赖准备流程应用；本轮未对本机生产 checkout 重复应用补丁，也未重新安装依赖。

## 外部许可和权属

源码仓库未自动设定自有代码的统一开源许可证。原算法赛 `ifly_tcpwer` 子集按已有成果保存，团队应确认其权属及本次工程增量。私人录音、声纹和比赛数据均不因此获得传播许可。

其他 Python 库、GPU 运行时、DLL、模型 NOTICE 仍须在制作运行包时完成汇总；本目录不是完整 SBOM 的替代。Review 的独立阻碍见 [REVIEW_STATUS.md](REVIEW_STATUS.md)。
