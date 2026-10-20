# 第三方源码、模型与本地修改

2026-10-15 来源盘点，2026-10-20 更新交付情况。精确源码版本见 `third_party_sources.json`；源码仓库保存来源与补丁，运行包另附相应依赖、模型和许可。

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

2026-10-20 的三模式运行包已固定七组实际模型文件，并在 PACKAGE_MANIFEST.json 中记录文件散列；licenses/models 保存模型许可，ENVIRONMENTS.json 保存两套随包环境的依赖名称与版本。

## FireRed 补丁

来源：`https://github.com/FireRedTeam/FireRedASR2S.git`，commit `4e7d9aaf4482a47cec1724807026b9b151926eb5`。

补丁：`patches/FireRedASR2S-aed-lazy-imports.patch`。声迹于 2026-10-14 修改了：

- `fireredasr2s/__init__.py`：完整系统类按访问导入；
- `fireredasr2s/fireredasr2/asr.py`：LLM 组件仅在 LLM 路径导入。

未更换 AED 模型、权重或推理参数。补丁与三份历史回归参照中涉及上游 FireRed 的代码适用 `licenses/FireRedASR2S-LICENSE.txt`。保留上游版权和许可，修改来源须随补丁说明一并提供。清单还保存修改后文件的字节 SHA，补丁应用后的 Git blob 内容与本机按规范化换行比较一致；原工作文件的 CRLF/LF 记录以清单为准。

在锁定的上游源码目录中可以先用 `git apply --check` 检查，再由后续依赖准备流程应用；本轮未对本机生产 checkout 重复应用补丁，也未重新安装依赖。

## 外部许可和权属

源码仓库未自动设定自有代码的统一开源许可证。原算法赛 `ifly_tcpwer` 子集按已有成果保存，团队应确认其权属及本次工程增量。私人录音、声纹和比赛数据均不因此获得传播许可。

Windows 包保留 Python、依赖目录中的许可证，以及 NVIDIA 模型 NOTICE 和模型协议；包根 THIRD_PARTY.md 汇总本次分发组件。源码与运行包关系见 [WINDOWS_DELIVERY.md](WINDOWS_DELIVERY.md)，Review 范围见 [REVIEW_STATUS.md](REVIEW_STATUS.md)。
