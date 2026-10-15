# 声迹本地应用（统一主干 v3）

在项目根目录双击 `打开声迹.bat` 即可使用。启动器会等待本地算法服务就绪，再使用
Edge 的无地址栏应用窗口打开页面；如果没有 Edge，则使用默认浏览器。当前电脑桌面已创建
“声迹”快捷方式。使用结束后可双击 `关闭声迹.bat` 释放显卡和后台进程。

当前版本支持：

- 导入本地音频并显示波形、播放和定位；
- 点击话筒开始录音，再次点击后自动编码为 WAV 并提交识别；
- 导入 SegLST JSON，按 speaker 展示；
- 直接编辑逐段转写文字；
- 导出 UTF-8 SegLST JSON；
- 通过异步任务接口接入本地算法，并显示排队、模型加载和推理阶段；
- 默认“统一识别”，支持短录音和长录音；另有“精细复核”和“快速预览”。
- 自动保存最近 20 条识别历史；只保存文字、时间和说话人，不保存原始录音；
- 记住上次选择的识别模式与人数；
- 提供本机声纹库：录入姓名和单人语音后只保存 192 维 CAM++ 声纹向量，原始录音随即删除；
- 可勾选本次对话的参与成员，识别结果直接显示真实姓名；
- 页面根据所选模式请求准备：Main/Review 预加载主干，Fast/纯文字按需加载，页面显示实际任务阶段；
- 在 8 GB 显卡上复用常驻 GPU 模型，避免每次识别重复搬运全部权重；显存不足时自动减小批次，
  必要时改为分阶段释放显存；精细复核和声纹库处理前释放主干显存；
- 支持安装为 PWA 桌面应用，并提供离线页面外壳。

稳定支持的音频格式为 WAV、MP3、FLAC 和 OGG。采样率及单/双声道不限，后端会统一转换为
16 kHz 单声道；单个文件上限为 512 MB。

本地 FastAPI 服务地址为 `http://127.0.0.1:8767`。它只监听本机地址，音频不会上传到公网。
启动日志保存在 `app/logs/`。

- 统一识别（`main`，默认）：Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B +
  Sortformer v2.1 + CAM++。所有长度都使用同一主干；文字采用 14 秒窗口、2 秒重叠和
  时间归属去重，最多同时处理 8 个窗口；人物原型只在本次录音内建立，不读取其他用户任务或比赛标注。
  使用比赛的 AHC 距离阈值 0.561、长发言门槛 2.1 秒及 Viterbi 连续性参数。
  极短录音没有长发言时，利用现有片段建原型；允许单人录音。指定人数只使用可用声学证据，
  不虚构缺少证据的人物。单路文字不复制给两个同时发言的人。
  指定人数时，如果长发言数量刚好等于人数、却集中在更少的局部轨道，不能据此把同一个人
  强制拆成多个人物原型；在局部轨道声学一致且重叠约束允许时，补用短发言建立人物原型。
  该补充路径的重叠文字优先按主要支持轨道的新发言起点归属，弱额外轨不提前触发换人。
  自动人数、已有充分长发言或轨道证据不一致时仍保留原流程；这不是所有双人录音准确性保证。
- 精细复核（`review`）：先运行上述 Main，再加入 MOSS 的 60 秒/5 秒重叠分块、
  FireRed 的 30 秒/2 秒重叠分块，映射人物后按 MOSS、Main、FireRed 顺序执行三路 MOVER。
  每块文字根据重叠区的时间归属保留，MOSS 局部人物按时间支持匹配到 Main；没有匹配证据的
  人物保留独立身份。该路线耗时更长，属于新接入的产品实验功能。
- 快速预览（`fast`）：保留 SenseVoiceSmall + FSMN-VAD + CAM++，供用户显式选择。
- 纯文字转写（`text`）：页面上传文件与话筒录音均可选择，使用同一 Qwen3-ASR-1.7B、BF16、
  最多 8 个窗口和 768 个生成 token。冷启动仅加载 ASR，不加载 ForcedAligner、Sortformer 或 CAM++。
  与 Main 共用一个持久 worker；已有 Main 时直接复用 Qwen，从纯文字切换 Main 时补加载其他模型。
  20 秒以内整段单次识别，避免不必要接缝；超过 20 秒仍使用原有 14 秒窗口、2 秒重叠，
  按相同文字的后缀/前缀保守去重，保留标点；无法可靠匹配的边界保留原文并警告。
  20 秒是本机真实失败录音、真实短/重复语音、稠密 20 秒及阈值上下测试后的保守策略，
  不是模型输入上限或任意音频的准确率保证。更长单窗口在实验中可运行，但稠密样例耗时增加，暂不扩大范围。
  该匹配不是时间对齐，边界存在重复或误匹配的可能。`text` 是合并后的正文，`chunks` 保留每个窗口原文，
  `merge_boundaries` 记录匹配证据。兼容行的 `speaker=null`、`speaker_analyzed=false`、
  `timestamp_kind=audio_extent`；`start_time=0`、`end_time=音频长度` 只表示整段音频范围，
  不是发言起止或词级时间戳，人物数量为 0（未分析）。不得指定人数或声纹档案。
  通过 `POST /api/jobs` 上传 `audio`、`mode=text` 即可运行；`POST /api/warmup?mode=text`
  返回按需准备，不启动整套 Main；`GET /api/health` 独立报告 text 可用状态。
  页面采用连续文字编辑区，不显示人物、发言时间或片段播放按钮，保留英文空格、大小写、标点和换行。
  历史保存结果类型，重新打开仍为纯文字；顶部提供 UTF-8 TXT 与 JSON 导出。纯文字 JSON 是带
  `mode=text`、`text` 和范围标记的对象，可用“打开结果”重新导入；原三种模式继续导出 SegLST 数组。
  历史依然只保存文字，不保存原音频，长期回听功能不在此次修改范围。
- 个性化声纹：建议每个人在安静环境单独录制 15–60 秒自然说话，只勾选本次真正参与
  对话的成员；系统先聚类，再将聚类中心一对一匹配到声纹库成员；
- 已知人数和声纹姓名匹配同时支持 `main`、`review`，没有 60 秒限制。
- 旧客户端的 `conversation` 请求迁移到 `main`，`accurate` 迁移到 `review`；浏览器旧设置
  自动迁移。历史结果保留原来的模式标记。

本服务复用了比赛 Main 的基础权重与核心逻辑，但不是 clean-v2 完整重放：没有运行比赛
全开发集 LoRA 候选验证、会话选择器、六路改字和五类 clean-v2 修复；不承诺原比赛 0.1448
能直接迁移到任意用户录音。原比赛代码、权重、提交文件和审计包未修改。

新前端使用以下接口：

```text
GET  /api/health
POST /api/warmup?mode=main
POST /api/jobs                  multipart: audio, mode, speaker_count, voiceprint_ids
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/result
GET  /api/voiceprints
POST /api/voiceprints           multipart: name, audio
DELETE /api/voiceprints/{id}
```

为兼容旧页面，`POST /api/transcribe` 仍保留同步快速识别。结果格式为：

```json
{"rows": [{"session_id":"demo","speaker":"spk1","start_time":0,"end_time":1.2,"words":"你 好"}]}
```

所有识别任务按本地 GPU 队列串行运行，避免多个模型同时抢占显存。任务和结果接口返回
`elapsed_seconds`、各阶段 `timings` 与证据不足的 `warnings`；页面显示实际用时。
`gpu_wait_seconds` 包括等待后台预加载/显卡队列的时间；首次打开并不代表模型已经加载完毕。
`/api/health` 返回进程启动耗时和模型批次配置，任务元数据记录实际批次和显存峰值。

速度配置（修改环境变量后重启服务）：`SOUNDTRACE_ASR_BATCH_SIZE=1..8`（默认 8），
`SOUNDTRACE_GPU_RESIDENT=0`（强制按阶段移回 CPU；默认在约 8 GB 及以上显卡启用常驻），
`SOUNDTRACE_CPU_THREADS`（默认 4）。这些配置不更换模型、窗口长度或分人阈值。
首次启动仍需加载本地大模型；本机本轮测得预加载约 49–74 秒，展示前应等待“统一识别模型已就绪”。

统一主干默认使用 `.venv_augmented` 的现有 NeMo/Qwen 环境；可通过
`SOUNDTRACE_MAIN_PYTHON` 指定兼容的 Python。模型读取 `project/user_data/model_data/` 中
锁定的比赛权重，运行时不下载模型。Windows 下会把 nagisa 自带的分词资源复制到
`%LOCALAPPDATA%/Soundtrace/runtime/python_packages/`，解决原生库无法读取中文路径的问题。
FireRed 模型也会建立到同一 runtime 目录的 ASCII 路径映射，原权重不会复制或改写。

验证入口：`.venv/Scripts/python.exe app/test_unified.py`。
真实 GPU/API 验证：`.venv/Scripts/python.exe app/benchmark_unified.py`，报告保存到
`app/validation/benchmark.json`。长音频样例由已有约 66 秒对话重复两次构成，用于验证
超过 60 秒后的路由、分块和身份一致性，不能作为新的独立识别质量测试。

2026-10-12 速度对比见 `app/validation/SPEED_REPORT_20261012.md`。
复测：`.venv/Scripts/python.exe app/benchmark_speed.py --label rerun --cases short long speech`。
其中 `speech` 由四段不同的现有开发录音拼接，作为吞吐量测试，不代表独立的连续对话质量评测。

分阶段性能观测的字段口径、GPU 同步开关和状态采样见 `PERFORMANCE.md`。
2026-10-12 短音频观测验证及修改前基线见 `validation/profiling_20261012/REPORT.md`；
这一轮仅补充计时，没有调整模型缓存或识别算法，不能作为新增提速成果。

Main / Fast 切换现在优先复用已加载的模型：Main 子进程和 Fast 对象保留在原有位置，
非活动路线的权重移回 CPU，再让所选路线使用 GPU。CPU 缓存会增加内存占用；内存余量不足时
释放非活动缓存，下次仍按用户所选模式重新加载。Review 保持原来的 Fast 释放方式，MOSS / FireRed
尚未做跨任务复用。录音模式真实麦克风验收仍待补充。

默认保护线为可用物理内存 `max(4 GiB, 系统内存的 15%)`，Windows 提交内存余量另保留 1 GiB。
启动和任务期间检查余量，后台每 5 秒检查一次，仅释放已停放到 CPU 的非活动缓存。
这些检查不保证任意录音或其他程序突发分配都不会耗尽内存；8 GB GPU 上未强制两套权重同时驻留。
可设置 `SOUNDTRACE_MODEL_REUSE=0` 恢复原有切换时释放模型的行为；
`SOUNDTRACE_CACHE_MIN_FREE_GB` 可指定物理内存保护线，修改环境变量后重启服务。
`psutil` 不可用或内存观测失败时不保留缓存。

本轮短音频真实切换证据与限制见 `validation/lifecycle_20261012/REPORT.md`。

启动时服务不再替用户默认预热 Main。页面恢复所选模式后，Main / Review 预热 Main；
Fast 在实际任务时加载自己的模型。改选 Fast 可以取消尚未用于识别的 Main 后台预热，
等待旧进程退出后再使用 GPU。已经排队或执行的 Main / Review 任务所需预热不会被取消。
模型缓存、内存保护线、识别参数和 Review 融合规则保持原样。
2026-10-13 的启动、首次结果、真实 Edge/GPU 文件验证见 `validation/startup_20261013/REPORT.md`。
