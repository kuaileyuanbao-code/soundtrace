# 声迹性能计时口径

本轮只增加观测，不更改模型、解码参数、窗口、缓存策略或 MOVER 配置。
时长均用 `perf_counter`，单位秒。墙钟 `observed_at` 只用于定位采样时刻。

## 使用与证据

- 默认 `SOUNDTRACE_PROFILE_SYNC=0`：轻量墙钟计时，不强制同步 CUDA。
- 精确区分相邻 GPU 阶段时，验证前设 `SOUNDTRACE_PROFILE_SYNC=1` 并重启服务；验证后恢复默认。
  同步在阶段边界进行，改变调度和计时开销，不改变识别参数。没有逐算子 profiler 或持续追踪。
- 每个任务保存 `app/jobs/<id>/performance.json`：时间、模型设备采样、源文件 SHA-256 和日志。
- Main 原有 `unified/metadata.json` 增加细分计时和加载观测；复核另有
  `unified/review/performance.json`、`firered_performance.json`、`mover_performance.json`。
  MOSS 完成一块就保存一次观测，后续阶段失败时仍可查看已有记录。
- 浏览器完成结果渲染后，以 `performance.now()` 回报客户端时间，保存为 `client_timing.json`。
  回报失败不影响识别，缺失数据不按零处理。
- `benchmark_profile.py` 只使用现有 6 秒样例，不读取已有识别答案；记录代码 SHA-256，拒绝覆盖同名报告。
  `benchmark_speed.py` 的历史记录仍保留原样。

## 任务和客户端时间

| 字段 | 起止口径 |
|---|---|
| `upload_receive_seconds` | FastAPI 已解析 multipart 后进入创建任务函数 → 服务端复制音频并准备入队；不含框架此前的 multipart 解析或完整网络上传 |
| `queue_wait_seconds` | 准备提交单线程执行器 → `run_job` 开始；包括前一任务及其观测落盘尚未返回的等待 |
| `gpu_wait_seconds` | `run_job` 准备获取 GPU 锁 → 获取该锁；包括剩余后台预加载等待，不是纯 GPU 计算 |
| `processing_seconds` | 获取 GPU 锁 → 结果文件写好且计算完成；包含转换、加载/恢复、推理、后处理；不含随后观测文件落盘 |
| `elapsed_seconds` | 保留原语义：`run_job` 开始 → 更新完成状态；不含执行器排队或此前上传 |
| `server_end_to_end_seconds` | 创建任务函数入口 → 结果准备完毕；不等于浏览器总等待 |
| `user_end_to_end_seconds` | 浏览器开始提交 → 取回结果、渲染并保存历史；包含请求、排队、轮询间隔和结果下载，不含之前录音/导入/波形解码或随后计时回报 |
| `client_submit_seconds` | 浏览器开始提交 → 收到已接受任务的响应 |
| `client_after_accept_seconds` | 收到已接受响应 → 结果渲染/历史保存完成 |

近似关系：`server_end_to_end = upload_receive + queue_wait + gpu_wait + processing`；
提交执行器与进入锁前的少量控制代码、采样和舍入产生小残差。不能跨进程或浏览器直接相减时间戳。

## Main

- `load_observation` 记录该进程最近一次启动耗时与各加载阶段。`incurred_in_this_call` 表示
  这次 `transcribe` 是否启动了进程；背景预加载耗时不能再加到任务的 GPU 锁等待上。
- `torch_import_seconds`、`qwen_dependency_import_seconds`、`sortformer_dependency_import_seconds`
  分别计时对应导入，不等于纯磁盘读取。
- `qwen_and_aligner_load_seconds` 是构造总时间，包含
  `aligner_load_and_place_seconds`；剩余记为 `qwen_load_and_place_seconds`。
  当前库的 `from_pretrained(device_map=...)` 将加载和初始 GPU 放置合在一起，此处不强行拆成磁盘与搬运。
- `sortformer_cpu_load_seconds` 是 CPU 恢复模型；`campp_cpu_load_seconds` 包括其依赖、模型和特征器初始化。
- 各模型 `*_to_gpu_seconds`、`*_to_cpu_seconds` 和 `cache_cleanup_seconds` 记录原有权重搬运及释放调用。
  若已在目标设备，`.to()` 是无实际权重搬运的调用；应结合前后设备采样判断。
- `asr_seconds` 包裹原始 Qwen `_infer_asr`，包含处理器、输入搬运、生成和解码；
  `forced_aligner_seconds` 包裹原始 `align`。`qwen_transcribe_seconds` 包含两者和库内前后处理。
- `sortformer_seconds` 包含其 `diarize` 内部准备和推理；`campp_features_seconds`、
  `campp_features_to_gpu_seconds`、`campp_seconds`、`campp_result_to_cpu_seconds` 拆分特征、输入搬运、前向和输出搬运。
- `speaker_postprocess_seconds` 为聚类/Viterbi；`seglst_postprocess_seconds` 为文字与人物合并。
- `main_other_seconds` 是当前成功尝试中未单独计时的开销，含状态扫描、时间归属去重、导入和控制代码。
  OOM 时分阶段值累计所有尝试，`worker_request_seconds` 与 `oom_retries` 才是整次请求口径，不能与最后一次 `main_seconds` 强行相加。
- `main_engine_seconds` 包含必要预加载、IPC 和 worker；`main_ipc_and_dispatch_seconds` 为扣除等待及 worker 后的残差。
- 旧 `qwen_seconds`、`diarization_seconds`、`main_seconds` 保留为父级总计，与新增子阶段重叠。

## review

- `moss_startup_seconds`：Popen → worker 发出 ready，包含进程与依赖导入。
  MOSS 的 ModelRunner 是惰性加载，ready 时权重可以仍为 `unloaded`。
- `moss_load_and_place_seconds`：原 `_ensure_loaded`，包含权重、处理器、精度转换和 GPU 放置。
  后续块可命中已加载模型，耗时近零；本轮没有改变加载时机。
- `moss_inference_seconds`：原 `generate_transcription`，包含输入准备、生成和解码。
  `moss_other_seconds` 为 worker 剩余工作；`moss_ipc_and_dispatch_seconds` 为请求扣除启动及 worker 各阶段后的残差。
- `firered_process_seconds` 是整个子进程；其 sidecar 拆出依赖导入、加载/放置、
  `firered_weights_to_gpu_seconds`、完整 `transcribe` 和其他工作。权重 GPU 搬运是加载总时间的子项。
  内部原有 `rtf` 口径较窄，不能代替 `firered_inference_seconds`。
- `mover_process_seconds` 是整个子进程；`mover_fusion_seconds` 仅包裹原始 `mover(...)`，
  不含依赖导入、输入读取、结果规范化/写入、Python 启动和退出。
- `review_other_seconds` 包括分块读写、文字归属、人物映射及未单独计时的控制工作。
  `main_park_before_review_seconds` 在 review 总计之前发生，单独记录。
- 父子阶段有重叠，不能把全部字段相加；review 顶层分区为 MOSS 请求、MOSS 关闭、FireRed 进程、MOVER 进程和 review 其他工作。

## 模型状态与误差

`unloaded / cpu_loaded / gpu_resident / mixed / unknown` 来自参数及 buffer 的真实设备集合，
不是 `gpu_resident` 策略布尔值。Main worker 开始任务和 park/完成后直接采样。
服务端执行开始/获取锁时的 Main 状态采用最近一次 worker 的张量采样，保留 `observed_at`
和 `snapshot_source`；正在加载时可能只是上次阶段采样，不伪装成实时查询。

默认不强制同步时，CAM++ 等 GPU 前向的异步工作可能在后续 CPU 输出搬运处结算，
单个子阶段不是严格 GPU 核执行时间；同步验证的阶段墙钟更适合归因，但仍包含 CPU 工作。
`cuda_sync_seconds` 是同步调用墙钟，其中部分就是等待实际 GPU 工作，并非全属纯观测开销；
它也与相邻阶段计时部分重叠，不另加到总耗时上。

计时器微基准只评估计时作用域本身，不包括设备扫描、函数包装、JSON 和文件写入。
前后短样例的功耗、文件缓存及后台负载会波动，少量样例不能给出严格的总体开销百分比。
