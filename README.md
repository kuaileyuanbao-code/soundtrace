# 声迹

Windows 本地语音转写与对话整理软件。现有 Text、Fast、Main、Review 四条识别路线，支持音频导入、浏览器麦克风、文字编辑、结果历史与导出。

这是从算法赛工作区中明确选取的**源码基线**，不是已经跨机验证的运行包。基础模型、Python 环境、第三方源码、原生运行库、个人录音和声纹均不随本仓库提供。保留的目录层级对应当前生产路径，不要求移动原工作区。

- [应用功能、模型路线和已知边界](app/README.md)
- [源码范围、检查和导出](delivery/SOURCE_BASELINE.md)
- [第三方来源与补丁](delivery/THIRD_PARTY.md)
- [Review 授权与交付状态](delivery/REVIEW_STATUS.md)
- [真实版本历史及已有成果说明](delivery/DEVELOPMENT_HISTORY.md)

## 当前状态

当前作者电脑已有四模式真实 GPU 和主要交互验证。2026-10-15 对指定人数但长发言原型覆盖不足的双人错误进行了限定修复，两份原录音回归成功；不能保证任意双人或长会议录音的人物全部正确。

Main/Text 暖态较快，首次模型准备仍可能等待几十秒。Review 开销及内存压力更高，其 MOVER 依赖的第三方分发条件尚未解决。源码还保留本机环境查找规则，依赖准备和跨机部署属于后续任务。

## 使用与验证入口

作者当前已配置电脑可双击 `打开声迹.bat`，用后执行 `关闭声迹.bat`。新电脑不能直接假定双击即可运行；须先准备依赖、权重及兼容运行时。尤其 `server.py` 启动时会读取 FireRed 两个源码文件的哈希，缺少第三方源码时即使只选 Text 也不能按当前代码直接启动。

在已按现有版本准备环境和第三方源码的电脑中：

```powershell
.venv/Scripts/python.exe -X utf8 -m unittest discover -s app -p 'test_*.py' -v
node app/test_modes_frontend.cjs
```

Python 测试需要现有开发依赖，路由测试还需要 Node.js 在 PATH；这是模型叶子被模拟的回归，不是 GPU 端到端测试。涉及私人原音的两项回归在纯源码检出时按设计跳过，未公开其音频或完整输出。正式发布仍欠独立环境安装与实际部署验证。

源码整理工具只使用 Python 标准库：

```powershell
python tools/source_baseline.py --check
```

本次基线不自动改变自有代码的许可，也不授予第三方组件额外权利。团队真实贡献、算法赛既有成果及 AI 辅助开发范围须如实披露；后续是否公开仓库由团队决定。
