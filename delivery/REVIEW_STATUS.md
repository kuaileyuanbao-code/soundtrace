# Review 配置与交付范围

更新日期：2026-10-20。

开发工作区的 Review 路线为 Main → MOSS → FireRed → MOVER，保持现有识别参数及融合规则。当前 Windows 运行包提供 Text/Fast/Main；Review 保留扩展入口，依赖未配置时返回明确错误，不切换到其他模式。

源码仓库保存本项目的 Review 调用代码、FireRed 按需导入补丁与第三方来源记录。Windows 包未附 MOSS/FireRed 模型、MOSS 环境、MOVER 实现及其原生构建。

MOVER 上游为 [nttcslab-sp/mover](https://github.com/nttcslab-sp/mover)，所用版本见 third_party_sources.json。其 [LICENSE](https://github.com/nttcslab-sp/mover/blob/main/LICENSE) 为评估用途协议，包含第三方转交条件；本次未取得额外分发授权，因此该组件未纳入源码仓库及运行包。

服务已支持缺少可选 Review 源码时正常启动 Text/Fast/Main。未来如单独配置 Review，需要同时核对组件授权、模型环境、SpyDER ABI 与 DLL、系统内存和 GPU 资源。
