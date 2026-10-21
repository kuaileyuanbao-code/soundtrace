# 源码清单与导出

首个真实基线创建于 2026-10-15，当前说明更新于 2026-10-21。参赛源码仓库 https://github.com/kuaileyuanbao-code/soundtrace 已公开，评委可直接查看源码和真实提交历史，也可下载或克隆仓库。

## 文件组织

source_selection.json 是显式文件清单；source_manifest.json 记录大小、类别和 SHA256。.gitignore 只允许跟踪选定文件，新增文件需先核对再加入。gitattributes 保留文件字节与换行，便于核对实际运行版本。

仓库包含应用、四路线调用脚本、原算法赛 ifly_tcpwer 必需子集、回归测试、FireRed 补丁、第三方来源、Windows 构建和校验工具。FireRed 历史回归参照是测试资料，真实 Git 历史由提交记录保存。

用户录音、声纹、任务结果、日志、模型、Python 环境、比赛数据及第三方完整 checkout 分别管理，不进入源码清单。运行包与源码仓库的对应关系见 WINDOWS_DELIVERY.md 与 windows_release_20261020.json。

## 检查

```powershell
python tools/source_baseline.py --check
```

工具检查路径、重解析跳转、文件类型、大小及散列，并扫描常见密钥和作者机器绝对路径。修改清单内文件后执行：

```powershell
python tools/source_baseline.py --refresh
python tools/source_baseline.py --check
```

新增文件同时加入 source_selection.json 与 .gitignore。检查工具只处理源码，不安装依赖或加载模型。

## 导出

真实提交完成后执行：

```powershell
python tools/source_baseline.py --export
```

工具确认清单与 HEAD 一致，在 delivery/exports 下创建新目录，保存相对层级及 SOURCE_SNAPSHOT.json 的 commit；保留旧导出。源码快照供代码检查，实际运行使用独立 Windows 运行包。

environment-observed.json 保存开发环境的历史观察；运行包 ENVIRONMENTS.json 保存随包环境的具体依赖。源码测试入口见根 README，app/README.md 中的历史实验入口保留其原始语境。
