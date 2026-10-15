# 跨平台一键复现

本目录属于 CatalystFlow 队伍的最终提交。

本目录解决“拿到提交包后怎样运行”的问题。所有脚本自动定位提交目录中的
`code/`、`data/` 和固定模型配置；默认把新结果写到 `reproduced_results/`，
不会覆盖随包提交的正式结果。

## 选择你的操作系统

| 平台 | 数值复现入口 | 首次运行是否需要网络 | 完整 Open Science 智能体 |
|---|---|---|---|
| Linux x86_64 | `bash runtime/run_reproduction.sh` | 否，使用随包离线环境 | 原生支持 |
| Windows 10/11 x86_64 | `.\runtime\run_reproduction_windows.ps1` | 否，使用随包原生离线环境 | 通过 WSL2/Ubuntu 使用 Linux 入口 |
| macOS Intel / Apple Silicon | `bash runtime/run_reproduction_macos.sh` | 需要 | 未提供随包 Open Science 原生归档 |
| Linux ARM64 | `bash runtime/run_reproduction_linux_arm64.sh` | 需要 | 未提供随包 Open Science 原生归档 |

所有数值入口先执行环境烟雾测试，再执行 2 × 20 × 40 正式复现。评委无需修改源码、
配置文件或路径参数。

## Linux x86_64：真正离线

```bash
bash runtime/run_reproduction.sh
```

`catalystflow-linux-x86_64.tar.gz` 由最终验证环境通过 `conda-pack` 生成。
脚本首次运行时从归档解压生成 `runtime/.portable_env/` 并执行 `conda-unpack`；
该目录是可再生成的本地缓存，不属于提交清单。首次运行无需
联网、安装 conda 或重新求解依赖。归档哈希与平台信息在
`portable_environment.json`。

## Windows 10/11 x86_64：真正离线

在提交目录根部打开 PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\runtime\run_reproduction_windows.ps1
```

`Set-ExecutionPolicy -Scope Process Bypass` 只对当前 PowerShell 进程临时生效，
关闭窗口后恢复，不修改系统级策略。启动器优先使用
`runtime/windows-x86_64/catalystflow-windows-x86_64.tar.gz`：首次解压到
`runtime/.portable_env_windows/` 并执行 `conda-unpack`，无需联网、安装 Conda
或重新求解依赖。归档 SHA-256、关键版本和验收记录见
`runtime/windows-x86_64/environment-lock.json`。

该归档在 Windows 10/11 x86_64 上完成了两个数据集各 20 个随机种子 × 40 轮的
端到端复现；40 条轨迹的查询索引、条件、反馈产率和 best-so-far 均与随包 Linux
基准轨迹一致。它锁定 Python 3.11.15、NumPy 1.26.4、SciPy 1.11.4、pandas 2.1.4、
scikit-learn 1.3.2、RDKit 2026.03.4，以及经实机验证的 OpenBLAS 后端。

若离线归档被刻意移除，启动器才回退到以下联网创建策略：

1. 优先读取随包的固定版本 micromamba 启动归档；
2. 若归档缺失，依次尝试清华大学镜像、北京外国语大学镜像和
   conda-forge 官方源；
3. 每个下载源最多尝试两次，带 180 秒超时、部分文件清理和最小文件大小检查；
4. 下载后核对固定 SHA-256，解压后再执行 `micromamba --version`；
5. 创建环境时依次尝试上述三个 conda-forge channel；
6. 成功后缓存到 `runtime\.platform_env_windows\`，后续运行直接复用。

镜像能显著降低单一站点阻塞风险，但任何依赖外部网络的首次环境下载都不能做
100% 可用性承诺。若单位网络同时屏蔽所有镜像和官方源，可在一台联网 Windows
机器上先运行一次，再连同 `.platform_env_windows` 缓存交付；或按
`code/environment-runtime.yml` 在本地 conda/micromamba 中预建环境。

## macOS

同一脚本自动识别 Intel (`x86_64`) 或 Apple Silicon (`arm64`)：

```bash
bash runtime/run_reproduction_macos.sh
```

首次运行创建 `runtime/.platform_env_macos/`，以后复用。要求 macOS 11 或更高
版本，并需要网络获取对应平台的 conda-forge 软件包。

## Linux ARM64

```bash
bash runtime/run_reproduction_linux_arm64.sh
```

首次运行创建 `runtime/.platform_env_linux_aarch64/`，以后复用。

## 包内输入与输出

正式复现统一使用零参数命令。输入固定为包内组织方原始数据 `data/`，输出固定为
包内 `reproduced_results/`：

```bash
bash runtime/run_reproduction.sh
```

Windows 同样使用零参数形式：

```powershell
.\runtime\run_reproduction_windows.ps1
```

启动器保留路径参数仅用于开发兼容，但提交评审不需要填写。脚本会在模型开始前拒绝
`/path/to/...` 等示例占位符，并检查输出目录是否可创建。

## Open Science 智能体

Linux x86_64 的 Open Science 2.0.23 官方发布归档已包含在提交包中。评委无需预装
Open Science、Node.js 或 npm。先验证数值环境和智能体引擎：

```bash
bash runtime/verify_offline_bundle.sh
```

配置 DeepSeek API Key。该命令把 Key 写入评委用户级 Open Science 密钥库，不写入
提交目录：

```bash
bash runtime/configure_deepseek.sh
```

配置脚本随后会自动运行赛事端点专用验证。Open Science 的通用 DeepSeek 凭据界面可能
按官方 DeepSeek 服务检查赛事 Key，从而显示 `Authentication Fails`；赛事 Key 只对
`openscience.json` 中的自定义比赛网关有效，应以脚本最后的 `AUTH_OK` 验证结果为准。
已保存凭据时可直接复核：

```bash
bash runtime/verify_deepseek.sh
```

随后启动工作区：

```bash
bash runtime/run_openscience.sh
```

`openscience.json` 已将 `catalystflow` 设为默认主智能体，Open Science 2.0.23 的 Web
界面不需要也不提供单独的主智能体选择器。启动器先运行
`prepare_openscience_workspace.sh`，在提交根目录建立独立的本地 Git 项目边界，避免
Web 会话继承上级项目配置；该运行时目录不进入提交清单。使用本次启动命令输出的 URL
新建会话，不要复用旧的 `prj_.../session/...` URL。指定端口被占用时，省略 `--port`
让 Open Science 自动分配端口。

启动器会校验 `runtime/openscience/runtime.json` 中记录的归档和二进制 SHA-256，
首次运行时解压至 `runtime/.openscience_runtime/`，并始终调用该包内二进制。不要在
Open Science 2.0.23 的 `web` 后附加 `.`；该形式会显示帮助并退出。

需要使用其他 Open Science 子命令时，统一通过包内命令包装器，例如：

```bash
bash runtime/openscience_cli.sh keys list
bash runtime/openscience_cli.sh debug agent catalystflow
```

CLI `run` 为一次性调用。完整复现应按“数值阶段”和“报告阶段”使用两个会话，并在第二
阶段提示词中传入第一阶段返回的 `job_id`；不要用不带会话 ID 的 `-c` 续接历史任务。
已验证命令与成功判据见 `doc/reproduction_guide.md` 第 2.3 节。完整在线证据检索优先使用
Web；只有提示词明确指定“离线模式”时，CLI 报告阶段才允许跳过外部检索。

离线归档解决的是 Open Science 引擎、Agent、Skill 和 MCP 工具的复现；DeepSeek
推理仍需要访问赛事 API、有效 Key 和可用额度。详细来源与支持边界见
`runtime/openscience/README.md`。

Windows 原生 PowerShell 脚本不启动智能体。Windows 评委复现完整智能体时，在
WSL2/Ubuntu 中进入同一提交根目录，执行上述三条 Bash 命令；此时使用的是包内 Linux
x86_64 数值环境和 Open Science 归档。提交包不宣称提供 Open Science 的 Windows
原生二进制。

## 复现承诺与边界

- Linux x86_64 离线归档已在最终提交验收中从干净状态重新解压并完成端到端轨迹
  一致性验证。
- Windows x86_64 离线归档具有独立 Windows 实机验收记录：环境烟雾测试、25 项数值
  单元测试和 40 条正式轨迹均通过；记录与哈希见
  `windows-x86_64/environment-lock.json`。在 Linux 主机上只能复核其归档哈希和内容，
  不能代替 Windows 执行原生二进制。
- Open Science 2.0.23 Linux x86_64 官方归档已真实解压，并确认能够发现
  `catalystflow (primary)`；该检查不调用 DeepSeek。
- macOS 和 Linux ARM64 启动器已完成路径、命令和环境预检的静态检查；Linux
  构建机无法代替对应操作系统执行其原生二进制。
- `code/environment-runtime.yml` 锁定模型运行所需的最小版本；
  `code/environment.yml` 另含绘图、文档与测试依赖。
- 不同平台的二进制文件、BLAS 和编译器构建不同；跨平台比较对象是查询索引、反应
  条件、反馈产率、best-so-far 和汇总指标，不要求环境归档逐字节相同。

Linux 运行结束后可执行：

```bash
python code/scripts/verify_final_submission.py --package-root .
```

此命令验证提交包本身；新运行结果可再用 `code/scripts/validate_submission.py`
逐数据集检查。

Windows 原生运行结束后使用：

```powershell
.\runtime\.portable_env_windows\python.exe .\code\scripts\verify_final_submission.py `
  --package-root .
```
