# CatalystFlow 提交内容核对表

本文件依据“偶联反应条件优化任务”初评材料要求，说明提交文件、复现入口与材料边界。

## 1. 初评材料对应关系

| 组织方要求 | 提交内容 | 核对结果 |
|---|---|---|
| `code/` 可复现实验代码 | `code/main/run_selected.py`、模型代码、配置、测试与脚本 | 完整 |
| Dockerfile 或环境文件 | `code/Dockerfile`、`code/environment.yml`、`code/environment-runtime.yml` | 完整 |
| 配置文件 | `code/config/production_config.json` 及结构映射 | 完整 |
| 运行日志 | `code/logs/reproduction_verification.log` 及结构化轨迹审计 | 完整 |
| 使用说明与复现指南 | `README.md`、`code/README.md`、`doc/reproduction_guide.md` | 完整 |
| 技术方案 | `doc/method_description.pdf`、同名 DOCX/Markdown | 完整 |
| 团队履历 | `doc/resume.pdf` | 完整 |
| 每个数据集、每个随机种子的完整轨迹 | `results/optimization_trajectories/`，共 40 个 `.pt` 文件 | 完整 |
| 最终结果、智能体分析与推荐 | `results/results_and_analysis.pdf`、同名 DOCX/Markdown/JSON | 完整 |
| 核心指标均值、标准差、95% 置信区间 | `results/summary_metrics.csv` | 完整 |

### 评审维度阅读索引

| 初评维度 | 主要证据位置 |
|---|---|
| 研究方法的创新性 | `doc/method_description.pdf` 第 1、3、7 节 |
| AI 方法设计与可行性 | `doc/method_description.pdf` 第 1–5 节；`agent/README.md` |
| 优化性能、鲁棒性与可复现性 | `results/results_and_analysis.pdf`、`summary_metrics.csv`、40 条轨迹、`code/logs/`、`doc/reproduction_guide.md` |
| 科学合理性与可解释性 | `doc/method_description.pdf` 第 5 节、`dft/`、`results/results_and_analysis.pdf` 第 6、9、10 节 |
| 团队能力与跨学科配置 | `doc/resume.pdf` |

该索引只帮助定位证据，不构成参赛队伍自评分。

## 2. 正式实验协议

- 数据集：Suzuki、Buchwald_sub4；
- 每轮采样数：1；
- 每个随机种子轮数：40；
- 每套数据随机种子数：20；
- 随机种子：100、200、…、2000；
- 总轨迹数：40；
- 总顺序查询记录数：1,600。

每轮轨迹包含 `step`、`query_index`、`condition`、`observed_yield`、
`best_so_far`、`predicted_yield`、`uncertainty`、`acquisition_value` 和
`selection_mode`。`.pt` 由 Python 标准库 `pickle` 保存，不依赖 PyTorch。

## 3. 复现入口

Linux x86_64 零参数数值复现：

```bash
bash runtime/run_reproduction.sh
```

该命令固定读取包内 `data/`，重新训练模型并生成 40 条新轨迹，输出到
`reproduced_results/`。数值复现不调用 DeepSeek，不加载提交轨迹，不需要联网。

Windows 10/11 x86_64 原生 PowerShell 数值复现：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\runtime\run_reproduction_windows.ps1
```

Windows 入口使用随包原生离线环境，输入输出和验收条件与 Linux 一致。两个平台均不
需要修改源码、固定模型配置或数据路径。

Open Science 智能体：

```bash
bash runtime/verify_offline_bundle.sh
bash runtime/configure_deepseek.sh
bash runtime/verify_deepseek.sh
bash runtime/run_openscience.sh
```

提交包已包含 Open Science 2.0.23 Linux x86_64 官方运行时。第一条命令在不调用
DeepSeek 的情况下校验数值环境、智能体引擎和 `catalystflow` 智能体发现过程；第二条
命令把 API Key 写入评委用户级密钥库；第三条通过赛事自定义端点验证 Key、DeepSeek、
Agent 与 MCP。Key 不写入提交目录。

完整智能体原生支持 Linux x86_64；Windows 评委通过 WSL2/Ubuntu 执行上述 Bash
命令。Windows 原生 PowerShell 入口只用于数值复现，提交包不包含 Open Science 的
Windows 原生二进制。

智能体用于新数据语义解析，以及轨迹完成后的已查询 Top-5 反应条件、DFT 解释、六维
现实可行性评估和最终推荐反应条件报告。具体提示词和新数据流程见
`doc/reproduction_guide.md`。

## 4. 数据泄漏控制

最终模型只读取初始训练数据、无结果候选条件和当前随机种子已经查询条件的反馈。
答案表只由 OfflineOracle 持有，并在候选索引确定后返回一行产率。全局最优值仅用于
轨迹完成后的指标计算。各随机种子从头训练且状态隔离；第三方数据、DeepSeek 和 DFT
不参与组织方数据对应的固定模型配置选择或正式候选排序。

外部数据、文献、预训练模型、化学数据库、大语言模型和 DFT 的用途与边界已在
`doc/method_description.pdf` 中单独说明。

## 5. 提交目录与大小

```text
CatalystFlow_submission/
├── code/          # 代码、配置、环境文件、日志和评测脚本
├── doc/           # 技术方案、团队履历和复现说明
├── results/       # 完整轨迹、核心指标及统一的结果与智能体分析报告
├── data/          # 组织方公开数据的包内副本
├── runtime/       # 数值环境、Open Science 官方运行时与跨平台启动器
├── skill/         # CatalystFlow Skill
├── agent/         # 受限工具与策略网关
├── .openscience/  # Open Science 智能体定义
├── dft/           # DFT 计算与特征快照
├── openscience.json
└── MANIFEST.json
```

提交包约 0.960 GB，低于 1 GB。`MANIFEST.json` 记录所有提交文件的 SHA-256 与字节数。
运行生成的 `runtime/.portable_env/`、`runtime/.openscience_runtime/`、
`reproduced_results/` 和 `.catalystflow-agent/` 不属于提交内容。
