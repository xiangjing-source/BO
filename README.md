# CatalystFlow：偶联反应条件优化智能体

> 🏆 **2026 年书生国智科探挑战赛优秀作品**

本仓库保存 CatalystFlow 的最终参赛版本：一个面向 Suzuki–Miyaura C–C 偶联和
Buchwald–Hartwig C–N 偶联反应条件优化的、可复现且可审计的智能体系统。

CatalystFlow 面向赛题规定的 Suzuki–Miyaura C–C 偶联和 Buchwald–Hartwig C–N
偶联反应条件优化。提交包包含赛题公开数据、固定模型配置、两套数据集各 20 个随机种子
× 40 轮的完整优化轨迹、四项核心指标、离线数值复现环境，以及基于 Open Science 和
DeepSeek 的智能体。

最终模型负责产率建模、不确定性估计、期望改进和候选排序；DeepSeek 负责新数据语义
解析、查询完成后的现实可行性评估和报告生成。DeepSeek 与 DFT 均不读取未查询候选的
产率，也不改变优化器排序。

> 本仓库保存个人最终提交的完整数据、代码、DFT 快照、优化轨迹、结果和文档。
> 两份超过 GitHub 普通 Git 单文件上限的离线运行时压缩包已作为
> [CatalystFlow 2026 最终提交 Release 附件](https://github.com/xiangjing-source/BO/releases/tag/catalystflow-final-submission-2026)
> 发布。下载后请分别放回 `runtime/` 与 `runtime/windows-x86_64/`，其余内容可直接克隆。

评委无需修改源码、配置文件或数据路径。将提交包完整解压到可写目录后，按操作系统
执行下述零参数命令即可重新生成数值轨迹；完整智能体复现还要求赛事 DeepSeek API
可访问、API Key 有效且额度充足。

## 1. 评委快速入口

| 赛题材料 | 提交文件 |
|---|---|
| 技术路线 | `doc/method_description.docx`（最终内容基准）、同名 PDF（评审版）和 Markdown（文本镜像） |
| 团队履历 | `doc/resume.pdf` |
| 最终结果、智能体分析与推荐 | `results/results_and_analysis.docx`（最终内容基准）、同名 PDF、Markdown 和 JSON |
| 技术路线图 | `results/figures/framework_overview.png` |
| 四项核心指标 | `results/summary_metrics.csv` |
| 40 条完整优化轨迹 | `results/optimization_trajectories/<dataset>/seed_<seed>.pt` |
| 环境与复现 | `doc/reproduction_guide.md`、`runtime/`、`code/environment.yml` |
| 代码、配置和日志 | `code/main/`、`code/config/`、`code/logs/` |
| 文件完整性 | `MANIFEST.json` |
| Open Science 智能体定义 | `.openscience/agent/catalystflow.md` |

提交轨迹严格遵循赛题协议：`Suzuki` 和 `Buchwald_sub4` 各使用 20 个固定随机种子
`100, 200, …, 2000`，每个随机种子 40 轮，每轮查询 1 个候选。每轮记录当前轮数、
候选索引、完整条件、观测产率、best-so-far、模型预测值、不确定性、采集函数值和选择
方式。

## 2. 最终模型

固定模型配置为 `code/config/production_config.json`。

| 数据集 | 最终模型 | 采集策略 |
|---|---|---|
| Suzuki | 随机森林与采用可交换分类协方差张量积核的高斯过程加权集成；预测均值权重为 0.75 和 0.25 | 期望改进为主，使用运行前固定的递减均匀探索概率 |
| Buchwald_sub4 | 使用独热编码、与产率标签无关的化学知识、分子指纹和物化描述符的随机森林 | 期望改进 |

最终模型不是在阶段性模型上继续训练或调参。模型结构与超参数仅根据初始训练数据的
内部验证确定，随后写入固定模型配置。每个随机种子独立初始化并从头训练，不继承模型
文件、拟合参数、超参数搜索状态、查询记录或随机状态。正式评估结果不用于后续模型
选择或调参。

## 3. 核心结果

以下数值为 20 次独立运行的均值、样本标准差和基于 t 分布的 95% 置信区间。

| 数据集 | 指标 | 均值 | 标准差 | 95% 置信区间 |
|---|---|---:|---:|---:|
| Suzuki | B1（`initial_round_found_best`） | 84.0425 | 8.6937 | [80.2323, 87.8527] |
| Suzuki | B40（`best_found`） | 97.3625 | 1.3957 | [96.7508, 97.9742] |
| Suzuki | t95（`round_to_95_global_best`） | 4.4000 | 8.7323 | [0.5729, 8.2271] |
| Suzuki | AUC（`AUC_best_so_far`） | 95.8840 | 1.1440 | [95.3826, 96.3853] |
| Buchwald_sub4 | B1（`initial_round_found_best`） | 74.3877 | 1.7673 | [73.6132, 75.1623] |
| Buchwald_sub4 | B40（`best_found`） | 86.0670 | 1.2052 | [85.5388, 86.5951] |
| Buchwald_sub4 | t95（`round_to_95_global_best`） | 12.5000 | 5.1350 | [10.2495, 14.7505] |
| Buchwald_sub4 | AUC（`AUC_best_so_far`） | 82.5350 | 1.4948 | [81.8799, 83.1902] |

最终复核在独立环境中从包内 `data/` 重新训练和顺序查询，生成 40 条轨迹和 1,600 条
记录；40/40 条轨迹均通过审计，与提交轨迹逐轮比较为 `identical`。复核记录见
`code/logs/reproduction_verification.log`。

## 4. 数值结果复现

数值复现支持 Linux x86_64 和 Windows 10/11 x86_64，均使用随包离线环境：

| 平台 | 在提交根目录执行 | 网络要求 |
|---|---|---|
| Linux x86_64 | `bash runtime/run_reproduction.sh` | 不需要 |
| Windows 10/11 x86_64 PowerShell | `.\runtime\run_reproduction_windows.ps1` | 不需要 |

Linux 命令：

```bash
bash runtime/run_reproduction.sh
```

Windows PowerShell 命令：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\runtime\run_reproduction_windows.ps1
```

两个零参数入口均自动解压对应平台的随包环境，固定读取 `data/`，并将新结果写入
`reproduced_results/`。模型执行阶段不以 `results/optimization_trajectories/` 中的提交
轨迹作为输入；新轨迹生成后，校验程序才读取提交轨迹进行逐轮比较。该流程不调用
DeepSeek，也不需要联网、安装 Conda 或下载数据。

如需自行创建环境：

```bash
conda env create -n catalystflow-test -f code/environment.yml
conda run -n catalystflow-test python code/main/run_selected.py \
  --data-root data --results-root reproduced_results \
  --dataset all --rounds 40 --seeds 20
```

macOS 和 Linux ARM64 的入口需要联网创建平台环境，未作为离线提交承诺。完整的平台
边界、验收条件和故障定位见 `doc/reproduction_guide.md` 与 `runtime/README.md`。

## 5. 智能体复现

提交包已包含 Open Science 2.0.23 Linux x86_64 官方运行时，不依赖评委电脑上的全局
安装。Linux x86_64 可直接运行；Windows 评委使用 WSL2/Ubuntu 执行同一组 Linux
命令。提交包不声称提供 Open Science 的 Windows 原生二进制。先将 DeepSeek API Key
写入 Open Science 用户级密钥库，再启动项目：

```bash
bash runtime/configure_deepseek.sh
bash runtime/run_openscience.sh
```

`configure_deepseek.sh` 保存 Key 后会通过赛事自定义端点执行一次 `AUTH_OK` 验证。
Open Science 的通用 DeepSeek 凭据界面可能按官方 DeepSeek 服务检查赛事 Key，并显示
`Authentication Fails`；该提示不适用于赛事自定义端点，以脚本最后输出的
`Competition DeepSeek endpoint authentication: PASS` 为准。已配置后也可单独执行
`bash runtime/verify_deepseek.sh` 复核。项目不会把 Key 写入提交目录。

如只验证离线运行时和智能体发现过程，不调用 DeepSeek：

```bash
bash runtime/verify_offline_bundle.sh
```

Open Science 2.0.23 会自动将 `catalystflow` 设为当前主智能体，无需在界面中另行
选择。启动器会在提交根目录建立仅用于 Open Science 项目标识的本地 Git 边界，避免提交
目录位于其他 Git 工作区内时错误继承上级项目及其模型配置；该运行时目录不进入提交清单。
请使用启动命令本次输出的 URL 新建会话，不要复用此前含其他 `prj_...` 标识的会话 URL。
若指定端口已被占用，可省略 `--port` 由 Open Science 自动分配端口。进入工作区后输入：

```text
检查项目内 data，使用最终模型执行两套数据集各 20 个随机种子、每个随机种子
40 轮的正式复现实验；核验 40 条轨迹和 1600 条查询记录，报告 B1、B40、t95、AUC，
并对两个数据集的已查询 Top-5 反应条件生成六维现实可行性评估、DFT 解释和
最终推荐反应条件报告。
```

CLI 的 `run` 是一次性命令。Open Science 2.0.23 对同时包含数值复现、外部检索和 11 次
结构化报告提交的超长单会话可能触发自动上下文压缩，因此 CLI 按两阶段执行。第一阶段只
完成数值复现并记录返回的 `job_id`：

```bash
bash runtime/openscience_cli.sh run \
  --agent catalystflow \
  --model deepseek/deepseek-v4-flash \
  --title cli-numerical-reproduction \
  "检查项目内 data，只执行 official_reproduction 数值阶段；完成后核验40条轨迹、1600条记录和基准一致性，读取 summary_metrics 并报告 B1、B40、t95、AUC，输出 job_id 后停止，不执行现实可行性评估。"
```

第二阶段使用新会话，并把实际 `job_id` 写入提示词：

```bash
bash runtime/openscience_cli.sh run \
  --agent catalystflow \
  --model deepseek/deepseek-v4-flash \
  --title cli-report-stage \
  "不要重跑数值优化。对 job_id=<第一阶段返回值> 完成已查询Top-5六维评估、DFT解释、10次候选增量提交和1次最终推荐提交，生成 enriched_report。"
```

第二阶段默认执行外部数据库优先协议；仅验证 CLI 结构化报告链路时，可明确指定“离线
模式”，此时相应判断必须标为 `llm_inference` 或 `unknown`。不要使用不带会话 ID 的
`-c` 续接历史任务；故障恢复使用 `--session <session_id>`，并在提示词中写明 `job_id`
和待完成阶段。Web 是完整在线证据报告的推荐交互入口。

智能体报告位于 `.catalystflow-agent/jobs/<job_id>/report/`，以 JSON 和 Markdown 保存。
提交包将数值结果和一次完整智能体运行合并为同一套 PDF、DOCX、Markdown 和 JSON 评审
报告：`results/results_and_analysis.*`。数值复现不依赖 API；DeepSeek 端点不可用时，仅
影响语义解析和现实可行性评估，不影响正式轨迹与四项核心指标。

数值作业完成后，受限工具会直接向智能体返回 `summary_metrics`，包括两个数据集的 B1、
B40、t95、AUC 的 20 次运行均值、样本标准差和 95% 置信区间。评委询问正式指标时无需
授予任意文件读取权限，也不会向 DeepSeek 返回未查询产率或候选池全局最优值。

Open Science 在本项目中承担智能体运行框架，而不是产率优化器：它加载
`catalystflow` Skill，连接受限 MCP 工具，调用 DeepSeek，管理会话与增量报告，并通过
权限表限制可用能力。`science_list_dbs`、`science_search`、`science_fetch`、
`provenance_*` 和 `artifact_snapshot` 已在 Agent 中授权。Top-5 报告默认先列出可用数据库，
再用 PubChem/ChEBI 核验身份和性质，用 Crossref/OpenAlex/Semantic Scholar、PubMed 或
Europe PMC 检索相关文献，并读取拟引用记录；没有直接证据时才降级为 LLM 推断或未知。
科学数据库不能替代供应商报价或官方 SDS。提交样例在数值 Top-5 冻结后完成外部证据
补充，共保存 20 条化学数据库、文献、供应商目录和 SDS 来源，见
`results/top5_external_evidence.json`。没有合格来源的维度仍标为 `llm_inference` 或
`unknown`；目标规模和地点未提供，因此预算和放大风险不作确定判断。包内 86 个 ORCA
计算构成独立的本地 DFT 证据库，只用于查询完成后的解释。

## 6. 输入、输出与数据边界

正式复现只读取包内 `data/`。新数据应放入 `data/<task>/`，推荐提供 JSON 清单（manifest），
声明训练表、无结果候选表、结果列、字段类型、单位和分子结构映射。未提供 manifest
时，DeepSeek 可在结果信息隔离后，仅根据文件名、表头和非结果元数据生成语义草案；确定性
程序继续检查结果列泄漏、字段缺失、重复条件、类别与组合新颖性、连续变量漂移、单位、
结构覆盖率和分布外情况。存在歧义时停止运行。

`*_test.csv` 仅由离线预言器（OfflineOracle）持有；最终模型在候选索引确定后才能得到
对应一行产率。`*_searchspace.csv` 仅用于原始数据完整性审计，正式运行入口不读取。
全局最优值只在轨迹完成后用于指标计算。

轨迹完成后，从合法查询记录中汇总每个数据集观测产率最高的五个唯一条件，称为
“已查询 Top-5 反应条件”。DeepSeek 从预算、采购难度、初步危险性、操作复杂度、
放大风险和废物处置负担六个维度进行评估，并从非阻断条件中给出“最终推荐反应条件”。
评估为初步判断；实验前仍须核对正式报价、库存、SDS、SOP、EHS 要求和废物处置路径。

## 7. 提交目录

```text
CatalystFlow_submission/
├── code/          # 可复现实验代码、固定模型配置、环境文件、日志和校验脚本
├── doc/           # 技术路线、团队履历、复现指南和提交说明
├── results/       # 40 条轨迹、四项核心指标及统一的结果与智能体分析报告
├── data/          # 赛题公开数据的包内副本
├── runtime/       # 数值环境、Open Science 官方运行时及跨平台启动器
├── skill/         # CatalystFlow Skill
├── agent/         # 受限工具与策略网关
├── .openscience/  # Open Science 智能体定义
├── dft/           # 可审计 DFT 计算与特征快照
├── openscience.json
└── MANIFEST.json
```

最终提交 ZIP 控制在 1 GB 以内。运行时生成的 `runtime/.portable_env/`、
`runtime/.openscience_runtime/`、`reproduced_results/` 和 `.catalystflow-agent/` 不属于
提交内容，并由清单生成器排除。
