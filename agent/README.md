# CatalystFlow × Open Science Agent

本目录把 `catalystflow` Skill、最终模型和 Open Science 智能体连接成一个完整系统，
并使用 DeepSeek（本项目使用的大语言模型）完成语义解析和现实可行性评估。
智能体的正式输入是包内组织方原始数据 `data/`；新数据须放在提交目录内，并以项目
相对目录或 manifest 输入。输出是可审计的
数值轨迹、已查询 Top-5 反应条件、DFT 解释（不参与候选排序）、现实可行性评估和
最终推荐反应条件。

## 1. 技术框架

```text
用户输入
  ├─ 包内组织方原始数据（data/）
  └─ 项目内新数据目录或 manifest
        │
        ▼
Open Science 智能体 / DeepSeek
任务路由、科学检索、证据组织、中文报告
        │
        ▼
catalystflow Skill
流程、输入契约、证据规则和失败策略
        │
        ▼
受限 MCP（9 个领域工具）
输入检查、正式作业、新数据状态、已查询 Top-5 报告、结构解析
        │
        ▼
最终模型或新数据顺序优化器
产率模型、不确定性、期望改进、单点推荐、真实反馈
        │
        ▼
已查询 Top-5 反应条件 → DFT 解释 + DeepSeek 六维现实可行性评估
        │
        ▼
非阻断条件中的最终推荐反应条件 + JSON/Markdown 报告
```

最终模型是唯一数值决策器。DeepSeek 不读取完整离线预言器答案表、不生成伪产率、不修改
候选索引、期望改进或 `optimizer_rank`。DFT 与 DeepSeek 均位于候选排序完成后的分析层，
`ranking_effect=none`。

固定模型配置仅由初始训练数据的内部验证确定；每个随机种子独立初始化并从头训练。
阶段性模型文件、查询记录和正式评估结果均不进入训练、调参或候选排序。

### 1.1 Open Science 模块的实际使用范围

| Open Science 能力 | 配置状态 | 本项目中的作用 | 本次正式复现 |
|---|---|---|---|
| Agent 与 DeepSeek provider | 已启用 | 理解任务、选择工作流、生成受约束工具参数和中文报告 | 已使用 |
| 项目 Skill | 已启用 | 加载 CatalystFlow 协议、输入契约、证据规则和失败策略 | 已使用 |
| 本地 MCP | 已启用 | 输入检查、2 × 20 × 40 作业、状态查询、新数据闭环和报告组装 | 已使用 |
| 会话与增量工具调用 | 已启用 | 保存 `job_id`，逐条提交 10 条候选评估并支持中断后继续 | 已使用 |
| `science_list_dbs/search/fetch` | 已授权 | Top-5 报告默认先检索可追溯来源；新数据身份、反应族、变量角色需要补充时也使用 | 提交样例已完成外部证据补充 |
| `provenance_record/query/review`、`artifact_snapshot` | 已授权 | 保存来源、检索时间和审阅记录 | 提交样例保留等价机器可读证据快照 |
| 包内 DFT 证据库 | 项目内置 | 读取 86 个 ORCA 计算及聚合特征，解释已查询条件 | 已使用，`ranking_effect=none` |

科学数据库工具是否可检索到具体数据库，由评委运行时的 Open Science 服务状态决定；
可用数据库清单在每次评估开始时实时查询。本项目优先使用 PubChem、ChEBI 核验化学品身份和性质，
使用 Crossref、OpenAlex、Semantic Scholar、PubMed、Europe PMC 检索相关文献，并在引用前
读取具体记录。`science_search` 必须给出正整数 `limit`。项目不保存外部凭据。
“工具已授权”不等于“结论已有数据库证据”；只有
报告 `sources` 中保存了来源标识、检索时间和直接支持的主张时，才能使用
`basis=verified_evidence`。提交样例在数值 Top-5 冻结后完成外部证据补充，20 条原始论文、
供应商目录与 SDS 来源及候选映射保存在 `results/top5_external_evidence.json`。目标规模
和地点未给出，因此预算和放大风险仍为 `unknown`；没有合格来源的判断仍标为
`llm_inference` 或 `unknown`。任意 shell、任意文件和普通网页权限默认拒绝，数据库内容
也不会进入产率预测、EI 或候选排序。

## 2. 输入

### 包内组织方原始数据

固定使用项目相对路径 `data`，其中必须包含 Suzuki 和 Buchwald_sub4 的训练表、无产率
候选表，以及由离线预言器隔离持有的带产率反馈表。正式复现不读取提交目录外的数据。
智能体执行“2 × 20 × 40 正式复现”：2 个数据集各 20 个固定随机种子，每个种子
40 轮单点顺序查询。

### 项目内新数据目录或 manifest

将新数据放在提交目录内（建议为 `data/<task>/`）。manifest 引用带产率训练表、不含
结果列的候选表、字段类型、单位和已审核结构映射。
若没有 manifest，`inspect_input` 先返回不含结果值的表元数据，DeepSeek 形成 manifest
草案，`new_dataset(action=draft_manifest)` 再执行确定性校验。
系统先执行 schema、泄漏、重复条件、类别/组合新颖性、连续变量漂移、结构覆盖率和
分布外（OOD）审计。新数据轮数至少为 5；每次真实反馈必须携带上一条推荐的
`transaction_id`。

## 3. 输出

### 正式复现

- 40 条轨迹与 1,600 条顺序查询；
- 轨迹审计、基准逐轮比较、四项指标及置信区间；
- 每个数据集的已查询 Top-5；
- 已查询 Top-5 的 DFT 解释（不参与候选排序）；
- 每条已查询 Top-5 反应条件的六维现实可行性评估；
- 每个数据集在存在非 `blocked` 条件时给出一个最终推荐反应条件；全部为
  `blocked` 时输出 `no_condition_recommended`；
- JSON/Markdown 报告、哈希、环境和限制说明。

### 新数据

- 输入/OOD 审计；
- 带事务 ID 的单点实验推荐；
- 可恢复状态与真实反馈记录；
- 轮数完成后的已查询 Top-5 反应条件、六维现实可行性评估，以及非阻断条件中的最终
  推荐反应条件；全部为 `blocked` 时输出 `no_condition_recommended`。

“最终推荐反应条件”只表示实验执行建议。`optimizer_rank` 保留数值排序，
`selected_optimizer_rank` 记录 DeepSeek 从已查询 Top-5 中作出的现实选择。

## 4. 关键文件

| 文件 | 作用 |
|---|---|
| `skill/catalystflow/SKILL.md` | 智能体核心工作流 |
| `skill/catalystflow/references/` | 输入、正式协议、新数据路由与证据政策 |
| `skill/catalystflow/assets/feasibility_assessment_template.json` | 统一现实可行性输出模板 |
| `skill/catalystflow/scripts/workflow_gateway.py` | 正式作业、新数据和报告状态机 |
| `skill/catalystflow/scripts/reproduction_worker.py` | 2 × 20 × 40 执行器 |
| `skill/catalystflow/scripts/report_builder.py` | 轨迹、指标、已查询 Top-5 和 DFT 报告 |
| `.openscience/agent/catalystflow.md` | 智能体模型、权限和行为定义 |
| `agent/mcp_server.py` | 受限 MCP JSON-RPC 服务 |
| `agent/policy_gateway.py` | 结构解析和人工审核边界 |
| `agent/start_mcp.sh` | MCP 启动入口 |

`skill/optimize-coupling-reactions/` 是经过测试的内部数值/结构适配内核，不再注册为
独立 Skill；正式用户入口只有 `catalystflow`。

## 5. 九个 MCP 工具

| 工具 | 作用 |
|---|---|
| `agent_status` | 返回协议、运行环境和边界 |
| `inspect_input` | 识别包内组织方原始数据或审计项目内新数据 manifest |
| `start_reproduction` | 启动新的 2 × 20 × 40 正式复现 |
| `reproduction_status` | 查询作业、轨迹和比较状态；完成后直接返回 B1、B40、t95、AUC 的均值、标准差和 95% 置信区间 |
| `prepare_feasibility_review` | 固定正式复现的已查询 Top-5 和评估模板 |
| `new_dataset` | 执行 draft_manifest/profile/init/recommend/feedback/status |
| `finalize_live_feasibility` | 完成新数据已查询 Top-5 评估和最终推荐反应条件 |
| `finalize_report` | 逐条保存正式数据 10 条已查询 Top-5 评估；全部到齐后保存两条最终推荐反应条件并组装报告 |
| `resolve_structures` | 解析并校验结构；新结果等待人工审核 |

## 6. DeepSeek 赛事网关

项目 `openscience.json` 已配置：

- Base URL：`https://fxb.supa.net.cn:6443/v1`
- 默认模型：`deepseek/deepseek-v4-flash`
- 备用模型：`deepseek/minimax-2.7`
- 超时：3,000,000 ms

API Key 只存到 Open Science 用户级密钥库：

```bash
bash runtime/configure_deepseek.sh
bash runtime/openscience_cli.sh keys list
bash runtime/verify_deepseek.sh
```

通用 DeepSeek 凭据检查可能把只适用于赛事自定义网关的 Key 显示为
`Authentication Fails`；以 `verify_deepseek.sh` 对项目端点返回的 `AUTH_OK` 为准。
Web 启动器会为提交根目录建立独立的本地 Git 项目边界。应使用本次启动输出的 URL
新建会话，不要复用指向上级项目的旧 `prj_.../session/...` URL。

不要把 Key 写入 JSON、`.env`、脚本、日志或提交包。“MCP 已接入”只说明本地工具连接
成功，不代表 DeepSeek 端点、模型名、额度和工具权限都有效。完整端到端运行还要求：

1. `openscience.json` 的 OpenAI 兼容地址含 `/v1`；
2. 模型名与赛事网关一致；
3. 智能体同时允许 MCP 名称空间和 `catalystflow_*` 真实工具名；
4. Key 有效且有额度；
5. Open Science 进程重新启动并读取新配置。

## 7. 运行

正式复现并生成完整报告：

```bash
bash runtime/run_openscience.sh
```

CLI 长任务采用两个会话：第一会话只运行 2 × 20 × 40 数值复现并返回 `job_id`；第二会话
以该 `job_id` 执行 Top-5 评估和 10+1 次增量报告提交。避免使用不带明确 session ID 的
`-c`，以免接入其他历史任务。具体命令见 `doc/reproduction_guide.md` 第 2.3 节。

当前验证的 Open Science 2.0.23 不要使用 `openscience web .`；该形式会显示帮助并退出。
启动器只调用提交包内经过 SHA-256 校验的官方 Open Science 2.0.23 Linux x86_64
运行时，不读取系统 `PATH` 中的同名程序。引擎来源、许可证与支持边界见
`runtime/openscience/README.md`。

该智能体入口原生支持 Linux x86_64。Windows 评委应在 WSL2/Ubuntu 中执行同一组
Bash 命令；Windows 原生 PowerShell 启动器只负责 2 × 20 × 40 数值复现。无论使用
Linux 原生还是 WSL2，输入均固定为提交目录内 `data/`，不需要修改 Agent、Skill、
MCP 或固定模型配置。

项目配置将 `catalystflow` 设为默认主智能体。Open Science 2.0.23 的 Web 界面不提供
单独的主智能体选择器；进入项目工作区并新建会话后直接发送：

```text
检查项目内 data，执行 2 × 20 × 40 正式复现实验；完成后评估全部已查询 Top-5 反应条件；存在非 blocked 条件时为每个数据集给出一个最终推荐反应条件，否则输出 no_condition_recommended。
```

智能体默认模型已固定为 `deepseek/deepseek-v4-flash`。无图形界面的宿主可使用
`openscience serve` 启动服务，再由兼容客户端选择同名智能体。

现实可行性评估采用增量工具调用：每次仅提交一条条件的六维评估，网关返回
`received_count` 和剩余条件；10 条全部通过后再提交两条最终推荐。不要把全部条件放入
一次工具调用。该协议可断点续传，并避免结构化参数超过模型输出上限。

默认使用 `execution_mode=inline`。只有宿主明确提供持久进程管理时才使用
`detached`，随后用 `reproduction_status` 查询。

新数据有 manifest 时按下列顺序；无 manifest 时先执行
`inspect_input → DeepSeek manifest 草案 → draft_manifest`：

```text
profile → resolve missing structures → init → recommend → real experiment
       → feedback(transaction_id, observed_yield) → ... → finalize_live_feasibility
```

## 8. 现实可行性与证据

DeepSeek 对已查询 Top-5 反应条件的六个维度逐项输出：

- `budget`：相对试剂和条件成本；
- `procurement_difficulty`：供应商覆盖、库存、交期和定制需求；
- `hazard`：初步操作危险性，不等于 SDS 或 EHS 结论。
- `operational_complexity`：装置、控制、后处理和分离复杂度；
- `scale_up_risk`：传热、传质、放热、产气和压力风险；
- `waste_disposal_burden`：废物流分类和处置负担。

每个维度包含 `level`、`basis`、`confidence`、`rationale` 和 `sources`。
`basis=llm_inference` 的置信度不得为 high；预算/采购的核实证据需要供应商目录或
项目采购凭证，危险性的核实证据需要官方 SDS/GHS。执行前仍需报价、SDS、SOP 和 EHS
复核。程序按 `low=3`、`medium=2`、`high=1`、`unknown=null` 计算总分、覆盖度和
安全门槛；观测产率不计入该分数，也不生成第二套排序。
若任一维度为 `unknown`，总分显示为破折号；例如 `—（4/6）` 表示仅四个维度可评分，
另外两个维度未知，括号内为已评分维度数/总维度数。

每次 Top-5 报告先执行一次数据库清单查询，再对两组 Top-5 的重复物质和主张去重检索；
搜索时显式设置返回条数，并读取拟引用记录。化学数据库可核验身份、结构和部分性质，
文献数据库可支持反应与操作背景；它们不是供应商报价库或官方 SDS 库。没有直接证据、
服务不可用或来源类型不满足该维度要求时，才使用 `llm_inference` 或 `unknown`，不得编造来源。

## 9. 验收

```bash
runtime/.portable_env/bin/python -m unittest discover -s skill/catalystflow/tests -v
runtime/.portable_env/bin/python -m unittest discover -s agent/tests -v

bash runtime/openscience_cli.sh debug skill
bash runtime/openscience_cli.sh debug agent catalystflow
```

运行缓存写入 `.catalystflow-agent/`，不属于提交清单；重新运行即可生成。正式提交将
一次完整智能体输出与数值结果合并保存在 `results/results_and_analysis.pdf`，同名
DOCX/Markdown 用于审阅和复用，JSON 为机器可读版本。

完整复现的程序化验收条件为：40 条轨迹、1,600 条查询记录、提交基准比较
`identical`、10 条已查询 Top-5 现实可行性评估、两条最终推荐以及
`optimizer_ranking_preserved=true`。DeepSeek 的定性说明可能随调用变化，但不得改变
轨迹、核心指标或 `optimizer_rank`。
