# CatalystFlow 复现指南

复现分为数值复现环境和智能体环境。生成 40 条模型轨迹不依赖 DeepSeek；新数据语义
解析和现实可行性评估才使用 DeepSeek。两类环境分开可避免 API 状态影响数值结果。
评委只需将提交包完整解压到可写目录并执行零参数入口，不需要修改源码、固定模型配置
或数据路径。

## 1. 数值复现环境

### 1.1 Linux x86_64

使用随包 Linux 环境归档，并固定读取包内组织方原始数据 `data/`：

```bash
cd CatalystFlow_submission
bash runtime/run_reproduction.sh
```

首次执行会在 `runtime/.portable_env/` 解压环境。该命令分别对 Suzuki 和
Buchwald_sub4 使用 20 个固定随机种子，每个种子从头训练并完成 40 轮顺序查询，
共输出 40 条轨迹和 1,600 条查询记录；新结果写入包内 `reproduced_results/`。
它不加载提交轨迹，也不调用 DeepSeek。不要把文档示例词当作路径参数；当前脚本会在
运行模型前拒绝 `/path/to/...` 等占位符并说明正确的零参数命令。

### 1.2 Windows 10/11 x86_64

在提交根目录打开 PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\runtime\run_reproduction_windows.ps1
```

该脚本使用 `runtime/windows-x86_64/` 中的 Windows 原生离线环境归档，固定读取
`data\` 并写入 `reproduced_results\`；无需安装 Conda 或联网。临时执行策略只在当前
PowerShell 进程生效。归档哈希、关键依赖版本和 Windows 实机验收记录见
`runtime/windows-x86_64/environment-lock.json`。

两个平台的成功标准相同：生成 40 个 `.pt` 轨迹文件、共 1,600 条顺序查询记录；
随后由校验程序检查每条轨迹恰有 40 轮、候选索引不重复、条件与候选表一致。Linux
x86_64 还可与提交轨迹执行逐轨迹精确比较。

### 1.3 手工创建环境（可选）

如需自行创建 Conda 环境：

```bash
conda env create -n catalystflow-test -f code/environment.yml
conda run -n catalystflow-test python code/main/run_selected.py \
  --data-root data \
  --results-root reproduced_results \
  --dataset all --rounds 40 --seeds 20
```

独立环境的名称不影响结果；复现所需关键版本为：
Python 3.11.15、NumPy 1.26.4、SciPy 1.11.4、pandas 2.1.4、
scikit-learn 1.3.2 和 RDKit 2026.03.4。关键是依赖版本、包内数据和固定模型配置一致。

macOS 和 Linux ARM64 的入口见 `runtime/README.md`；它们首次创建环境需要联网，
不属于本提交的离线复现承诺。模型参数仍来自同一
`code/config/production_config.json`。

## 2. 智能体环境

Open Science 2.0.23 Linux x86_64 官方发布归档已包含在 `runtime/openscience/`，因此
Linux x86_64 不需要预装 Open Science、Node.js 或 npm。Windows 评委如需复现完整
智能体，应在 WSL2/Ubuntu 中进入提交目录并执行本节 Bash 命令；Windows 原生
PowerShell 入口只负责数值复现。先执行离线验收：

```bash
bash runtime/verify_offline_bundle.sh
```

该命令校验并解压数值环境和 Open Science 引擎，运行数值环境烟雾测试，并确认包内
Open Science 能发现 `catalystflow (primary)`；它不调用 DeepSeek。

API Key 只保存到评委用户级 Open Science 密钥库，不写入提交目录：

```bash
bash runtime/configure_deepseek.sh
bash runtime/openscience_cli.sh keys list
bash runtime/run_openscience.sh
```

配置脚本保存 Key 后会通过赛事自定义端点要求返回 `AUTH_OK`。Open Science 的通用
DeepSeek 凭据检查可能访问官方 DeepSeek 服务并把赛事 Key 显示为
`Authentication Fails`；这不是本项目端点的最终判定。以
`Competition DeepSeek endpoint authentication: PASS` 为准，也可运行
`bash runtime/verify_deepseek.sh` 单独复核。Key 不写入提交目录。

`runtime/run_openscience.sh` 校验官方归档及原生二进制 SHA-256，首次运行解压到
`runtime/.openscience_runtime/`，并在提交根目录建立独立的本地 Git 项目边界，再从该
目录启动包内引擎。该边界只用于防止 Web 会话错误继承上级 Git 项目及其模型配置，已从
提交清单排除。Open Science 2.0.23
在 `web` 后附加 `.` 时会打印帮助并退出，因此不要使用 `openscience web .`。
无图形环境可执行 `bash runtime/openscience_cli.sh serve`。

Open Science 2.0.23 读取 `openscience.json` 中的 `default_agent=catalystflow`，进入
工作区后会自动使用该主智能体，不需要寻找智能体选择器。使用本次启动命令输出的 URL
新建会话，不要复用旧的 `prj_.../session/...` URL；若指定端口已被占用，省略 `--port`
让 Open Science 自动分配端口。新建会话后发送：

```text
检查项目内 data，
使用最终模型执行 2 × 20 × 40 正式复现实验；复现完成后核验 40 条轨迹和
1,600 条查询记录，比较提交基准，并对两个数据集的已查询 Top-5 反应条件
完成六维现实可行性评估、DFT 解释和最终推荐反应条件报告。
```

### 2.3 CLI 两阶段复现

`openscience_cli.sh run` 一次执行一个会话。为避免 Open Science 2.0.23 在“数值复现、
外部检索和 11 次报告提交”的超长单会话中自动压缩上下文，先只运行数值阶段：

```bash
bash runtime/openscience_cli.sh run \
  --agent catalystflow \
  --model deepseek/deepseek-v4-flash \
  --title cli-numerical-reproduction \
  "检查项目内 data，只执行 official_reproduction 数值阶段；完成后核验40条轨迹、1600条记录和基准一致性，输出 job_id 后停止，不执行现实可行性评估。"
```

记录返回的 `job_id`，再启动新的报告阶段会话：

```bash
bash runtime/openscience_cli.sh run \
  --agent catalystflow \
  --model deepseek/deepseek-v4-flash \
  --title cli-report-stage \
  "不要重跑数值优化。对 job_id=<第一阶段返回值> 完成已查询Top-5六维评估、DFT解释、10次候选增量提交和1次最终推荐提交，生成 enriched_report。"
```

报告阶段默认执行外部数据库优先协议。明确指定“离线模式”时才跳过检索，并把无直接证据
的维度标为 `llm_inference` 或 `unknown`。`-c` 只代表最近会话，可能误接历史任务；故障
恢复应使用 `--session <session_id>`，同时写明 `job_id` 和待完成阶段。成功判据为
`feasibility_status=completed`，且同时存在 `enriched_report.json` 与
`enriched_report.md`。

若某数据集的五条条件均为 `blocked`，报告输出 `no_condition_recommended`，不强制给出
实验条件。

智能体依次调用：

```text
agent_status → inspect_input → start_reproduction
→ prepare_feasibility_review → DeepSeek 评估 → finalize_report
```

`finalize_report` 使用可恢复的增量协议：DeepSeek 按两个数据集各 rank 1–5 的顺序，
每次只提交一条 `candidate_assessment`，共 10 次；网关逐条校验并保存。收到
`received_count=10` 后，再单独提交两条 `final_recommendations`，由程序组装
`enriched_report.json` 和 `enriched_report.md`。该设计避免大型嵌套工具参数超过模型
输出上限；已验证的条目可在 API 中断后继续。

`start_reproduction` 默认使用 `inline`；如宿主明确支持持久后台进程，可用
`detached` 并通过 `reproduction_status(job_id)` 查询。运行输出位于
`.catalystflow-agent/jobs/<job_id>/`，最终报告为
`report/enriched_report.json` 和 `report/enriched_report.md`。

评委确认完整智能体复现成功时，应同时看到：新 `job_id`、`trajectory_files=40`、
`query_records=1600`、`comparison_status=identical`、10 条候选现实可行性评估、两条
最终推荐，以及 `optimizer_ranking_preserved=true`。LLM 生成的定性说明可能随调用
变化；轨迹、核心指标和优化器排名应保持一致。

数值作业完成后，`start_reproduction` 和 `reproduction_status` 会直接返回受限的
`summary_metrics`：两个数据集的 B1、B40、t95、AUC 各自包含 20 次运行均值、样本标准差
和 95% 置信区间。该对象不包含候选池全局最优值或未查询产率；询问正式指标时无需读取
任意文件，也不能用已查询 Top-5 产率替代。

Agent 已授权 Open Science 的 `science_list_dbs`、`science_search`、`science_fetch`、
`provenance_*` 和 `artifact_snapshot`，但这些模块不是正式数值复现的依赖，也不能向
优化器提供产率或排序信息。Top-5 报告的默认顺序为：列出可用数据库，对重复物质和主张
去重，检索化学数据库与文献数据库，读取拟引用记录，保存直接支持主张的来源；数据库
不可用、未命中或来源不能支持该维度时，才使用 `llm_inference` 或 `unknown`。预算和采购
仍需供应商目录或采购记录，危险性仍需官方 SDS/GHS。提交样例在数值 Top-5 冻结后完成
外部证据补充，20 条来源、检索日期、直接支持的主张和候选映射保存在
`results/top5_external_evidence.json`；未获合格来源的维度仍保存为 `llm_inference` 或
`unknown`。新数据出现身份、结构、反应角色或单位歧义时，Agent
还会根据确定性工具返回的 enrichment request 使用数据库模块；检索结果必须保留来源并
经过程序校验或人工审核。

## 3. 新数据

先把新数据复制到提交目录内，建议位置为 `data/<task>/`。后续只向智能体提供项目相对
路径；绝对路径和逃逸出提交目录的相对路径会被拒绝。

有 manifest 时：

```text
inspect_input(manifest) → new_dataset(profile) → 处理 blockers
→ new_dataset(init) → recommend → 实验 → feedback → …
→ finalize_live_feasibility
```

无 manifest 时：

```text
inspect_input(新数据目录)
→ DeepSeek 根据文件名、表头、行数、粗粒度数据类型、程序识别为无结果候选表的
  少量非数值条件代表值、
  用户提供的单位和化学说明形成 manifest 草案
→ new_dataset(draft_manifest)
→ 确定性校验通过后 init → recommend/feedback 闭环
```

DeepSeek 只提出表角色、字段类型、单位和化学角色的语义映射。所有表名、表头和代表值
均作为不可信数据处理；未分类表、数值列、结果列和可能的答案表不提供代表值。候选结果列、重复条件、
可能答案表的角色阻断、单位、结构覆盖率和 OOD 检查由程序执行；存在歧义时停止。未查询候选产率不进入提示词、
训练、模型选择或候选排序。

## 4. 验收条件

- 40 个 `.pt` 文件，每个恰好 40 条连续且不重复的查询；
- 总计 1,600 条记录，`query_index` 与实际输入候选表逐行对应；
- 正式复现与提交基准比较为 `identical` 或报告明确差异；
- 最终模型配置在运行前固定，各随机种子独立初始化并从头训练；
- 六维现实可行性分数和安全门槛由程序计算，DeepSeek 不修改 `optimizer_rank`；
- DFT 解释标明 `ranking_effect=none`；
- 报告中的报价、库存和初步危险性结论注明证据或 `llm_inference/unknown`。

本提交在 2026-08-14 以包内 `data/` 完成最终复核：新生成 40 条轨迹、
1,600 条查询记录均合法，与提交基准逐轨迹、逐轮比较为 `40 identical / 0 different`。
四项汇总指标与 `results/summary_metrics.csv` 一致。

提交包完整性校验：

```bash
runtime/.portable_env/bin/python code/scripts/verify_final_submission.py \
  --package-root .
```

该命令检查赛题要求的材料、包内原始数据、40 条轨迹及逐轮字段、统一结果与智能体分析报告、
数值环境归档、Open Science 官方归档、DFT 边界、PDF/DOCX 有效性和
`MANIFEST.json` 哈希。首次执行数值复现后，
`runtime/.portable_env/` 已自动建立；如尚未建立，可先运行
`bash runtime/bootstrap_portable_env.sh`。

Windows 原生数值复现完成后，可使用对应环境执行同一提交校验器：

```powershell
.\runtime\.portable_env_windows\python.exe .\code\scripts\verify_final_submission.py `
  --package-root .
```

智能体引擎的独立验收命令为：

```bash
bash runtime/verify_openscience_runtime.sh
```

该命令使用临时用户数据目录，避免把本机 Open Science 安装或配置误计入验证结果。
