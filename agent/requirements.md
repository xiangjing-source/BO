# CatalystFlow 智能体行为契约

本文件定义 Open Science 智能体必须满足的输入、输出和边界。详细使用说明见
`agent/README.md`，工作流事实源见 `skill/catalystflow/SKILL.md`。

## 1. 角色分工

- **最终模型**：唯一负责产率模型、不确定性、期望改进、候选排序和单点选择；
- **离线预言器（OfflineOracle）或真实实验**：候选索引确定后只返回一条产率；
- **Open Science/DeepSeek**：任务路由、结构/单位语义、科学证据和中文报告；
- **DFT 解释器**：候选排序完成后报告支持、风险、覆盖率和可信度，不参与候选排序；
- **DeepSeek 现实可行性评估**：对已查询 Top-5 反应条件评估预算、采购难度、初步危险性、
  操作复杂度、放大风险和废物处置负担，并从非 `blocked` 条件中选择最终推荐反应条件；
  全部为 `blocked` 时输出 `no_condition_recommended`。

DeepSeek 与 DFT 的 `ranking_effect` 必须为 `none`。

## 2. 输入

### 2.1 组织方数据正式复现

输入为包含 Suzuki 与 Buchwald_sub4 训练表、无产率候选表，以及由离线预言器隔离持有的带产率反馈表的项目
相对目录。必须执行 2 × 20 × 40 正式复现，不得以随包轨迹替代新运行。

### 2.2 新数据

输入为 manifest 及其引用的：带产率训练表、无结果列候选表、类别/连续/分子字段、
单位和已审核结构映射。必须先 profile；有 blocker 时不得初始化。真实反馈必须携带
待执行推荐的 `transaction_id`。

### 2.3 现实可行性评估

输入只能是数值运行完成后的已查询 Top-5 反应条件和 DFT 解释。不得加入未查询
候选、伪产率、替代排名或新的数值分数。

## 3. 输出

正式复现必须输出 40 条轨迹、1,600 条查询、轨迹审计、基准比较、四项指标、每组
已查询 Top-5 反应条件、DFT 解释、10 条六维现实可行性评估，并在存在非 `blocked`
条件时为每组输出一个最终推荐反应条件及 JSON/Markdown 报告；全部为 `blocked` 时输出
`no_condition_recommended`。

正式数据的 10 条六维评估必须逐条提交和持久化；全部条目通过确定性校验后，才能单独
提交两个最终推荐反应条件并组装完整报告。不得要求模型在一个工具调用中生成全部
10 条嵌套评估。

新数据必须输出输入/OOD 审计、事务化单点推荐、反馈状态；轮数完成后输出 5 条六维
现实可行性评估、一个非阻断最终推荐反应条件及 JSON/Markdown 报告；全部为 `blocked`
时输出 `no_condition_recommended`。

`optimizer_rank` 是不可变的数值排序；`selected_optimizer_rank` 是最终推荐反应条件的
执行选择，不是新的产率排名。

## 4. 六维评估模板

每条已查询 Top-5 反应条件必须分别包含：

- `budget`：相对试剂和条件成本；
- `procurement_difficulty`：供应商覆盖、库存、交期和定制需求；
- `hazard`：初步操作危险性。
- `operational_complexity`：装置、控制、后处理和分离复杂度；
- `scale_up_risk`：传热、传质、放热、产气和压力风险；
- `waste_disposal_burden`：废物流分类和处置负担。

每个维度包含 `level`、`basis`、`confidence`、`rationale`、`sources`：

- `basis=verified_evidence`：必须有符合维度要求的来源；
- `basis=llm_inference`：只能使用低或中置信度，不得冒充实时报价、库存或 SDS；
- `basis=unknown`：`level` 同时为 `unknown`，不附虚构来源。

每次 Top-5 报告必须先调用数据库清单，对重复物质和主张去重后检索，并读取拟引用记录。
化学数据库用于身份、结构和性质核验，文献数据库用于直接相关的反应与操作依据；检索
未命中、服务不可用或来源不能直接支持当前维度时，才使用 `llm_inference` 或 `unknown`。
化学与文献数据库不能替代供应商报价或官方 SDS，所有检索均在优化器排名固定后执行。

完整结构见 `skill/catalystflow/assets/feasibility_assessment_template.json`。
程序固定映射 `low=3`、`medium=2`、`high=1`、`unknown=null`，并计算总分、覆盖度和
安全门槛；观测产率不进入该总分。

## 5. 不可违反的边界

- 不读取、请求、推断或复原未查询候选的产率；
- 不把预测产率当作真实反馈，不批量调用 oracle；
- 不修改固定模型配置、模型参数、候选索引、期望改进或优化器排序；
- 不编造结构、CAS、价格、库存、危险性、DFT 数值或文献来源；
- 新结构只有经过 RDKit 校验和人工审核后才能进入数值特征管道；
- DFT 解释不参与候选排序，也不作因果证明；
- 最终安全决策必须核对最新 SDS、实验室 SOP 并取得 EHS 批准；
- API Key 不进入项目文件、日志、状态或报告；
- 不输出比赛自评分或项目评分。

任一边界失败时返回结构化错误或等待人工审核状态，不得用自然语言绕过验证器。
