# CatalystFlow 代码、运行与验证

本目录包含实际决定候选排序的最终模型代码。最常用入口是
`main/run_selected.py`，唯一固定模型配置是 `config/production_config.json`。
复现结果时优先使用提交根目录 `runtime/` 中对应操作系统的一键脚本；审查实现时可按
本页从环境、运行到校验逐步操作。

评委复现正式结果不需要修改本目录文件。Linux x86_64 运行
`bash runtime/run_reproduction.sh`；Windows 10/11 x86_64 在 PowerShell 运行
`.\runtime\run_reproduction_windows.ps1`。以下手工环境和直接 Python 命令仅供代码审查
或二次开发。

本目录只说明“最终模型怎样产生数值轨迹”。完整智能体的输入、输出、已查询
Top-5 反应条件、现实可行性评估和最终推荐反应条件见提交根目录 `README.md` 与
`agent/README.md`；DeepSeek 和 DFT 均不进入本目录的候选排序。

## 40 轮顺序优化闭环

```text
带标签初始数据
      │
      ▼
拟合代理模型 ──► 预测所有未查询候选的均值和不确定性
      ▲                              │
      │                              ▼
加入已查询反馈 ◄── 离线预言器（OfflineOracle） ◄── 期望改进排序并选择一个候选
```

上述流程每个随机种子独立执行 40 次。贝叶斯优化负责顺序更新；期望改进采集
函数用于权衡预测产率和模型不确定性。代理模型只看初始训练数据与本随机种子已经查询的
反馈，不看尚未查询的测试 Yield。

这里的“代理模型”是具体实现为随机森林和/或高斯过程的预测模型；称“代理”是因为它
近似昂贵真实实验或离线预言器的响应，并提供候选排序所需的均值与不确定性。

## 最终路线

`config/production_config.json` 是唯一固定模型配置：

| 数据集 | 最终模型 | 采集策略 |
|---|---|---|
| Suzuki–Miyaura | 预测均值由权重 0.75 的随机森林和权重 0.25 的可交换分类协方差函数的张量积核高斯过程融合 | 以期望改进为主；均匀探索概率在运行前固定为从 0.15 递减到 0.01 |
| Buchwald–Hartwig（内部标识：Buchwald_sub4） | 一个随机森林同时读取独热编码分类变量、标签无关化学知识、32 位 Morgan 分子指纹和物化描述符 | 只按期望改进排序，不叠加均匀随机探索 |

密度泛函理论（Density Functional Theory，DFT）特征直连、梯度提升树、
CatBoost、支持向量回归、稀疏线性回归和大语言模型偏好引导不是最终运行依赖。
固定模型配置未启用 DFT 直连；DFT 仅用于候选查询后的解释。

两套任务在代码层共用输入预检、随机种子隔离、代理模型接口、期望改进、反馈闭环、轨迹和指标实现；`production_config.json` 在第 1 轮前按数据集标识与字段结构固定选择分支。运行中不会依据公开测试反馈改换模型或超参数。当前正式运行入口只支持组织方两套字段结构，未知反应类型会明确报错。

## 数据边界

| 组件 | 允许读取 |
|---|---|
| 优化器 | `train.csv`、`test_features.csv`、固定结构/知识特征、同一 seed 已查询反馈 |
| 离线预言器（OfflineOracle） | `test.csv`，仅在一个索引被选中后返回该行 Yield |
| 指标器 | 完成的 40 步轨迹；结束后才读取全局最优 |
| 人工审计附件 | `searchspace.csv`；正式运行入口不读取 |

`load_public_dataset()` 拒绝候选表中的 outcome 列。每个 seed 使用独立优化器、随机状态、查询集合和反馈历史。
提交包包含组织方发布的完整公开数据，但文件权限由上述代码边界而不是“文件是否存在”决定。

## 文件索引

```text
main/run_selected.py          最终两数据集统一入口
config/production_config.json 固定模型配置
config/mechanistic_preference_priors.json 仅供标签盲偏好模块回归测试
compliant_bo.py               高斯过程、随机森林、候选模型与期望改进
protocol.py                   数据边界、oracle、指标、轨迹
molecular_features.py         Buchwald 结构指纹和物化描述符
chemical_knowledge.py         标签无关化学知识表示
dft_features.py               DFT 特征抽取与历史消融接口
dft_explainer.py              DFT 解释器（不参与候选排序）
scripts/explain_dft_condition.py 解释一条已选智能体推荐
scripts/generate_dft_explanations.py 从已完成轨迹生成案例、覆盖率与事后统计
scripts/render_markdown_for_office.py 文档发布辅助脚本（非模型依赖）
scripts/validate_submission.py
scripts/verify_final_submission.py
logs/README.md               日志与审计文件位置说明
environment.yml               完整开发与复现环境声明
environment-runtime.yml       跨平台启动器使用的最小锁定运行环境
Dockerfile                    容器入口
```

`.pt` 文件是标准库 pickle 对象，不依赖 PyTorch。

## 从零建立环境

只有在不使用 `runtime/` 一键脚本时才需要手工建环境。推荐 conda-forge，
因为化学信息学工具 RDKit 的二进制复现更稳定：

```bash
conda env create -f code/environment.yml
conda activate catalystflow
python -c "import numpy, scipy, pandas, sklearn, rdkit; print(numpy.__version__, scipy.__version__, pandas.__version__, sklearn.__version__, rdkit.__version__)"
```

锁定版本：

- Python 3.11.15
- NumPy 1.26.4
- SciPy 1.11.4
- Pandas 2.1.4
- scikit-learn 1.3.2
- RDKit 2026.03.4
- Matplotlib 3.11.1
- Pillow 12.3.0

正式模型无需 XGBoost、CatBoost、PyTorch、ORCA、图形处理器或网络。

## 运行 2 × 20 × 40 正式复现

数据根目录：

```text
data/
├── Suzuki/                    # 组织方 6 个原始文件
└── Buchwald_sub4/             # 组织方 6 个原始文件
```

运行：

```bash
python code/main/run_selected.py \
  --data-root data \
  --results-root reproduced_results \
  --dataset all --rounds 40 --seeds 20
```

固定 seeds 为 100、200、……、2000。

## 校验

```bash
python code/scripts/validate_submission.py \
  --dataset-dir data/Suzuki \
  --dataset-name Suzuki \
  --results-dir reproduced_results/Suzuki

python code/scripts/validate_submission.py \
  --dataset-dir data/Buchwald_sub4 \
  --dataset-name Buchwald_sub4 \
  --results-dir reproduced_results/Buchwald_sub4

python -m unittest discover -s code/tests -p 'test_*.py'
```

校验器检查 20 个 seed、每个 40 步、query index 有效且不重复、condition 与候选行一致。

## 图表与补充统计的审计边界

提交包保留最终图、矢量 PDF、逐随机种子统计和所有绘图数值表。它们位于
`results/figures/`，其中 `model_progression_curves.csv` 是逐轮曲线数据，
`model_progression_metrics_per_seed.csv` 是逐种子四指标，
`model_progression_metrics_summary.csv` 是均值、标准差和 95% 置信区间。
这些内容是在固定模型配置确定且 40 轮轨迹完成后生成，不返回优化器，也不参与参数选择或候选排序。

## DFT 解释（不参与候选排序）

解释器只在候选已经选定后工作。对 Skill/Agent 生成的
`next_experiment.json` 可运行：

```bash
python code/scripts/explain_dft_condition.py \
  --dataset Suzuki \
  --recommendation next_experiment.json \
  --feature-table results/audits/mechanistic_dft_component_features.csv \
  --output next_experiment_dft_explanation.json
```

输入可以是智能体的完整推荐对象，也可以是直接条件字典。输出同时保留模型
预测、不确定性、期望改进和选择模式，并追加同角色 DFT 百分位、支持证据、
风险提示、覆盖率和可信度。命令不读取产率答案表、不修改优化状态，也不影响
当前或后续候选排序。

正式轨迹的可复现批量解释：

```bash
python code/scripts/generate_dft_explanations.py \
  --results-root results \
  --feature-table results/audits/mechanistic_dft_component_features.csv \
  --output-dir results/dft_explanations
```

其中组分—产率相关属于轨迹完成后的探索性统计，并报告多重检验校正结果；
不能用它反向解释成模型选择依据。

## 容器运行

```bash
mkdir -p reproduced_results
docker build -t catalystflow -f code/Dockerfile .
docker run --rm \
  -v "$PWD/reproduced_results:/workspace/reproduced_results" \
  catalystflow \
  --data-root /workspace/data --results-root /workspace/reproduced_results \
  --dataset all --rounds 40 --seeds 20
```

提交包保留组织方原始 `test.csv` 和 `searchspace.csv` 以支持完整离线复现。
`test.csv` 只能由 `OfflineOracle` 在选择完成后按索引释放一条 Yield；
`searchspace.csv` 不进入生产文件映射。`data/README.md` 给出完整权限边界。

## 跨平台一键复现

Linux x86_64 使用包内离线 `conda-pack` 归档：

```bash
bash runtime/run_reproduction.sh
```

首次运行自动解压至 `runtime/.portable_env/`、执行 `conda-unpack`，随后使用包内
`data/` 运行 2 × 20 × 40 正式复现。无需网络、无需安装 conda，也无需重新求解依赖。

Windows 10/11 x86_64（PowerShell，随包原生离线环境）：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\runtime\run_reproduction_windows.ps1
```

macOS Intel 或 Apple Silicon：

```bash
bash runtime/run_reproduction_macos.sh
```

Linux ARM64：

```bash
bash runtime/run_reproduction_linux_arm64.sh
```

Windows 脚本优先使用 `runtime/windows-x86_64/` 中的 `conda-pack` 离线归档；
首次只需解压并运行 `conda-unpack`，无需网络、Conda 或逐个安装包。该归档锁定
经实机验证的 OpenBLAS 后端，并在 Windows 上逐轮复现了全部 40 条 Linux 基准轨迹。
仅当归档被移除时，脚本才回退到包内 micromamba 启动归档及镜像下载策略。
macOS 和 Linux ARM64 会按同一个 `environment-runtime.yml` 创建平台原生环境，
首次需要网络；之后会复用提交目录内的 `.platform_env_*`。正式复现入口为：

```bash
bash runtime/run_reproduction.sh
```

不同操作系统的二进制不能互换。Linux x86_64 归档在最终提交验收中重新解压并完成
端到端轨迹一致性验证；Windows x86_64 归档具有单独的 Windows 实机验收记录，本次
Linux 文档审查只复核其 SHA-256 和归档内容。macOS 与 Linux ARM64 启动器仅完成静态
检查。详细边界见 `runtime/README.md`。

## 外部数据兼容策略

1. 先删除非有限或超出 `[0,100]` 的产率记录，不做截断；
2. 保留所有推荐时可知的条件变量；
3. 类别字段独热或结构化，连续字段只用训练/公开无标签 schema 确定缩放；
4. 不删除“多出来的变量”，否则可能把不同条件折叠成同一表示；
5. 新分子结构按 direct SMILES/InChI → cache → PubChem → OPSIN → RDKit → 人工复核；
6. 未解析或有歧义的结构阻止运行，不静默填零；
7. 外部适配器结果不作为最终模型在新数据上的性能结论。

## 在新的数据集上使用

正式入口 `main/run_selected.py` 只接受两套官方字段结构，这是为了保证官方结果
可精确复现。新的数据应先走 Skill 中的预检、DeepSeek 语义解析和逐轮适配器，
不得通过修改组织方原始文件绕过数据集路由。

### 情形一：同一反应类型、字段与官方数据完全一致

准备：

```text
data/new_task/
├── train.csv             带 Yield 的初始实验
└── candidates.csv        不含 Yield 的候选条件
```

先运行：

```bash
python skill/optimize-coupling-reactions/scripts/preflight_reaction_data.py \
  --train data/new_task/train.csv \
  --candidates data/new_task/candidates.csv \
  --variables Electrophile,Nucleophile,Ligand,Base,Solvent \
  --yield-column Yield \
  --report data/new_task/preflight_report.json
```

Buchwald 数据应把 `--variables` 改为实际使用的全部条件字段。预检通过后，再用
`reaction_agent.py init`、`recommend` 和 `feedback` 逐轮运行。最终模型的设计可以
作为参考，但新数据必须从头训练并单独评估，不能沿用官方四项指标。

### 情形二：同一反应族，但出现额外变量

- 类别变量，例如钯源、胺或催化剂类别：作为新的独立类别字段编码；
- 连续变量，例如温度、停留时间、浓度或催化剂负载：只根据初始训练表和公开
  无标签候选表确定缩放规则，并进入连续变量通道；
- 分子变量：建立经过审查的结构映射，无法解析时停止；
- 重复条件：使用全部字段判断；不能先删掉额外字段再去重。

Suzuki 外部适配器对连续变量使用标准化后的连续核，对类别变量使用类别通道；
Buchwald 外部适配器优先使用结构随机森林。两者保留相同的推荐、反馈、状态和
防泄漏接口，但不声称与最终模型在数值上完全相同。

### 情形三：新的反应类型

当前生产路由器会拒绝未知反应类型。正确流程是：

1. 确定反应角色、连续变量、类别变量和分子字段；
2. 建立只依赖结构和无标签字段的特征注册表；
3. 在该任务自己的训练标签上比较通用混合变量模型、随机森林和适合的数据模型；
4. 使用与正式种子不重叠的训练内伪闭环选择参数；
5. 如有独立外部数据，只把它用于兼容性压力测试；
6. 固定模型配置和哈希后，再进行正式顺序优化。

这体现的是可审计的迁移接口，而不是“任何新反应都无需验证即可获得当前成绩”。

## 校验完整提交目录

```bash
python code/scripts/verify_final_submission.py --package-root .
```

校验器会核对哈希清单、2 × 20 × 40 轨迹、轨迹字段、DFT 索引、PDF/DOCX、
两套组织方完整公开数据、候选—离线预言器（OfflineOracle）行对齐和便携环境哈希。它还通过静态
检查确认生产数据映射不读取 `searchspace.csv`。86 个 ORCA 计算及聚合特征
均纳入哈希清单。
