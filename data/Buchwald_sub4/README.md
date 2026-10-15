# Buchwald-Hartwig C-N 偶联反应优化评测集 — Buchwald_sub4（外部评测集）

## 数据集信息

| 属性 | 值 |
| --- | --- |
| 反应类型 | Buchwald-Hartwig C-N 偶联反应 |
| 评测用途 | 化学合成反应条件优化竞赛评测集 |
| 评测分组 | 外部评测集，公开 |
| 优化目标 | 最大化 `Yield` (%) |
| 优化变量数量 | 4 |
| 固定组分或条件数量 | 4 |
| 本产物完整搜索空间样本数 | 790 |
| 本产物原始训练集样本数 | 7 |
| 本产物测试样本数 | 783 |
| 公开命名规范 | 试剂与反应组分值仅使用 IUPAC 名称，未加入试剂统一为 `Nothing` |

## 数据来源与文献索引

本评测集整理已发表的高通量 Buchwald-Hartwig C-N 偶联反应数据，用于离散反应条件空间中的离线黑盒优化评测。当前公开文件对源数据进行了列名规范化、IUPAC 名称映射、固定训练/测试集切分和防泄漏候选池整理。

数据来源文献：

Ahneman D T, Estrada J G, Lin S, et al. Predicting reaction performance in C–N cross-coupling using machine learning[J]. Science, 2018, 360(6385): 186-190. [DOI: 10.1126/science.aar5169](https://www.science.org/doi/10.1126/science.aar5169)

## 反应任务描述

Buchwald-Hartwig C-N 偶联反应通过钯催化将芳基卤代底物与胺类亲核试剂偶联形成新的 C-N 键，是药物分子、功能材料和含氮芳香族化合物合成中的关键反应。本任务要求优化算法在离散候选空间中选择芳基卤代物、配体、添加剂和碱的组合，以高效发现高产率条件。

本数据集中芳基卤代物包含氯、溴、碘代物，可用以下概式表示：
（注：基于反应特性，尽管使用的芳基卤代物不同，但其主产物相同）

```text
Ar-X + Ar'-NH2  →  Ar-NH-Ar'
        [Pd catalyst / ligand / additive / base / solvent]
```

优化目标为 `Yield` (%)。算法需要在固定训练/测试划分下，基于少量初始带标签训练实验数据，在未标注的测试候选池中识别高产率的离散反应条件组合。

## 固定反应组分与实验条件

| 固定项 | 固定值 |
| --- | --- |
| `Reactant1` | 4-methylaniline |
| `Product` | N-(4-ethylphenyl)-4-methylaniline |
| `Solvent` | methylsulfinylmethane |
| `Catalyst` | palladium(2+) diacetate |

## 文件说明

| 文件 | 面向对象 | 说明 |
| --- | --- | --- |
| `buchwald_sub4_searchspace.csv` | 组织者 | 本产物子集完整的 Buchwald-Hartwig 反应条件搜索空间，包含 `Yield`。 |
| `buchwald_sub4_train.csv` | 参赛者 | 初始化阶段唯一可用的带标签先验数据，包含 `Yield`。**注意：本文件已合并全部 5 个 Buchwald 子集的训练样本，共 35 条，详见下方训练集特殊处理说明。** |
| `buchwald_sub4_test.csv` | 组织者/参赛者 | 本产物子集带标签测试数据；作为每轮实验推荐的真实值反馈依据。 |
| `buchwald_sub4_test_features.csv` | 参赛者 | 本产物子集不含 `Yield` 的测试候选池，建议作为公开探索空间发布。 |
| `options.json` | 参赛者 | 离散反应变量候选项配置，仅包含 IUPAC 名称列表（含固定组分）。 |
| `README.md` | 组织者/参赛者 | 数据说明、使用规则、实验配置与提交格式。 |

## 列名说明

| 列名 | 说明 |
| --- | --- |
| `Product` | 目标产物；以 IUPAC 名称表示。本产物子集中该列值固定为 `N-(4-ethylphenyl)-4-methylaniline`。 |
| `Reactant2` | 芳基卤代物试剂；以 IUPAC 名称表示。 |
| `Ligand` | 钯催化体系中的配体；以 IUPAC 名称表示。 |
| `Additive` | 添加剂；以 IUPAC 名称表示。 |
| `Base` | 碱；以 IUPAC 名称表示，`Nothing` 表示未加入该类试剂。 |
| `Yield` | 反应产率 (%)，优化目标。 |
| 备注 | 固定组分 `Reactant1`、`Solvent` 和 `Catalyst` 未纳入 CSV 列中，若有需要可在 `options.json` 中查阅其固定取值。 |

## 数据切分说明

### 本产物子集训练/测试集切分

完整搜索空间保留源 CSV 的原始行顺序。原始训练集根据特定的索引抽取，本产物共 7 条，相同反应类型但为其他产物共28条（**详细解释见“训练集特殊处理说明（重要）”**）；测试集使用本产物未进入训练集的行，共 783 条。本产物的训练集和测试集索引不重叠，且二者样本数之和等于本产物完整搜索空间样本数。

`buchwald_sub4_test_features.csv` 与 `buchwald_sub4_test.csv` 行顺序一致，但移除了 `Yield` 列。参赛者应使用该文件作为公开候选池；组织者/参赛者使用 `buchwald_sub4_test.csv` 离线返回被查询点的真实产率。

### 训练集特殊处理说明（重要）

本评测集包含 5 个 Buchwald 子集（Buchwald_sub1 至 Buchwald_sub5），它们共享相同的反应类型（Buchwald-Hartwig C-N 偶联）、相同的优化变量（`Reactant2`、`Ligand`、`Additive`、`Base`）以及相同的固定组分（`Reactant1`、`Solvent`、`Catalyst`），但各自对应不同的目标产物 `Product`。

为充分利用跨产物结构信息，**每个子集文件夹中的 `buchwald_sub4_train.csv` 并非仅包含本产物的 7 条原始训练样本，而是合并了全部 5 个 Buchwald 子集的训练样本，共计 35 条（5 × 7）。** 合并后训练集的 `Product` 分布如下：

| Product | 训练样本数 | 来源子集 |
| --- | ---: | --- |
| 4-methyl-N-[4-(trifluoromethyl)phenyl]aniline | 7 | Buchwald_sub1 |
| N-(4-methylphenyl)pyridin-2-amine | 7 | Buchwald_sub2 |
| N-(4-methylphenyl)pyridin-3-amine | 7 | Buchwald_sub3 |
| N-(4-ethylphenyl)-4-methylaniline | 7 | Buchwald_sub4 |
| N-(4-methoxyphenyl)-4-methylaniline | 7 | Buchwald_sub5 |

参赛者在使用 `buchwald_sub4_train.csv` 时，可通过 `Product` 列区分不同产物的训练样本。本处理方式的优势在于：参赛者可利用跨产物的反应条件-产率关系来辅助对本产物反应空间的探索，例如通过多任务学习或元学习策略提升有限先验数据下的优化效率。

**注意：** 上述训练集合并仅影响 `buchwald_sub4_train.csv`。`buchwald_sub4_searchspace.csv`、`buchwald_sub4_test.csv` 和 `buchwald_sub4_test_features.csv` 仅包含本产物子集对应的数据，`Product` 列值固定。

## 数据统计

### 本产物搜索空间 Yield 统计

| 指标 | 完整搜索空间 | 原始训练集（7 条） | 测试集（783 条） |
| --- | ---: | ---: | ---: |
| Count | 790 | 7 | 783 |
| Mean | 33.39 | 13.95 | 33.57 |
| Std | 29.15 | 13.50 | 29.20 |
| Min | 0.00 | 3.86 | 0.00 |
| 25% | 4.58 | 5.58 | 4.58 |
| 50% | 25.86 | 8.12 | 26.50 |
| 75% | 63.72 | 14.56 | 63.88 |
| Max | 86.60 | 45.39 | 86.60 |

### 合并训练集 Yield 统计（35 条）

| 指标 | 值 |
| --- | ---: |
| Count | 35 |
| Mean | 20.16 |
| Std | 20.23 |
| Min | 0.60 |
| 25% | 5.22 |
| 50% | 11.54 |
| 75% | 31.43 |
| Max | 81.79 |

## 变量空间统计

| 优化变量 | 候选数量 |
| --- | ---: |
| `Reactant2` | 3 |
| `Ligand` | 4 |
| `Additive` | 22 |
| `Base` | 3 |

## 使用规则与防泄漏说明

1. 参赛者可以使用 `buchwald_sub4_train.csv` 作为初始化阶段的带标签先验数据。
2. 参赛者可以使用 `options.json` 作为反应条件候选配置。
3. 参赛者可以使用 `buchwald_sub4_test_features.csv` 作为公开测试候选池进行优化搜索。
4. 参赛者不得将 `buchwald_sub4_test.csv` 中的 `Yield` 用于训练、特征工程、超参数调节、提示词构造、候选排序或任何其他优化决策。
5. `buchwald_sub4_test.csv` 应被视为组织者/参赛者侧结果查询文件，而不是参赛者侧输入文件。
6. 比赛**设置内部隐藏评测数据集，用于检测排行榜过拟合和数据泄漏**。
7. 内部隐藏评测数据可用于辅助最终排名，最终排名可能会同时考虑公开评测集与内部隐藏评测集表现。
8. 在结果查询前，任何直接或间接使用隐藏标签、测试标签或由标签派生信息的行为都应视为数据泄漏。
9. 参赛者可使用跨产物训练样本（合并训练集中的其他 Product 对应行）辅助建模，但不得将其他产物的测试集 `Yield` 用于本产物的优化决策。

## 实验配置

| 配置项 | 固定设置 |
| --- | --- |
| 初始化 | 仅提供 `buchwald_sub4_train.csv`（合并 35 条）作为带标签先验数据 |
| 候选池 | 提供 `buchwald_sub4_test_features.csv` 作为未标注候选池 |
| 结果查询 | 组织者/参赛者使用 `buchwald_sub4_test.csv` 为被查询点返回 `Yield` |
| 随机种子 | [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000] |
| 独立优化运行次数 | 20 |
| 每次运行优化迭代数 | 40 |
| 每轮采样预算 | 1 个数据点 |
| 每次运行查询测试点总数 | 40 |
| 20 次运行累计查询点数 | 800 |

## 输入格式

算法输入应包含：

- `buchwald_sub4_train.csv`：初始带标签观测；
- `buchwald_sub4_test_features.csv`：可推荐的未标注候选池；
- `options.json`：离散变量取值空间（可选）。

## 输出结果提交格式

参赛者应提交 20 个 `.pt` 文件，每个随机种子/独立运行对应一个文件。文件名必须与固定种子列表一致：

```text
seed_100.pt
seed_200.pt
...
seed_2000.pt
```

每个 `.pt` 文件应能重建该随机种子下 40 步优化轨迹。推荐保存为长度为 40 的记录列表，或保存为包含 `trajectory` 字段的字典。每一步建议包含：

| 字段 | 要求 |
| --- | --- |
| `step` | 1 到 40 的整数 |
| `query_index` | `buchwald_sub4_test_features.csv` / `buchwald_sub4_test.csv` 中的行索引 |
| `condition` | 反应条件变量到 IUPAC 取值的字典 |
| `observed_yield` 或 `actual_yield` | 本地离线评测时查询结果返回后记录 |
| `predicted_yield` | 可选；模型预测的产率 |

推荐 `.pt` 对象示例：

```python
{
    "seed": 100,
    "dataset": "Buchwald_sub4",
    "trajectory": [
        {
            "step": 1,
            "query_index": 123,
            "condition": {
                "Reactant2": "...",
                "Ligand": "...",
                "Additive": "...",
                "Base": "..."
            },
            "observed_yield": 53.6,
            "predicted_yield(optional)": 45.6
        }
    ]
}
```

提交文件需满足以下校验要求：

1. 每个文件必须包含恰好 40 个被查询点。
2. `query_index` 必须指向 `buchwald_sub4_test_features.csv` 中的有效行。
3. 同一运行内的 `query_index` 不应重复。
4. 必须提交恰好 20 个 `.pt` 文件。
5. 文件名必须匹配固定种子列表。
6. 提交的 `condition` 必须与对应 `query_index` 行中的反应条件一致。

## 核心优化指标参考

1. **首轮最大值 `initial_round_found_best`**：第一轮推荐并获得反馈后，已发现目标值中的最大值。该指标越高越好，反映初始设计、热启动或先验知识利用质量。
2. **全程最优值 `best_found`**：完整优化预算内发现的最高目标值。该指标越高越好，是衡量有限实验预算下最终发现能力的核心指标。
3. **达到 95% 全局最优的轮数 `round_to_95_global_best` / `t95`**：第一次达到该数据集全局最优值 95% 的轮数。该指标越低越好；若预算内未达到，可记为大于最大轮数或按主办方指定惩罚值处理。它反映算法接近优质实验条件的速度。
4. **best-so-far 优化曲线面积 `AUC_best_so_far`**：对每轮截至当前的历史最优值序列计算面积或平均面积。该指标越高越好，能同时体现早期发现能力和后续稳定提升能力，避免只看最后一轮结果。

所有指标建议按 20 次独立运行分别计算，并报告均值、标准差和 95% 置信区间。

## 目录结构

```text
Buchwald_sub4/
├── buchwald_sub4_searchspace.csv
├── buchwald_sub4_train.csv
├── buchwald_sub4_test.csv
├── buchwald_sub4_test_features.csv
├── options.json
└── README.md
```
