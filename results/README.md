# CatalystFlow 结果与证据导航

本目录保存最终模型的 2 × 20 × 40 正式复现实验结果，以及生成报告图所需的逐轮
数据。它不是调参工作区；所有统计均在固定模型配置和轨迹完成后计算。

统一术语：两个数据集各运行 20 个固定随机种子、每个种子 40 轮，称为
“2 × 20 × 40 正式复现实验”；轨迹完成后从合法查询记录汇总的五条高产率条件称为
“已查询 Top-5 反应条件”；DeepSeek 从中基于六维现实可行性评估选择的执行建议称为
“最终推荐反应条件”。

文中的“代理模型”是以较低成本近似“反应条件→产率”关系的预测模型；“代理”
强调其在顺序优化中代替昂贵实验或离线反馈接口，输出预测均值和不确定性。
离线预言器（OfflineOracle）仅在候选索引确定后返回对应产率，未查询候选产率不参与候选选择。

## 四项组织方评测指标

| 指标 | 含义 | 方向 |
|---|---|---|
| `initial_round_found_best` | 第 1 轮反馈后已发现的最高产率 | 越高越好 |
| `best_found` | 40 轮预算结束时发现的最高产率 | 越高越好 |
| `round_to_95_global_best` / t95 | 首次达到候选池全局最优值 95% 的轮数；未达到记为 41 | 越低越好 |
| `AUC_best_so_far` | 40 个 best-so-far 值的算术平均，兼顾早期速度与最终水平 | 越高越好 |

正式均值如下；标准差和 95% 置信区间见 `summary_metrics.csv`。

| 数据集 | 首轮最大值 | 最终最好值 | t95 | AUC |
|---|---:|---:|---:|---:|
| Suzuki | 84.043 | 97.363 | 4.4 | 95.884 |
| Buchwald–Hartwig（Buchwald_sub4） | 74.388 | 86.067 | 12.5 | 82.535 |

候选池全局最高产率分别为 Suzuki 99.9、Buchwald 86.598。这里的
“全局最高”仅在 40 轮结束后用于评估，不参与推荐。

## 评审材料导航

- `results_and_analysis.docx`：手动定稿的最终结果、智能体分析与推荐内容基准；同名
  PDF 为评审版，Markdown 为可检索文本镜像，JSON 为机器可读版本；
- `figures/framework_overview.png`：技术路线图的独立高分辨率版本；
- `figures/optimization_performance_overview.png`：一页总览；
- `figures/t95_attainment_curves.png`：达到 95% 全局最优的累计比例；
- `figures/core_metrics_confidence_intervals.png`：四项指标的均值与区间；
- `figures/per_seed_best_so_far_heatmaps.png`：每个随机种子的逐轮表现；
- `optimization_trajectories/`：两套数据各 20 条完整 `.pt` 轨迹；
- `dft_explanations/`：正式轨迹的 DFT 解释（不参与候选排序）；
- `audits/`：DFT 特征边界与快照；
- `external_validation/`：外部数据适配测试，不作为组织方数据上的性能结论；
- `top5_external_evidence.json`：Top-5 冻结后的 20 条外部来源、检索日期、直接支持的
  主张和候选映射；这些来源不进入产率预测、EI 或 `optimizer_rank`。

统一报告中的智能体分析是一次固定运行样例。重新调用 DeepSeek 时，预算、采购、安全等定性
说明和 `selected_optimizer_rank` 可能在非 `blocked` 条件之间变化；正式数值轨迹、
四项核心指标和不可变的 `optimizer_rank` 不应变化。

现实可行性总分的完整范围为 6–18。表格中的 `—（4/6）` 表示六个维度中只有四个
具备可评分信息，另外两个为 `unknown`，因此不计算总分；括号内为已评分维度数/总维度数。

## `.pt` 轨迹怎样读取

文件扩展名为 `.pt`，内容实际是 Python 标准库 `pickle`，不需要 PyTorch：

```python
import pickle

with open("results/optimization_trajectories/Suzuki/seed_100.pt", "rb") as f:
    run = pickle.load(f)

print(run["seed"], run["dataset"])
print(run["trajectory"][0])
```

每轮至少记录轮数、候选索引、完整条件、观测产率、best-so-far、模型预测、
不确定性、采集函数值和选择模式。完整字段定义见
`figures/trajectory_schema.json`。

## 怎样读图

实线表示 20 个独立运行的均值，阴影表示均值的 95% 置信区间，而不是所有单次
运行的取值范围。热图每一行对应一个随机种子，可帮助识别均值背后的偶然性或
不稳定种子。所有 PNG 为高分辨率位图；主要结果图同时保留 PDF 版本和生成图形的
CSV 数值表，便于放大审阅与独立重绘。
