# CatalystFlow 队伍 DFT 数据：计算内容、用途与边界

本目录保存项目自建的密度泛函理论（Density Functional Theory，DFT）电子结构数据库。它为化学机理与模型行为提供独立证据，不是测试产率的替代品。

## 数据概况

| 项目 | 内容 |
|---|---|
| 完成计算 | 86 条（Buchwald 52，Suzuki 34） |
| 理论层级 | B3LYP-D3(BJ)/def2-SVPD |
| 三维结构 | GaussView 6 搭建并人工检查 |
| 电子结构软件 | ORCA 6.1.0-f.0 |
| 数据库文件 | 552 个，包括输入、输出、波函数、结构和聚合特征 |

## DFT 在项目中的两种作用

1. **研发层**：电子结构信息用于反应机理对齐、特征假设和真实消融试验，帮助判断哪些电子效应值得建模。
2. **解释层**：最终模型选定候选后，解释器才读取同角色 DFT 百分位，输出支持证据、风险、覆盖率和可信度。

最终冻结排序器不直接读取原始 DFT 数值，解释结果不写回模型、不重新排列候选，并明确记录 `ranking_effect: none`。这保证报告不会把选择后的机理证据误写成模型的数值选择原因。

## 从哪里开始看

- `orca_electronic_structure_database/README.md`：数据库字段和文件结构；
- `../results/audits/mechanistic_dft_component_features.csv`：冻结解释特征表；
- `../results/dft_explanations/dft_explanation_examples.md`：真实轨迹解释案例；
- `../results/dft_explanations/dft_coverage_summary.csv`：角色覆盖率；
- `../code/dft_explainer.py`：只读、选择后解释器实现。

## 防泄漏边界

`source_tables_leak_safe/` 只含允许的训练表和无产率候选表。DFT 子目录不复制测试产率表或完整搜索空间表；提交包的官方原始数据统一位于 `data/`，并受离线预言器（`OfflineOracle`）边界控制。分子总能量不会跨不同分子直接比较；解释使用同化学角色的百分位与稳健相对位置。

## 软件引用

1. Neese, F. Software Update: The ORCA Program System—Version 6.0. *WIREs Comput. Mol. Sci.*, **2025**, *15* (2), e70019. https://doi.org/10.1002/wcms.70019.
2. Dennington, R.; Keith, T. A.; Millam, J. M. *GaussView*, Version 6; Semichem Inc.: Shawnee Mission, KS, 2016.
