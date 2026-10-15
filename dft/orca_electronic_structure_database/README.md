# CatalystFlow 队伍 ORCA 电子结构结果数据库

本目录汇总两类反应中已完成并通过检查的 86 条 ORCA 计算：Buchwald_sub4 52 条，Suzuki 34 条。计算层级为 B3LYP-D3(BJ)/def2-SVPD。

## 文件怎样对应一种物质

先从 `calculation_index.csv` 按 `dataset`、`role` 和材料名称定位记录，再进入 `Buchwald_sub4/` 或 `Suzuki/` 下相应分子目录。每个目录通常包含：

- `*.sp.out`：能量、轨道、偶极矩、Hirshfeld 电荷与 Mayer 键级；
- `*.sp.gbw`：ORCA 波函数/分子轨道文件；
- `*.sp.property.txt`：属性文件；
- `optimized.xyz`：优化后的三维结构；
- `*.sp.inp`：单点计算输入；
- `metadata.json`：名称、SMILES、原始字段角色和路径。

聚合表包括 `molecular_features.csv`（分子级能量、前线轨道、能隙、偶极矩）、`atomic_features.csv`（原子级属性）和 `mayer_bond_orders.csv`（键级）。`calculation_index.csv` 是主索引，`pending_calculations.csv` 记录未完成项；本次计划内的中性无争议结构均已完成。

## 与模型和解释器的关系

研发阶段对这些特征做过真实消融；最终冻结排序模型未采用 DFT 直连。生产解释器只在推荐完成后读取聚合快照，按配体、碱、亲电试剂等同角色比较相对百分位。缺失计算会降低覆盖率与可信度，不会被有利数值填补，也不会触发换候选。

## 来源与安全边界

`source_tables_leak_safe/` 只保存训练表、无标签候选表、选项和组织方说明，不含 `test.csv` 或 `searchspace.csv` 的测试产率。`train_occurrences`、`test_feature_occurrences` 和 `source_occurrence_detail` 只记录允许表中的出现位置。
