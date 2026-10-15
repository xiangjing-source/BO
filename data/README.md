# CatalystFlow 队伍提交：组织方公开数据与访问边界

本目录随提交包保留组织方发布的两套完整公开数据，目的是让评委无需再次下载即可
执行 2 × 20 × 40 正式复现。文件存在不等于模型可以任意读取：代码将“优化器输入”
与“离线预言器（OfflineOracle）”严格分离。

最容易误解的一点是：`test.csv` 虽然为了离线复现在磁盘上，但不会作为
DataFrame 传给模型。它相当于封存的实验仪器记录，只能在模型先提交一个候选
索引后，由离线预言器（`OfflineOracle.query(index)`）返回这一行的产率。

## 目录与权限

```text
data/
├── README.md
├── Suzuki/
│   ├── README.md
│   ├── options.json
│   ├── suzuki_train.csv
│   ├── suzuki_test_features.csv
│   ├── suzuki_test.csv
│   └── suzuki_searchspace.csv
└── Buchwald_sub4/
    ├── README.md
    ├── options.json
    ├── buchwald_sub4_train.csv
    ├── buchwald_sub4_test_features.csv
    ├── buchwald_sub4_test.csv
    └── buchwald_sub4_searchspace.csv
```

| 文件类型 | 允许的访问方式 |
|---|---|
| `*_train.csv` | 初始带标签训练数据，优化器可读 |
| `*_test_features.csv` | 不含 Yield 的候选池，优化器可读并排序 |
| `*_test.csv` | 只由 `OfflineOracle` 持有；候选索引确定后，`query(index)` 仅返回该行 Yield |
| `*_searchspace.csv` | 原始公开数据完整性与人工审计附件；生产入口不读取 |
| `options.json` | 公开变量取值说明，可用于 schema 检查 |

一次合法交互如下：

```text
优化器读取 train.csv + test_features.csv
        │
        ├─ 对未查询候选预测并确定索引 i
        │
        └─ OfflineOracle.query(i)
                         │
                         └─ 只返回第 i 行 Yield，加入本 seed 历史
```

赛事规则禁止将 `*_test.csv` 中的 Yield 用于训练、特征工程、超参数调节、提示词构造、候选排序或其他优化决策。本项目通过以下代码边界执行该要求：

1. `load_public_dataset()` 只打开训练表和无标签候选表，并拒绝候选表含 `Yield`；
2. 优化器对象只接收 `PublicDataset`，其中不存在测试 Yield；
3. 候选索引完成选择后，`OfflineOracle.query(index)` 才释放一条反馈；
4. 每个 seed 拥有独立 oracle 和已查询索引集合；
5. 全局最高产率只在 40 轮轨迹完成后用于计算 t95；
6. `searchspace.csv` 不在生产数据映射 `DATASET_FILES` 中。

因此，完整公开数据被包含是为了“一键离线复现”，不是扩大模型权限。若在真实实验或隐藏评测环境运行，可由组织方用同一 `OfflineOracle` 接口替换本地测试表。

## 直接复现

从提交目录运行：

```bash
python code/main/run_selected.py \
  --data-root data \
  --results-root reproduced_results \
  --dataset all --rounds 40 --seeds 20
```

公开数据原始来源和具体使用规则见两个子目录中的组织方 `README.md`。这些
README 和 CSV 属于来源记录，本提交保持原文与原字节不变；面向本项目的解释
集中在本页。所有文件均纳入 `MANIFEST.json` 的 SHA-256 清单。
