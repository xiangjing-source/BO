# 日志、轨迹与审计证据

本目录属于 CatalystFlow 队伍提交的代码审计层。

本项目把“能重算结论的结构化证据”作为正式日志，而不重复保存大段控制台文本。
这样可以直接核对每轮选了什么、当时模型知道什么，以及最后如何计算指标。

| 位置 | 可审计内容 |
|---|---|
| `results/optimization_trajectories/` | 每个数据集、每个随机种子的 40 轮选择、反馈、预测与采集值 |
| `results/summary_metrics.csv` | 两套数据的四项正式指标、标准差和 95% 置信区间 |
| `results/figures/trajectory_field_audit.csv` | 轮数、字段、连续 step 和索引唯一性 |
| `results/figures/trajectory_schema.json` | `.pt` 顶层结构与逐轮字段的定义 |
| `results/audits/` | DFT 特征快照、来源边界和哈希 |
| `MANIFEST.json` | 提交包全部文件的 SHA-256 和字节数 |
| `code/logs/reproduction_verification.log` | 最终零参数复现的协议、环境、轨迹比较与指标摘要 |

`.pt` 是 Python 标准库 `pickle`，不依赖 PyTorch。读取示例见
`results/README.md`。测试 Yield 不会写入模型调试日志；只有正式选中后返回的
`observed_yield` 会进入对应 seed 的轨迹。

重新运行时，`code/main/run_selected.py` 会在指定结果目录创建运行元数据、
逐轮轨迹和指标文件，不覆盖提交包内的参考结果。若还需保存控制台输出：

```bash
python code/main/run_selected.py \
  --data-root data --results-root reproduced_results \
  --dataset all --rounds 40 --seeds 20 \
  > reproduction_console.log 2>&1
```

控制台日志只是辅助诊断；复现结论应以结构化轨迹、校验器输出和哈希清单为准。随包的
`reproduction_verification.log` 汇总最终独立数值复核，以及 2026-08-14 使用包内
Open Science 与赛事 DeepSeek API 完成的端到端智能体验收，便于评委在重新运行前快速
核对预期输出。日志不包含 API Key。
