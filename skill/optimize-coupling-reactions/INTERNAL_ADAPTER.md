# CatalystFlow 内部数值与结构适配器

本目录不是独立的 Open Science Skill，也不是评委或用户入口。正式入口只有
`../catalystflow/SKILL.md`；CatalystFlow 工作流通过受限 MCP 在内部调用本目录脚本。

## 职责

- `preflight_reaction_data.py`：检查结果泄漏、字段、重复条件和结构覆盖率；
- `reaction_agent.py`：为新数据维护单点 `recommend → feedback` 状态；
- `resolve_structures.py`：解析并校验结构，输出待人工审核或已审核状态；
- `select_fingerprint_config.py`：依据标注规模和内存上限选择结构表示容量；
- `validate_trajectory.py`：校验实时轨迹；
- `self_test.py`：内部适配器自测。

正式比赛的 2 × 20 × 40 数值复现不调用实时适配器，而由
`../../code/main/run_selected.py` 和固定模型配置执行。内部适配器不读取未查询产率，
不调用 LLM 预测产率，也不负责已查询 Top-5 的现实可行性评估。

详细策略保留在 `references/`，但均由 `catalystflow` Skill 路由，不单独注册。
