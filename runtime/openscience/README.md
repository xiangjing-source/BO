# Open Science 离线运行时

本目录保存 Open Science 2.0.23 的官方 Linux x86_64 npm 发布归档，用于复现
CatalystFlow 智能体。归档、归档内原生二进制及许可证均由
`runtime/openscience/runtime.json` 固定并校验。

Open Science 是第三方 Apache-2.0 开源软件；官方源代码仓库为
`https://github.com/synthetic-sciences/openscience`。本项目没有修改其二进制。
CatalystFlow 自有的智能体定义、Skill、MCP 服务和模型代码分别位于：

- `.openscience/agent/catalystflow.md`
- `skill/catalystflow/`
- `agent/`
- `code/`

评委无需安装 Open Science、Node.js 或 npm。执行
`bash runtime/run_openscience.sh` 时，启动器从本目录校验并解压官方归档到
`runtime/.openscience_runtime/`，在提交根目录建立独立的本地 Git 项目边界，然后启动
项目工作区。该边界防止会话误用上级 Git 项目及其模型配置；它与运行缓存均不属于提交
清单。请使用本次启动命令输出的 URL，不要复用旧项目的会话 URL。

离线归档只解决智能体引擎复现。DeepSeek 推理仍需赛事 API 网络、有效 API Key 和
可用额度；API Key 不包含在提交包内，也不会写入项目文件。

支持边界：Linux x86_64、Linux 内核 5.1 或更高版本。其他平台可根据
`runtime.json` 中的版本号从官方发布源安装对应平台的 Open Science，但本提交不把
未经实机验证的平台二进制声明为离线可复现。
