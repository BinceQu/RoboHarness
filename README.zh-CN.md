<div align="center">

<img src="docs/assets/roboharness-wordmark.png" alt="RoboHarness" width="500">

### 一个简单而有效的机器人 harness

**[项目主页](https://bincequ.github.io/RoboHarness/) · [English](README.md) · [安装说明](docs/setup.md) · [逐例提示词](docs/case-prompts.md)**


[![RoboHarness 总览](docs/assets/overview.png)](docs/assets/overview.pdf)

<sub>RoboHarness 总览：持续跟踪视觉关键点，将其转换为几何约束，并在闭环中执行与验证。[查看原始 PDF](docs/assets/overview.pdf)</sub>

</div>

## 项目介绍

**贡献者：** [BinceQu](https://github.com/BinceQu) 和 **Codex**（AI 编程助手）。

RoboHarness 无需训练模型，将视觉点选、关键点跟踪和几何约束相结合，为 LLM agent
提供具身控制面板。

LLM 在二维图像中标记感兴趣的点，并通过光流跟踪与深度反投影持续获取这些点的位置；
借助这些信息，它能够组合准确的基本动作，完成复杂操作。

仓库包含 **9 个任务、45 个归档 case**。参考实验使用 Claude Code 2.1.259 和
`Qwen3.8-Flash-Next-FP8`。Codex 是可选 harness，归档中没有对应的 Codex 成绩。

## 仓库结构

```text
BEHAVIOR/           上游子模块，固定为 v3.9.1
interface/          evaltest 接口、RGB-D 工具、机器人配置和 idle gate
harness/
  claude_code/      参考实验所用的 Claude Code harness
  codex/            可选 Codex harness
prompt/             仅保留归档 case 实际使用的提示词
tasks/              case、实例 ID、预算、提示词哈希与来源
reference_results/  原始官方评分与初始观测记录
roboharness/        启动、进程管理和归档约束检查
scripts/            安装、一键运行、验证和成绩报告
validation_results/ 新测试评分与验证证据
```

完整历史轨迹、BEHAVIOR 数据集、模型权重和密钥不随仓库分发。归档来源和哈希保留在清单中，
新测试的完整轨迹保存于本机 `runs/`。

## 安装与一键运行

需要 Linux x86-64、NVIDIA RTX GPU、Python 3.11 和 CUDA 12.4 工具链。
验证机器每卡显存为 48 GB；该值是测试环境配置，不是经过验证的最低要求。

```bash
git clone --recurse-submodules https://github.com/BinceQu/RoboHarness.git
cd RoboHarness
./scripts/setup.sh
cp configs/example.json configs/local.json
```

按[安装说明](docs/setup.md)准备 BEHAVIOR 数据、Claude Code 2.1.259 和模型服务，
并在 `configs/local.json` 中填写本机的数据目录、Python 路径和服务地址。
模型服务使用环境变量提供认证信息。环境准备好后：

```bash
./run.sh --list
./scripts/reproduce_task.sh task01 --gpu 0

# 查看计划，不启动仿真。
./scripts/reproduce_task.sh task03 --gpu 0 --dry-run

# 仅测试指定实例，用于排查。
./scripts/reproduce_task.sh task08 --gpu 0 --instances 301,304
```

默认运行该任务全部五个实例：**301、304、306、308、310**，对应归档 slot
**0、3、5、7、9**。task04 没有归档数据，因此不提供该任务的复现入口。

包装脚本首次运行时将 `configs/local.json` 复制到 `.local/session-config.json`，
后续复用该 session 文件。之后调整配置请修改该文件；不会改动全局 Claude 或 Codex 配置。
通过 `task_ports` 可为每个任务指定 HTTP、policy、idle gate 三个端口，
参见[全部使用 1507* 的示例](docs/setup.md#session-configuration-and-ports)。

运行器依次加载归档场景和提示词，保存计划、会话、模型轨迹、官方评分 JSON 与汇总到新的
`runs/<run-id>/` 目录。网页接口位于 `http://127.0.0.1:<port>/`；远程访问可使用 SSH 转发。

## 复现与评分规则

- **每个任务使用专用 prompt，同一任务的所有实例共享该 prompt。** 按论文描述，
  prompt 由人工编写，包含完成任务的执行步骤和模型必须遵守的行为边界。
  每个任务的五个实例（301、304、306、308、310）均使用该任务的同一份 prompt。
- 使用 **Challenge 2025 ×2**，严格采用归档中的最终整数步数预算。
- 默认 `session_timeout_s: 0`，不增加墙钟时限，可运行超过 72 小时；仿真步数限制仍有效。
- 发布版保留所选轨迹中恢复出的 prompt 版本，归档中同一任务的版本差异见
  [逐例提示词记录](docs/case-prompts.md)。运行时按归档映射加载并记录来源与 SHA-256，
  同时检查已恢复的 Skill 上下文和 MCP 命名空间。
- 每个任务的全部五例完成后，比较 **mean Q-score**，绝对误差容限为 `1e-6`。
  单例分数允许不同，各任务分别验收。
- 以轨迹目录报告的均分为目标。原始 JSON 与目录汇总冲突时保留两者及说明，不改写历史成绩。
- 被墙钟时限截断的结果保留作排查证据，不作为有效的完整复现结果。

生成报告：

```bash
python3 scripts/report_validation.py runs/YOUR_RUN_A runs/YOUR_RUN_B \
  --output validation_results/latest --check-live --require-match
```

`--check-live` 用于实际运行评测的 Linux 主机；分析复制来的运行目录时省略它。
`--require-match` 对未完成、失败、均分不匹配或归档约束未满足的运行返回非零退出码。

## 开发与许可

安装依赖后执行 `./scripts/check.sh` 运行 CPU 检查。
提交问题或修改前请阅读[贡献说明](CONTRIBUTING.md)。

仓库代码采用 [MIT 许可证](LICENSE)，第三方组件见[许可说明](THIRD_PARTY_NOTICES.md)。
BEHAVIOR 数据、NVIDIA Isaac Sim 和模型权重遵循各自的许可与访问条件。
