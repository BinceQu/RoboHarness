<div align="center">

<img src="docs/assets/roboharness-wordmark.png" alt="RoboHarness" width="500">

### 一个简单而有效的机器人 harness

**[项目主页](https://bincequ.github.io/RoboHarness/) · [English](README.md) · [安装说明](docs/setup.md) · [任务提示词](docs/case-prompts.md)**

[![RoboHarness 总览](docs/assets/overview.png)](docs/assets/overview.pdf)

<sub>RoboHarness 总览：持续跟踪视觉关键点，将其转换为几何约束，并在闭环中执行与验证。[查看原始 PDF](docs/assets/overview.pdf)</sub>

</div>

## 项目介绍

**贡献者：** [BinceQu](https://github.com/BinceQu) 和 **Codex**（AI 编程助手）。

RoboHarness 无需训练模型，将视觉点选、关键点跟踪和几何约束相结合，为 LLM agent
提供具身控制面板。

LLM 在二维图像中标记感兴趣的点，并通过光流跟踪与深度反投影持续获取这些点的位置；
借助这些信息，它能够组合准确的基本动作，完成复杂操作。

BEHAVIOR 实验使用 Claude Code 2.1.259、`Qwen3.8-Flash-Next-FP8` 和 R1 Pro 机器人。
仓库同时提供 Codex harness。

## 仓库结构

```text
BEHAVIOR/           BEHAVIOR 仿真器与评测器，固定为 v3.9.1
interface/          RGB-D 观测、机器人控制和交互界面
harness/
  claude_code/      Claude Code harness
  codex/            Codex harness
prompt/             各项家居操作任务的专用提示词
tasks/              任务定义、评测实例和步数预算
reference_results/  参考评测成绩
roboharness/        任务运行与评测流程
scripts/            安装、一键运行、验证和成绩报告
validation_results/ 评测报告与成绩
```

## 安装

需要 Linux x86-64、NVIDIA RTX GPU、Python 3.11 和 CUDA 12.4 工具链。
验证机器每卡显存为 48 GB；该值是测试环境配置，不是经过验证的最低要求。

```bash
git clone --recurse-submodules https://github.com/BinceQu/RoboHarness.git
cd RoboHarness
./scripts/setup.sh
cp configs/example.json configs/local.json
```

按[安装说明](docs/setup.md)准备 BEHAVIOR 数据与 Claude Code 2.1.259，
并在 `configs/local.json` 中填写本机的数据目录和 Python 路径。

## 运行任务

模型服务和评测运行器是两个独立进程。评测期间需要保持自己的模型服务运行。
`setup.sh` 安装仿真与 agent 环境，不会下载大模型权重或启动推理服务。

**1. 连接自己的模型服务。**

默认的 Claude Code harness 使用 **Anthropic 兼容的 `/v1/messages` 接口**，
服务须支持图片输入、工具调用和流式输出。论文配置使用 `Qwen3.8-Flash-Next-FP8`。
如果服务只支持 OpenAI Chat Completions，先按[桥接说明](docs/setup.md#chat-completions-services)
配置适配器；Codex 使用另一套 [Responses 配置](docs/setup.md#codex-model-service)。

在**评测机器**上进入本仓库目录，打开终端，将以下地址、模型名和密钥替换为自己的服务信息：

```bash
export ROBOHARNESS_MODEL_URL='http://YOUR_MODEL_HOST:31000'
export ROBOHARNESS_MODEL='Qwen3.8-Flash-Next-FP8'
export ANTHROPIC_API_KEY='YOUR_MODEL_SERVER_KEY'
export ANTHROPIC_AUTH_TOKEN="$ANTHROPIC_API_KEY"
```

地址填写服务 origin，**不带 `/v1`**；模型名必须与服务实际提供的 ID 一致。
服务未启用认证时，密钥填写 `local-no-auth`。
只有模型服务与评测在同一台机器，或已通过 SSH 转发到评测机器时，才使用 `127.0.0.1`。

先发送一个小请求，确认模型服务可用：

```bash
python3 - <<'PY' | curl --fail-with-body --silent --show-error \
  "${ROBOHARNESS_MODEL_URL%/}/v1/messages" \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" \
  --data-binary @-
import json, os
print(json.dumps({
    "model": os.environ["ROBOHARNESS_MODEL"], "max_tokens": 128,
    "messages": [{"role": "user", "content": "Reply with OK."}]
}))
PY
```

预期返回包含 `"type": "message"` 的 Messages 响应。这一步检查连接、认证和模型名；
服务还须支持上面的图片与工具功能。连接失败、401/403、404 等问题见
[模型连接排查](docs/setup.md#model-connection-troubleshooting)。

**2. 保存本次评测的配置。**

在同一终端执行以下命令。首次从安装配置创建 session 文件，后续保留已有设置，
只更新模型地址和模型名：

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
path = Path(os.environ.get("ROBOHARNESS_SESSION_CONFIG", ".local/session-config.json"))
source = path if path.exists() else Path("configs/local.json")
config = json.loads(source.read_text())
config.update(model_url=os.environ["ROBOHARNESS_MODEL_URL"],
              model=os.environ["ROBOHARNESS_MODEL"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, indent=2) + "\n")
print("Saved", path)
PY
```

默认 session 文件为 `.local/session-config.json`，也可通过 `ROBOHARNESS_SESSION_CONFIG` 指定其他文件。
检查其中的配置：`data_path` 指向 [BEHAVIOR 数据根目录](docs/setup.md)，
`interface_python`、`evaluator_python`、`agent_python` 指向已安装的环境。
使用 `setup.sh` 安装时可沿用示例中的 Python 路径。需要时将 `cache_dir` 设为 Git 仓库之外、
空间充足且可写的临时目录。相对路径均相对于仓库根目录。
密钥只保存在 shell 环境中，新开终端后需重新 export；后续配置修改应写到这个 session 文件。

**3. 核对计划，再启动评测。**

```bash
claude --version  # 论文配置要求 2.1.259
./run.sh --list
./scripts/reproduce_task.sh task01 --gpu 0 --dry-run
./scripts/reproduce_task.sh task01 --gpu 0 --run-dir runs/task01-001
```

核对 dry-run 输出中的 `model_url`、`model`、`gpu`、三个端口和五个实例 ID。
dry-run 只解析配置，不请求模型或检查仿真安装。最后一条命令启动接口、官方评测器与
Claude Code agent，按顺序运行全部五个实例。
`--gpu 0` 选择的是**仿真 GPU**；模型服务使用哪些 GPU，由启动自己的模型服务时决定。
需要为仿真与模型服务分别预留足够资源。

换任务时将 `task01` 替换为列表中的任务；task04 未包含在内。
每次运行使用新的输出目录，也可省略 `--run-dir` 让程序自动生成。
先试跑单个实例可添加 `--instances 301` 并使用单独目录；完整均分需要全部五个实例。

**4. 查看轨迹与成绩。**

默认配置下，task01 的网页接口是评测机器上的 `http://127.0.0.1:15071`。
若评测在远程机器上，在**自己的电脑**执行以下命令，再用浏览器打开该地址：

```bash
ssh -N -L 15071:127.0.0.1:15071 YOUR_USER@YOUR_EVALUATION_HOST
```

启动日志会显示 HTTP、policy、idle gate 三个端口，需要时通过
[`task_ports`](docs/setup.md#session-configuration-and-ports) 修改。
上面的启动命令将结果保存到：

| 路径 | 内容 |
| --- | --- |
| `runs/task01-001/summary.json` | 已完成实例数、分数与当前 mean Q-score |
| `runs/task01-001/output/json/` | 各实例的官方评分 |
| `runs/task01-001/instance_301/` | 实例 301 的 prompt、agent 输出、错误日志和轨迹 |
| `runs/task01-001/run.log` 和 `runs/task01-001/logs/` | 运行器、接口与评测器日志 |

全部五个实例完成后，生成对比报告：

```bash
python3 scripts/report_validation.py runs/task01-001 \
  --output .local/reports/task01-001 --check-live --require-match
```

打开 `.local/reports/task01-001/README.md` 查看结果。
任务未完成或评测结果不匹配时，`--require-match` 返回非零退出码。
`runs/` 和 `.local/` 均已被 Git 忽略。

## 复现与评分规则

论文中的 BEHAVIOR 评测设置如下：

- **每个任务使用专用 prompt，同一任务的所有实例共享该 prompt。** prompt 由人工编写，
  包含完成任务的执行步骤和模型必须遵守的行为边界，见[任务提示词](docs/case-prompts.md)。
- **每个任务评测五个实例。** 使用采样种子 `20260911`，所有任务均选取 slot
  **0、3、5、7、9**，对应实例 **301、304、306、308、310**。每个实例进行一次正式 rollout。
- **Challenge 2025，两倍步数预算。** 每个任务的最大步数为该任务人类演示平均长度的两倍，
  按仿真控制步计数。达到任务目标或步数上限时结束，见[步数预算](docs/provenance.md#evaluation-budgets)。
- **Mean Q-score。** 使用 BEHAVIOR Challenge 2025 官方评测器的 `q_score.final`，
  对全部五个实例的分数取不加权平均，报告值保留四位小数。

运行器默认使用 `session_timeout_s: 0`，不增加墙钟时限。实际运行耗时取决于硬件、模型服务和
并发负载；评测预算按仿真控制步数计算。

报告逐任务比较完整的五例均分与参考值，容限为 `1e-6`，单例分数允许不同。
`--require-match` 在任务未完成、均分不匹配或未通过评测检查时返回非零退出码。
在评测主机上使用 `--check-live`；分析复制来的运行目录时省略它。添加 `--watch` 可持续更新报告。

## 开发与许可

安装依赖后执行 `./scripts/check.sh` 运行 CPU 检查。
提交问题或修改前请阅读[贡献说明](CONTRIBUTING.md)。

仓库代码采用 [MIT 许可证](LICENSE)，第三方组件见[许可说明](THIRD_PARTY_NOTICES.md)。
BEHAVIOR 数据、NVIDIA Isaac Sim 和模型权重遵循各自的许可与访问条件。
