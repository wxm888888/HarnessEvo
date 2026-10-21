# RL harness 进化流水线

本文描述当前训练接入。旧的独立 pipeline 已删除；训练触发的一轮进化只有一个入口：

```bash
python -m harness_evolution.rl --config /path/to/rl_evolve.yaml
```

`trainer/src/trainer/evolve.py` 只负责在整步边界暂停训练、执行上面的命令并检查退出码。证据挑选、候选生成、门控、发布和版本状态更新都在 `harness_evolution.rl` 内完成。

## 1. 一轮发生什么

```text
训练完成 step N
  -> 从最近 rollout dump 挑选失败/成功轨迹
  -> 拼接 agent-artifacts 并做 preflight
  -> 诊断证据、生成候选 commit
  -> 静态 gate（typecheck/build/smoke）
  -> dynamic 模式下做 parent/candidate OpenSandbox A/B
  -> ACCEPT：持久化 candidate commit、发布二进制、记录 memory、promote version_state
     REJECT：记录拒绝，继续使用 parent
  -> step N+1 的 sandbox 启动时重读 version_state
```

关键边界：

- `HARNESS_VERSION_STATE` 是运行中 harness commit 的唯一真相。新血统由 `HARNESS_BASELINE_COMMIT` 初始化；下一步 sandbox 从 `{OPENCODE_AFS_ROOT}/{commit7}/opencode` 安装。
- `select_evidence` 同时读取 `rollout_XXXXXX/*.json` 和同根目录下的 `agent-artifacts/<session_id>/`。缺少完整消息、工具 schema 或有效 verifier 结果的轨迹不会进入进化器。
- memory 完全由 evolve 配置管理。只有最终晋升的候选才追加 `accepted_lineage`；训练器不读写 memory。
- `gate_mode: static_only` 只适合接线 smoke。生产验收使用 `dynamic`，按同任务 parent/candidate 的 net wins 判定。
- ACCEPT 的候选必须先 fetch/tag 到持久 baseline repo，再发布到 AFS，最后写入 version state。持久化或发布失败会写入 gate report 并终止本轮，不能返回 ACCEPT。
- 普通 async/lookahead 在触发进化的 step 暂停 N+1 预取，等待进化完成后再采样；其他 step 保留预取。当前不支持与 async carryover、nbank disk、rolling weights 同时启用进化，启动时会明确报错。

## 2. 训练接线

启动脚本通常自动生成命令；手工接线的最小形态如下：

```bash
export HARNESS_EVOLVE_INTERVAL=40
export HARNESS_EVOLVE_CMD="python -m harness_evolution.rl --config $HARNESS_EVOLVE_CONFIG"
export HARNESS_VERSION_STATE="$RUN_ROOT/harness_version_state.json"
export HARNESS_BASELINE_COMMIT=5c23e88419c4743b9be42cea132f2fb1e6cb63ff
export POLAR_TRAJECTORY_DIR="$RUN_ROOT/trajectories"
export OPENCODE_AFS_ROOT=/sandbox-data/duchenyu/opencode
```

基线 commit 必须同时存在于 OpenCode git 仓库和 AFS；首次 rollout 前应确认：

```bash
stat /sandbox-data/duchenyu/opencode/5c23e88/opencode
```

训练器相关开关：

| 变量 | 作用 |
| --- | --- |
| `HARNESS_EVOLVE_INTERVAL` | 每 N 个训练 step 触发；`0` 关闭 |
| `HARNESS_EVOLVE_CMD` | 一轮进化的 shell 命令 |
| `HARNESS_EVOLVE_STRICT` | `1` 时流水线错误终止 smoke/训练；默认保留当前 harness 继续 |
| `HARNESS_EVOLVE_ON_START` | `1` 时在恢复训练进入循环前补跑一轮 |
| `HARNESS_EVOLVE_ON_START_TRAJ_ROOT` | ON_START 使用的旧 trajectory 根目录 |

`HARNESS_EVOLVE_ROLLOUT_ID` 由训练器在调用命令时注入，不需要用户设置。

DAPO/lookahead launcher 默认关闭进化，但保留调用方显式设置的开关。`HARNESS_EVOLVE_INTERVAL=0` 与 `HARNESS_EVOLVE_ON_START=1` 可以组合为仅恢复时运行一次；launcher 仍会准备版本状态、采集、Bun、配置和 Python 路径。此时应设置 `HARNESS_EVOLVE_ON_START_TRAJ_ROOT` 指向旧 run 的证据目录。

`POLAR_TASK_FORMAT=auto` 默认根据输入 JSONL 首条非空记录的 MiMo metadata 选择训练模板，也可显式设为 `mimo` 或 `terminal_bench`。该变量会传入 rjob。

## 3. rl_evolve.yaml 与九个覆盖变量

`rl_evolve.yaml` 决定证据配额、evolver、gate 和输出位置。下面九个环境变量可以按一次训练覆盖共享 YAML；唯一真相是 `harness_evolution/rl/__main__.py::_ENV_OVERRIDES`：

| 环境变量 | YAML 字段 | 含义 |
| --- | --- | --- |
| `HARNESS_EVOLVE_EVOLVE_CONFIG` | `evolve_config` | 候选生成配置 |
| `HARNESS_EVOLVE_GATE_CONFIG` | `gate_config` | 静态/动态门控配置 |
| `HARNESS_EVOLVE_GATE_MODE` | `gate_mode` | `dynamic` 或 `static_only` |
| `HARNESS_EVOLVE_EVOLVER` | `evolver` | evolver preset，例如 `sglang` 或 `codex` |
| `HARNESS_EVOLVE_WORK_ROOT` | `work_root` | 每轮工作目录根路径 |
| `HARNESS_EVOLVE_STAGE_DIR` | `stage_dir` | 版本/二进制 staging 目录 |
| `HARNESS_EVOLVE_WINDOW` | `window` | 向后扫描的 rollout 目录数 |
| `HARNESS_EVOLVE_N_FAIL` | `n_fail` | 每轮最多选取的失败任务数 |
| `HARNESS_EVOLVE_N_SUCCESS` | `n_success` | 每轮最多选取的成功任务数 |

命令行的 `--n-fail`、`--n-success`、`--window`、`--evolver` 和 `--gate-mode` 优先于环境变量；环境变量优先于 YAML。

## 4. 证据与任务身份

每条可用证据至少包含：

```text
instruction.md
result.json
agent/trajectory.json
agent/openai-messages.jsonl
agent/openai-tools.json
agent/verifier-summary.json
agent/artifact-status.json   # complete=true
```

MiMo 任务以 `dataset_task_id` 去重，并把 `task_file`、SHA-256、dataset type 和 image tag 带到 conversion manifest。dynamic gate 据此重建同一个任务，而不是退回固定 canary。

源数据集字段在提交 Polar 时进入 task metadata 的 `dataset_*` 命名空间，避免与 runtime `task_id` 混淆。`SessionResult -> Sample` 后，dump 优先读取 `metadata.polar.trajectory_metadata.task_metadata` 中的这些字段，兼容旧嵌套原名及顶层字段。不要用手工合并 Sample metadata 或修改 manifest 掩盖接缝问题。

正常 verifier 失败（reward 0）是有效负样本；runtime、agent timeout、verifier timeout、无效 verifier 和不完整 trace 属于基础设施/采集失败，不得伪装成负样本。

## 5. 门控与发布

静态 gate 顺序是 no-op 检查、typecheck、build、binary diff 和 smoke。dynamic gate 对 parent 和所有静态 survivor 使用相同任务、模型、timeout 和 verifier，并逐任务计算 win/loss/tie：

```text
net = wins - losses
ACCEPT iff net >= accept_net_wins
```

当前门控复用原进化模块的 infra error / imputed-zero 评分规则，按 2026-10-17 的决定保持不变；证据 preflight 对 infra/采集失败的过滤是另一处边界。

`gate_1179.yaml` 显式设置 `heldout_gate.backend: opensandbox`，避免残留 Harbor 模板影响后端选择。OpenSandbox gate 会在运行前发布两侧二进制。只有在确认 commit 对应二进制已经存在于默认 AFS 路径时，调试重跑才可设置 `HARNESS_AFS_PUBLISH=0`。

## 6. 输出与退出语义

每轮写到 `work_root/stepXXXXXX_MMDD-HHMMSS/`，主要产物包括：

```text
tb_job/                 # 被选证据和 conversion_manifest.json
workspaces/             # 候选 worktree
gate-reports/           # static/dynamic gate 报告
round.json              # 本轮最终结果
```

最后一行 stdout 总是 JSON。退出码 `0` 包括 ACCEPT、REJECT 和没有候选值得门控；非零表示流水线自身失败。默认训练会保留当前 harness 继续，`HARNESS_EVOLVE_STRICT=1` 会把错误向上抛出。

## 7. 环境注意事项

- Polar OpenSandbox runtime 需要 Python 3.11+。
- Codex preset 使用当前进程的 `CODEX_HOME` 和代理设置；通过 rjob/Ray 时这些显式变量会保留。ChatGPT 凭证目录必须在训练容器内可见，参见 [Codex 验证记录](../../docs/codex-validation-20261017.md)。
- 从登录节点做 smoke 必须设置 `POLAR_SANDBOX_RUN_ONLY=1`、绝对路径 `RUN_ROOT` 和 `POLAR_SANDBOX_CAPACITY_ROOT=$RUN_ROOT/capacity`。
- `HARNESSRL_TB21_RUNTIME` 不仅要是环境变量，还必须加入 `PYTHONPATH`，否则 trace 归一化会静默降级。
- `AGENT_GATEWAY_URL` 必须从 OpenSandbox 内可达；登录节点内部的 Kubernetes `.svc` 地址不一定可达。应使用训练配置中的 LLM-Proxy/evalservice 地址。
- MiMo 任务至少需要 8 GiB sandbox memory，并以任务 JSON 的绝对 `cwd` 同时设置 runtime workdir 和 `TASK_WORKSPACE`。

建议 smoke 使用 `HARNESS_EVOLVE_INTERVAL=1`、少量证据和 `HARNESS_EVOLVE_STRICT=1`。先用 `static_only` 验接线，再用 `dynamic` 验证真实 parent/candidate 对照。

2026-10-17 已完成 ON_START 真机链路验收（真实 Codex、动态 ACCEPT、隔离晋升、下一只沙箱安装），范围与收据见 [ON_START 验收记录](../../docs/on-start-validation-20261017.md)。GPU checkpoint 恢复及后续训练尚未验收。

独立进化侧的证据导出在完整校验成功后默认清理中间 context 和 raw wire，并支持后续 materialize 复用保留的 OpenAI exports；导出失败则保留原始文件。可用 HARNESSRL_KEEP_LLM_CONTEXT=1、HARNESSRL_KEEP_RAW_TRACE_WIRE=1 保留调试文件。构建环境、原生依赖等待及证据保留的合入与回归见 [2026-10-18 合入记录](../../docs/independent-fixes-merge-20261018.md)。
