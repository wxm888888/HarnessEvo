# 进化模块介绍与循环启动指南

> 仓库：`/mnt/shared-storage-user/yuanjiarui-p/wxh/harnessRL/harness-evolution-new`
> 本文讲**进化循环每一步具体做什么、按什么规则判定**，以及**怎样启动一次完整循环**。
> 目录结构、配置字段、prompt 修改、扩展方式见 [guide.zh.md](guide.zh.md)。

## 1. 概览

进化模块的目标：让一个强模型（目前是 Codex gpt-5.6-sol）读 agent 在真实任务上的运行轨迹，找出 agent harness（目前是 OpenCode）的问题，直接修改 harness 源码；再用一个固定的评测集判断改动是否真的有效，只有有效的改动才保留下来。

一次循环由若干 step 组成，每个 step：

```
parent harness（OpenCode 的某个 commit + 编译好的二进制）
 │
 ├─ ① 采样        用 parent 在 OpenSandbox 跑 32 道新的 TMAX 题（held-in）
 ├─ ② 轨迹标准化  每道题整理成一组固定的轨迹文件
 ├─ ③ 诊断        Codex 逐条分析轨迹，再跨轨迹汇总成改动方向
 ├─ ④ 生成候选    Codex 在独立的 git worktree 里改 OpenCode 源码（默认 2 个候选）
 ├─ ⑤ 静态门控    no-op / typecheck / build / 二进制差异 / smoke
 ├─ ⑥ held-out 门控  89 道 TB2.1 题，候选 vs parent
 └─ ⑦ 晋级        有候选严格超过 parent → 成为新 parent；否则 parent 不变
```

两个数据集严格分开：

| | held-in（TMAX） | held-out（TB2.1） |
|---|---|---|
| 用途 | 产生轨迹，供诊断和改代码 | 只用于评测和晋级 |
| 规模 | 每步 32 道，从 4487 道题的 catalog 里抽 | 固定 89 道 |
| 是否每步变化 | 每步抽一批新的、互不重叠的题，无论是否晋级 | 不变 |
| 轨迹是否给 evolver 看 | 是 | **从不** |

晋级只看 held-out，不看 held-in：因为 held-in 每步都换题，不同 step 之间的 held-in 分数没有可比性。

## 2. 各阶段详解

### ① 采样

- 代码：`evaluation/sampling.py`，后端 `evaluation/backends/opensandbox.py`。
- 用 parent 的 OpenCode 二进制在 OpenSandbox 里跑本步的 32 道 held-in 题，每道题 `sampling_repeats` 次（通常 1 次）。
- 本步抽到哪些题记录在 `run_dir/held-in-schedule.json`。
- 晋级后，**下一步的采样用的就是刚晋级的新 parent**，所以下一轮诊断看到的总是当前最好版本的轨迹。

### ② 轨迹标准化

- 代码：`traces/materialize.py`、`traces/verifier_feedback.py`、`traces/sanitize.py`。
- 每道题整理成一个目录，包含：

| 文件 | 内容 |
|---|---|
| `instruction.md` | 题面 |
| `llm-context.jsonl` | 原始数据：OpenCode 每次调用模型的完整请求和响应（每行带完整上下文，体积大） |
| `openai-messages.jsonl` | 去重后的一条完整对话，即模型实际看到和说出的内容（含 reasoning） |
| `trajectory.json` | OpenCode 自己的会话事件流：步骤、工具调用状态、时间戳 |
| `openai-tools.json` / `openai-request.meta.json` / `openai-messages.meta.json` | 工具定义、每次请求的参数、消息和请求的对应关系 |
| `verifier-summary.json` | reward、状态、失败类别、失败阶段、是否超时 |
| `verifier-feedback.json` | pytest 每项的通过情况和 stdout/stderr 摘录 |
| `sanitized_result.json` | 脱敏后的评测结果（reward、异常、token 用量、超时设置） |

- `trajectory.json`、`openai-messages.jsonl`、`openai-tools.json`、`verifier-summary.json` 四个都存在才算完整轨迹；不完整的、以及基础设施错误的（`traces/preflight.py` 判断）不参与诊断。

### ③ 诊断

当前使用的证据策略是 `per_trace_then_aggregate`（`diagnosis/per_trace_then_aggregate.py`），分两层：

**逐条诊断。** 每条轨迹单独起一个 Codex，工作目录里是这条轨迹全部文件的软链接，prompt 是 `diagnosis/prompts/trace_diagnosis.md`。成功和失败的轨迹都会分析（`analyze_passed` / `analyze_failed`），并发数由 `diagnosis_concurrency` 控制。输出 `per_trace/<trace>.json`，主要字段：

| 字段 | 含义 |
|---|---|
| `terminal_cause` | 最终成功或失败的直接原因 |
| `first_divergence_step` | 第一个走错的步骤 |
| `agent_mechanism` | agent 的行为机制 |
| `harness_relevance` | `level`（none/low/medium/high）+ `reason`：这次结果和 harness 有多大关系 |
| `success_patterns` / `failure_patterns` | 可复用的成功做法 / 失败模式 |
| `causal_chain` / `evidence_refs` | 因果链和证据引用（指向轨迹里的具体事件） |

**跨轨迹汇总。** 再起一个 Codex 读本步所有逐条诊断（prompt：`diagnosis/prompts/aggregate.md`），输出 `aggregate.json` / `aggregate.md`，主要包括：

- `shared_patterns`：多条轨迹共有的模式；
- `harness_change_hypotheses`：建议的 harness 改动方向（含目标机制、建议改法、支撑轨迹）；
- `task_specific_patterns`：只和个别题目有关的模式（不建议据此改 harness）。

诊断结果按轨迹内容缓存在 `run_dir/diagnosis-cache/<hash>/`，续跑时不会重复诊断。

### ④ 生成候选

- 代码：`proposer/batch.py`、`proposer/requirement.py`。
- 每个候选从 parent commit 新建一个 git worktree，`node_modules` 从 parent 的 harness 目录链接过来（`harnesses/opencode/build.py` 的 `link_shared_node_modules`）。
- Codex 收到的主指令由 `proposer/prompts/candidate.md` 拼装，依次包括：
  1. 角色和可修改范围（只能改当前仓库，不能写题目答案）；
  2. 必读 skill：`harnesses/opencode/skills/opencode-architecture-evolution`（OpenCode 源码结构和改动规范）；
  3. 证据说明：诊断结果在哪、怎么用；
  4. 可选的晋级历史（memory，见 ⑦）；
  5. 改动范围约束：**必须改变运行时行为**，只改 AGENTS.md、文档、`.opencode/skills/`、注释的候选无效；只有证据直接指向 prompt 问题时才允许只改 prompt，且要配回归测试；
  6. 验证要求：补针对性测试、跑相关测试和 `git diff --check`；
  7. 最后写一份 `.harness_evolver/candidate-report.json`（选了哪些假设、根因、改了什么行为、风险等），这份报告不提交。
- 改完后提交成一个 commit。候选数由 `loop.candidates` 控制。

### ⑤ 静态门控

代码：`evaluation/static_gate.py`。按顺序检查，任何一项失败，候选就不进 held-out：

| 检查 | 默认 | 说明 |
|---|---|---|
| `no_op` | 开 | 候选没有改任何文件 → 失败 |
| `diff_limit` | 关 | 开启后限制改动规模（默认最多 20 个文件、800 行） |
| `typecheck` | 开 | 对 `opencode...` 相关包跑 typecheck |
| `build` | 开 | 用 Bun 编译出 OpenCode 二进制 |
| `binary_diff` | 开 | 编出来的二进制必须和 parent 不同 |
| `smoke` | 开 | 二进制能正常启动（默认 60 秒超时） |

### ⑥ held-out 门控

- 代码：`loop/two_sided.py` 的 `_run_two_sided_gate`，计分 `evaluation/scoring.py`。
- 所有通过静态门控的候选在 89 道 TB2.1 题上各跑 `gate_repeats` 次；parent 也需要同样的成绩。
- **parent 的成绩按 commit 缓存**（`run_dir/baseline-cache/`），同一个 parent 只评测一次。候选晋级后，直接用它这次的成绩作为新 parent 的成绩，不重跑。
- 单道题遇到沙箱或基础设施错误会重试（`SANDBOX_TRIAL_RETRIES`），仍失败的按 0 分计入（`raw_errors` / `imputed_zero_attempts`）。
- **晋级条件**：候选的 89 题跑完整，且通过率**严格大于** parent。
- 多个候选都满足时，按（held-out 通过率、通过次数、错误数少者优先）排序取第一。

### ⑦ 晋级与记忆

- 结果写进 `step-XX/gate.json` 和 `step-XX/step.json`，循环状态写进 `run_dir/state.json`（当前 parent commit、下一步编号、历史）。**判断进度以 `state.json` 为准**。
- 晋级的 parent 二进制保存在 `run_dir/parent-binaries/`。
- 可选记忆 `memory: accepted_lineage`：每次晋级追加一条记录（step、parent/winner commit、改动摘要、改动文件、前后 held-out 分数），之后生成候选时以只读文件的形式放进 worktree，告诉 Codex 以前哪些改动被接受过。只记录晋级的改动，不记录失败的。默认 `none`。

## 3. 启动一次完整循环

下面以集群上直接运行为例。rjob 提交、Codex 登录检查、网络预检这些集群步骤仍由各自的提交脚本负责，脚本最后调用 `hevo run` 即可。

### 3.1 启动前检查

| 检查项 | 怎么确认 |
|---|---|
| Codex 凭证有效 | `CODEX_HOME` 指向有效的登录目录，例如 `.codex-chatgpt/<机器名>`。`.codex-rjob/codex56sol-v8-20261008` 已失效（401） |
| Codex 能出网 | 设置代理 `HTTP_PROXY=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128` |
| 模型有容量 | 用一句话的 prompt 先试一下：`codex exec -m gpt-5.6-sol --skip-git-repo-check "Reply with exactly: ok" < /dev/null`。返回 “Selected model is at capacity” 说明暂时不可用 |
| Bun 可用 | `/mnt/shared-storage-user/yuanjiarui-p/wxh/harnessRL/.bun/bin/bun` 加进 `PATH` |
| 起始 harness 依赖完整 | `initial.harness_path` 下的 `node_modules` 必须可用。如果它本身是指向别的工作区的链接，那个工作区不能被清理（见第 5 节） |
| 被测模型服务可用 | `gate_model.gateway` 上的模型副本（`gate_model.names`）能正常返回 |
| OpenSandbox 可用 | 地址和访问凭证写在 `env_files` 指向的 KEY=VALUE 文件里，参考旧仓库 `scripts/submit_harnessRL_tmax32_*_rjob_20261011.sh` 导出的变量 |

### 3.2 准备 evolve.yaml 和 gate.yaml

从旧仓库的模板复制一份到自己的目录，再改输出路径：

```bash
T=/mnt/shared-storage-user/yuanjiarui-p/wxh/harnessRL/harness-evolution/fresh-templates
mkdir -p ~/my-run && cp $T/codex56sol-diagnosis-v8-20261008/{config.yaml,gate.yaml} ~/my-run/
# 带 memory 的版本：$T/codex56sol-diagnosis-v8-memory-20261010/
```

必须改的字段：

- `config.yaml`：`initial.harness_path` 和 `initial.base_commit`（起始 harness）；`loop.workspace_root`、`loop.report_root`、`evidence.options.result_root`、`evidence.options.analyzer.xdg_root`、`task_cache_root`、`jobs_root`、`build` 下的 xdg 目录等所有输出路径，指向新 run 自己的目录；Codex 路径 `client_init.command_prefix`。
- `gate.yaml`：`paths` 下的 workspace、report、`version_state`，以及 `heldout_gate.jobs_dir` 等输出路径。

> 输出路径一定要换成新目录，否则会覆盖或混入旧实验的结果。

### 3.3 写 run.yaml

参考 `configs/examples/run.yaml`。下面是和 2026-10-11 实验等价的写法（已验证 `hevo run` 生成的参数、环境变量和原启动脚本一致），把 `run_dir` 和配置路径换成自己的：

```yaml
run_dir: /mnt/shared-storage-gpfs2/safewt-share02/sunyoubang/yuanjiarui/wangxianghe/evoloop/<新 run 名>
evolve: /home/<you>/my-run/config.yaml
gate: /home/<you>/my-run/gate.yaml
data:
  held_in_catalog: /mnt/shared-storage-user/yuanjiarui-p/duchenyu/tmax-hamishivi/tmax_hamishivi_all_4487.jsonl
  held_in_size: 32
  held_out: /mnt/shared-storage-user/yuanjiarui-p/wxh/harnessRL/local_runs/tb21_all_89.json
  task_catalogs:
    - /mnt/shared-storage-user/yuanjiarui-p/duchenyu/tmax-hamishivi/tmax_hamishivi_all_4487.jsonl
    - /mnt/shared-storage-user/yuanjiarui-p/wxh/tb21-eval-qwen36-fixed/data/tb21_polar_pjlab_images.jsonl
loop:
  steps: 10
  candidates: 2
  sampling_repeats: 1
  gate_repeats: 2            # 89 题各跑几次；1 次噪声很大，见第 5 节
  validation_pool_size: 32
  sampling_concurrency: 32
gate_model:
  names: [<模型副本 1>, <模型副本 2>]
  gateway: http://<模型服务地址>
env_files:
  - /path/to/opensandbox.env  # 沙箱凭证，--dry-run 不会打印其中的值
env:
  HARNESSRL_OPENCODE_SETUP: afs_bin
  HARNESS_AFS_PUBLISH: "1"
  HARNESS_AFS_PUBLISH_IMAGE: ailab/hamishivi:swerl-tmax-v3-0e7ceb295a7d  # 上传二进制用的临时沙箱镜像
  HARNESSRL_GATE_TIMEOUT_SEC: "10800"
  SANDBOX_TRIAL_RETRIES: "2"
  OPENCODE_SETUP_TIMEOUT_SEC: "180"
  OPEN_SANDBOX_RECEIPT_RECOVERY_TIMEOUT: "60"
  AGENT_TIMEOUT_POLICY: metadata
  OPEN_SANDBOX_CAPACITY_MODE: local   # 必须写，否则下面两项不生效
  SANDBOX_CAPACITY_SCOPE: run
  SANDBOX_CAPACITY_LIMIT: "64"
```

**沙箱放行模式**：`OPEN_SANDBOX_CAPACITY_MODE` 默认是 `strict`，每次创建沙箱前都要读一遍整个 `ailab-safethm` 项目的沙箱列表，项目里沙箱数和 CPU 申请总量都低于 384 才放行（这个上限写在 polar 的 `capacity.py` 里）。别人的任务占满时，我们的循环会一直排队，1 小时后报 `Sandbox admission timeout`。旧的启动脚本和上面的配置都用 `local`：只按本次运行计数，最多 `SANDBOX_CAPACITY_LIMIT` 个沙箱；记账目录默认是 `<run_dir>/capacity`，run id 默认是 run_dir 的目录名，`hevo run` 会自动补上。

相对路径都相对于 run.yaml 所在目录。完整字段见 `src/harness_evolution/core/run_config.py`。

### 3.4 启动脚本

```bash
#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/shared-storage-user/yuanjiarui-p/wxh/harnessRL/harness-evolution-new
PY=/mnt/shared-storage-user/yuanjiarui-p/.conda/envs/yjr_sglang/bin/python
CODEX_BIN=/mnt/shared-storage-user/yuanjiarui-p/wxh/.local/codex-cli/yuanjiarui-1-zfj8l-3194331-worker-0/bin/codex
BUN_BIN=/mnt/shared-storage-user/yuanjiarui-p/wxh/harnessRL/.bun/bin/bun
PROXY=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128

export PYTHONPATH=$REPO/src PYTHONUNBUFFERED=1
export CODEX_HOME=/mnt/shared-storage-user/yuanjiarui-p/wxh/.codex-chatgpt/<机器名>
export CODEX_CI=1
export HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY http_proxy=$PROXY https_proxy=$PROXY
export NO_PROXY=127.0.0.1,localhost,::1,.pjlab.local,.svc,.svc.cluster.local no_proxy=127.0.0.1,localhost,::1,.pjlab.local,.svc,.svc.cluster.local
export BUN_BIN PATH="$(dirname $PY):$(dirname $CODEX_BIN):$(dirname $BUN_BIN):$PATH"

"$PY" -m harness_evolution.cli run "$1" "${@:2}"
```

保存为 `launch.sh`，然后：

```bash
bash launch.sh my-run.yaml --dry-run      # 1. 只打印主循环参数和环境变量，不改任何东西
nohup bash launch.sh my-run.yaml > my-run.log 2>&1 &   # 2. 正式启动
```

`hevo run` 会把 evolve/gate 配置冻结成 `run_dir/config.yaml`、`run_dir/gate.yaml`。如果 `run_dir/state.json` 已存在，全新启动会被拒绝，防止覆盖已有实验。

### 3.5 监控

```bash
"$PY" -m harness_evolution.cli status <run_dir>      # 每步是否晋级、held-out 分数
tail -f <run_dir>/console.log                         # 实时日志
cat <run_dir>/step-XX/gate.json                       # 某一步每个候选的静态检查和 held-out 结果
cat <run_dir>/diagnosis-cache/<hash>/aggregate.md     # 某一步的诊断汇总
```

按以往实验，一个完整的 step（采样、诊断、生成候选、89 题评测）大约 1.6～2.5 小时；`step-XX/step.json` 的 `duration_sec` 记录了每步实际耗时。

### 3.6 中断后续跑

```bash
nohup bash launch.sh my-run.yaml --resume > my-run.resume.log 2>&1 &
```

续跑以 `state.json` 为准：已完成的 step 不重跑，已完成的诊断和 parent 成绩从缓存读取，评测到一半的 step 会复用已完成的题目。

### 3.7 不需要卡的小规模验证

只验证“诊断 → 生成候选 → typecheck/build”，用已有轨迹，不采样也不评测，约 20 分钟：

```bash
"$PY" -m harness_evolution.cli evolve --config evolve.yaml --job-dir <已有的 held-in 采样目录> --rollouts 1
```

可直接参考已跑通的例子：`/mnt/shared-storage-gpfs2/safewt-share02/sunyoubang/yuanjiarui/wangxianghe/evoloop/smoke-harness-evolution-new-20261012/`（`run_smoke.sh` + `config.yaml`）。

### 3.8 用 MiMo 题目做 held-in 和门控

held-in 可以从 TMAX 换成 MiMo（MiMo v2.6 RL 开源题，`docker.io/xiaomimimo/mimo-v2.6-rl-oss:<task_id>`）。门控（held-out）既可以保留 89 道 TB2.1，也可以换成一组固定的 MiMo 门控集；`configs/examples/run.mimo.yaml` 用的是后者，TB2.1 留作循环结束后的最终评估。

**数据。** 同事用 Qwen3.6-35B-A3B base 对 2762 道 MiMo 题各跑了 8 次（`duchenyu/eval-pipelines/qwen36-base-mimo-tmax-k8-20260930/`），按成功次数分成 `screening/mimo/success_<c>_of_8.txt`。只保留成功 1～7 次的题：0/8 的太难、8/8 的太简单，都提供不了改进信号。

```bash
S=/mnt/shared-storage-gpfs2/safewt-share02/sunyoubang/yuanjiarui/duchenyu/eval-pipelines/qwen36-base-mimo-tmax-k8-20260930
"$PY" -m harness_evolution.tasks.mimo_catalog \
  --mimo-jsonl $S/inputs/mimo.jsonl \
  --screening  $S/screening/mimo \
  --quarantine $S/screening/env_quality/quarantine_registry.json \
  --output     <输出>.jsonl          # 旁边会生成 <输出>.summary.json
```

已生成好的一份：`evoloop/data/mimo-heldin-qwen36-1to7-20261014.jsonl`，1709 道（opensource-code 1672、terminal_bench 37）。可用 `--min-success/--max-success/--families` 调整筛选范围。

**MiMo 门控集。** 上面的命令加 `--gate-size 100 --gate-seed 20261014 --gate-output <门控>.json`，会从成功 2～6 次的题里按（题型, 成功次数）分层抽 100 道作为固定门控集，并把它们从 held-in 输出里去掉，两边不会重叠（主循环启动时也会检查）。会生成三个文件：`<门控>.json`（题目 id 列表，给 `held_out`）、`<门控>.jsonl`（这些题的 catalog 行，给 `task_catalogs`）、held-in 的 `<输出>.jsonl`。已生成好的一份：

- 门控：`evoloop/data/mimo-gate100-qwen36-2to6-20261014.json`，100 道（code 99、terminal_bench 1；成功 2/3/4/5/6 次的分别 26/23/17/19/15 道）
- held-in：`evoloop/data/mimo-heldin-qwen36-1to7-minus-gate100-20261014.jsonl`，1609 道

门控题和进化题同分布但不重叠，测的是对没见过的同类题的泛化；不要直接用每步进化的 32 道题做门控（候选正是针对这些题改的，分数会偏高，而且每步题目不同，父代基准无法复用）。

**配置。** 只需改 `run.yaml` 的 `data`，见 `configs/examples/run.mimo.yaml`：`held_in_catalog` 指向 held-in 文件，`held_out` 指向门控 `.json`，`task_catalogs` 列门控 `.jsonl`（held-in catalog 会自动加入）。想保留 TB2.1 门控时，`held_out` 用 `tb21_all_89.json`，`task_catalogs` 列 TB2.1 的 catalog。

**和 TB 题的区别**（`tasks/catalog.py` 识别 catalog 里的 `task_file`，`evaluation/backends/mimo.py` 负责执行）：

| | TB 题 | MiMo 题 |
|---|---|---|
| 工作目录 | 全局 `workdir`，`HOME` 同目录 | 每道题自己的 `cwd`（`/testbed`、`/workspace/repo`、`/app`）；agent 用 `HOME=/home/user`，OpenCode 配置不进仓库；准备、打补丁和 verifier 用 `HOME=/root`（和筛选一致：`/root/.bashrc` 才会把 pyenv 等加进 PATH，否则部分题的测试脚本报 `pyenv: command not found`） |
| 资源 / 超时 | 全局设置 | 每题至少 8 GiB、按题目 CPU；agent 900 秒（与筛选时一致），verifier 用题目自己的超时 |
| agent 之前 | 上传 `tests/` | code 题：记录 git 基线，删除基线之后的历史（防止看到参考答案） |
| 评测 | `tests/test.sh` | code 题：恢复测试补丁涉及的文件 → 打隐藏测试补丁 → 跑 `test_command`，退出码 0 为通过；terminal_bench 题：上传题目里内嵌的 `tests/` 后跑 `test.sh` |
| 额外产物 | — | `agent/model.diff`（agent 的改动）、`mimo/` 下每一步命令的输出 |

评测流程按筛选脚本（`duchenyu/reports/mimo-smoke-20260926/run_smoke.py`）实现，held-in 通过率可以直接和筛选结果对比。拿不到可信 verifier 结果时（没有退出回执、补丁打不上、verifier 自身异常退出）记为基础设施错误，不算 agent 失败。

**和筛选脚本不同的地方：** 筛选时额外给沙箱里的 Python/Node 限制了并行线程数（`mimo_sandbox_resilience.py`），这里没有加；`model.diff` 目前只保存在采样目录里，还没有放进诊断的输入。

**状态：** 单元测试覆盖了题目解析、补丁路径、回执判定，以及用假沙箱跑通 code 题和 terminal_bench 题的完整 trial；用真实数据验证了 10 步×32 道的选题和全部题目解析。2026-10-14 在真实沙箱里用 Qwen3.6 冒烟测试了 5 道题（4 道 code、1 道 terminal_bench，`evoloop/mimo-sandbox-smoke-20261014/`），通过 3 道、失败 2 道、基础设施错误 0；准备、补丁、verifier 和轨迹文件都正常。

## 4. 已有实验结果

| | OpenCode 进化（Qwen3.6） | Codex 进化（gpt-5.6-sol） |
|---|---|---|
| run 目录 | `evoloop/harnessRL-tmax32-fulltb21-loop-20261007-qwen36-diagnosis-v8` | `evoloop/harnessRL-tmax32-fulltb21-loop-20261008-codex56sol-v8-rjob` |
| held-out 起点 → 终点 | 23 → 28（step 1、5 晋级） | 26 → 34（step 1、8 晋级） |
| 进了 held-out 的候选 | 10 / 20 | 20 / 20 |
| gate_repeats | 1 | 1 |

每步分数图（由各 step 的 `gate.json` 整理）：

- [figures/qwen36-gate-scores.svg](figures/qwen36-gate-scores.svg)
- [figures/codex56sol-gate-scores.svg](figures/codex56sol-gate-scores.svg)

2026-10-11 开 memory 的两次实验在 step 0 就中断了，没有结果。

## 5. 已知问题和经验

1. **单次评测噪声大。** `gate_repeats: 1` 时，同一 parent 下的候选分数在 22～31 之间波动，差 1～2 题基本是噪声。晋级时又直接拿候选这一次的成绩当新 parent 的成绩，偶然偏高的成绩会抬高之后的门槛。建议至少 `gate_repeats: 2`。
2. **`node_modules` 链接失效会让所有候选 typecheck 失败。** Qwen 实验 step 7～10 的 8 个候选全部报 `Module not found .../rl_rollout_opencode_modular_20260928_.../shared_repo/node_modules/...`：parent 目录的 `node_modules` 链接到了一个已被清理的旧工作区。启动和续跑前确认 parent 目录下的依赖可用。
3. **沙箱错误按 0 分计。** 每次 89 题评测通常有 4～12 道题因沙箱错误被记为 0 分；个别候选会高达 54 道（Qwen 实验 step 6），这种成绩不可信。看分数时同时看 `gate.json` 里的 `raw_errors`。
4. **Codex 容量不稳定。** gpt-5.6-sol 有时长时间返回 “Selected model is at capacity”。开 `service_tier="fast"` 反而更容易被拒。
5. **弱模型诊断会把 verifier 的问题当成 harness 的问题。** Qwen3.6 诊断时把 “verifier 期望值错了” 判成和 harness 高度相关，因为把测试框架（test harness）和 OpenCode harness 混淆了；gpt-5.6-sol 没有这个问题。汇总阶段会把这类情况归入 `task_specific_patterns`，大多数不会影响改动。
6. **候选可能什么都没改。** Qwen 实验里有 2 个 no-op 候选，会被静态门控直接淘汰。
7. **2026-10-11 开 memory 的实验在 step 0 中断**，日志分别是 “analysis preflight found no eligible held-in traces” 和大量 “verifier dependency missing”，说明当时采样环境有问题，启动前要确认采样能产出完整轨迹。
8. **新仓库还没有跑过完整的 89 题评测。** 正式实验切换到新仓库前，建议先跑 1～2 步的小实验确认端到端可用。
