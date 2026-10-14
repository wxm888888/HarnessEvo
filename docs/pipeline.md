# HarnessRL 进化实验 Pipeline 手册

本文定义 HarnessRL 在 Terminal-Bench 2.1 和 TMAX 上进行“采样 → 诊断 → 汇总 → harness 改进 → 编译 → 动态 gate → 下一轮”的统一流程。

所有新实验的结果、轨迹、日志、状态、候选工作区和二进制都写入 GPFS2：

```text
/mnt/shared-storage-gpfs2/safewt-share02/sunyoubang/yuanjiarui/wangxianghe/
```

不要把新的实验结果写入 `/mnt/shared-storage-user/yuanjiarui-p/wxh`。代码和只读配置可以从 wxh 或 dcy 读取，但每个实验必须使用 GPFS2 下独立的 `RUN_ROOT`。

## 1. 实验目标和数据划分

每一轮有两个用途不同的数据集：

- **Held-in：32 个新的 TMAX 题目**。只用于采样、逐轨迹诊断、跨轨迹汇总和 harness 进化。每轮换一批题目，不因为候选晋级与否而重复上一轮的题目。
- **Held-out：完整的 89 个 TB2.1 题目**。只用于候选 gate 和最终比较，不作为进化模型的输入。

晋级只根据 89 个 held-out TB2.1 题目的结果。held-in 分数可以记录和分析，但不能代替 held-out gate 分数。

推荐的循环关系：

```text
当前 parent harness
       │
       ├── 32 个新的 TMAX held-in → 轨迹诊断和汇总 → 生成 N 个候选
       │
       └── 完整 89 个 TB2.1 held-out → baseline/candidate gate
                                      │
                    最高且满足约束的候选成为下一轮 parent
                                      │
                         下一轮重新抽取 32 个 TMAX
```

候选数必须是可调参数，通常使用 2；不要把候选数量写死在代码里。

## 2. 推荐目录布局

```text
$RUN_ROOT/
├── manifest.json
├── console.log
├── configs/
│   ├── evolution.yaml
│   ├── gate.yaml
│   └── held-out-89.json
├── held-in-schedule.json
├── state.json
├── step-00/
│   ├── sampling/
│   ├── diagnosis/
│   └── baseline/
├── step-01/
│   ├── sampling/
│   ├── diagnosis/
│   ├── candidates/
│   ├── gate-reports/
│   ├── gate.json
│   └── step.json
└── logs/
```

每一轮都应该有独立的 `step-XX`，不要覆盖上一轮的轨迹或 gate 结果。候选源码工作区和 Bun 二进制也放在 GPFS2。

## 3. 阶段 A：实验前检查

启动前固定记录：

- parent harness 的完整 commit 和二进制路径；
- 模型 endpoint、replica 列表和端口；
- TMAX catalog、TB2.1 catalog 和 task 数量；
- agent timeout、verifier timeout 和重试次数；
- OpenSandbox API、capacity scope、run id 和本地容量账本目录；
- 采样并发、候选数量、gate 并发和统一 sandbox 创建池；
- 当前代码版本、配置文件 checksum 和 Python 环境。

必须确认：

```bash
# TMAX catalog
/mnt/shared-storage-user/yuanjiarui-p/duchenyu/tmax-hamishivi/tmax_hamishivi_all_4487.jsonl

# TB2.1 catalog
/mnt/shared-storage-user/yuanjiarui-p/wxh/tb21-eval-qwen36-fixed/data/tb21_polar_pjlab_images.jsonl
```

在真正跑 32/89 题之前先做一个 sandbox smoke test，确认创建、命令执行、verifier 和删除都成功。模型 endpoint 也要先检查四个 replica 是否可达。

## 4. 阶段 B：采样 baseline 和 held-in 轨迹

每个 trial 的实际顺序是：

1. 创建 sandbox；
2. 上传任务目录和 verifier 所需文件；
3. 安装当前 parent harness 二进制；
4. 启动 OpenCode；
5. 通过 OpenAI-compatible endpoint 请求模型；
6. 保存请求上下文和 OpenCode 事件；
7. agent 阶段结束后执行 verifier；
8. 保存通过、失败、超时或基础设施错误；
9. 删除 sandbox，并确认删除结果。

agent timeout 和 verifier timeout 是两个阶段。agent 超时后仍然进入 verifier；只有整个 sandbox 生命周期或控制面失败，才记为 infrastructure error。基础设施错误应按统一策略重试，不能静默转换成普通失败。

baseline 可以复用，但复用前必须检查：

- task 集合完全相同；
- parent commit 和模型 endpoint 相同；
- timeout、重试和 verifier 配置相同；
- 轨迹文件和结果文件没有被截断；
- 复用路径写入 `manifest.json` 和 `state.json`。

如果任一项不一致，应重新采样 baseline。

实现上的复用粒度是 parent commit + 固定的 89 个 held-out 题目：

- 同一个 parent 在后续 step 中继续使用同一份 89-task baseline 结果；
- 每轮变化的 32 个 held-in 只单独补采样，并与缓存的 89-task 结果合并；
- 候选晋级时，候选已经完成的 89-task 结果直接成为新 parent 的 baseline，不再重复跑；
- 缓存保存在 run 目录的 `baseline-cache/<commit>.json`，并校验题目集合、模型 endpoint、timeout、重试和任务目录元数据。

## 5. 单条轨迹的标准输入

进化模块处理一条可分析轨迹时，标准输入是以下五类文件：

```text
instruction.md
trajectory.json                 # 或等价的 trajectory.jsonl
openai-messages.jsonl           # OpenAI 增量消息格式
openai-tools.json               # 工具定义
verifier-summary.json           # 脱敏后的结果摘要
verifier-feedback.json          # verifier 执行输出与解析后的失败检查
```

各文件职责：

- `instruction.md`：题目要求和约束；
- `trajectory.json`：OpenCode 事件时间线，包含 assistant 输出、工具调用和工具结果；
- `openai-messages.jsonl`：实际请求的增量消息，保留初始提示以及每轮新增消息，不能每轮重复写完整上下文；
- `openai-tools.json`：当时模型可见的工具 schema 和说明；
- `verifier-summary.json`：只保留 reward、通过/失败、阶段、退出码、超时和基础设施错误类别等信息。
- `verifier-feedback.json`：保留限长、脱敏后的 verifier stdout/stderr，并解析测试数量、失败检查名称和 assertion；不包含 verifier 源码。

沙箱中的 `agent/llm-reqs/*-resp.sse` 是底层模型请求/响应的原始流，只用于低级调试，不是进化输入。OpenSandbox gate 默认不下载这类原始 wire 文件；如需专门排查代理或模型协议问题，可显式设置 `HARNESSRL_KEEP_RAW_TRACE_WIRE=1`。

以下文件不是进化模型的主要语义输入：

- `verifier.txt`：禁止提供，避免泄露 checker 细节和产生 reward hacking；
- `agent.remote.jsonl`：只作为调试备份，不能代替 OpenAI 增量轨迹；
- `trial.json`、`result.json`：用于状态统计和恢复，通常不直接送入模型；
- 原始 verifier stdout/stderr、隐藏测试和期望答案：禁止提供给进化模型。

如果某条 trial 只有 `agent.remote.jsonl` 或只有旧版 `trajectory.json`，它可以进入调试和统计，但不能被标记为“完整上下文轨迹”。

## 6. 阶段 C：artifact preflight

进入诊断前必须检查本轮 32 个 held-in trial：

```bash
find "$HELD_IN_DIR" -name 'instruction.md' | wc -l
find "$HELD_IN_DIR" -name 'trajectory.json' -o -name 'trajectory.jsonl' | wc -l
find "$HELD_IN_DIR" -name 'openai-messages.jsonl' | wc -l
find "$HELD_IN_DIR" -name 'openai-tools.json' | wc -l
find "$HELD_IN_DIR" -name 'verifier-summary.json' | wc -l
find "$HELD_IN_DIR" -name 'verifier-feedback.json' | wc -l
find "$HELD_IN_DIR" -name 'result.json' | wc -l
```

同时生成 `coverage.json`，至少记录：

```json
{
  "selected": 32,
  "diagnosable": 0,
  "passed": 0,
  "failed": 0,
  "infrastructure_errors": 0,
  "missing_artifacts": [],
  "fallback": false
}
```

基础设施错误和缺少完整上下文的 trial 必须留在统计中，但不能伪装成正常失败轨迹交给诊断模型。缺少完整上下文的 trial 会保留在 coverage 中并跳过诊断证据；只要本轮仍有至少一条可分析的完整轨迹，提案阶段就可以继续。只有完整轨迹数量为 0 时，才阻止提案。

## 7. 阶段 D：逐轨迹诊断

每次诊断只分析一条 trial。推荐读取顺序：

1. `trajectory.json`；
2. `openai-messages.jsonl`；
3. `openai-tools.json`；
4. `instruction.md`；
5. 脱敏后的 `verifier-summary.json`；
6. `verifier-feedback.json`，先确认 verifier 实际拒绝了什么；
7. 必要时读取 `llm-context.jsonl` 核对请求边界。

正确和错误轨迹都要分析。正确轨迹用于提取有效策略，错误轨迹用于发现失败模式；不能只保留 reward 为 1 的轨迹。

单轨迹诊断要回答：

- 任务目标和限制是什么；
- agent 的关键行动和工具调用是什么；
- 哪个决策、工具行为或上下文因素导致成功/失败；
- 是否存在重复调用、无效循环、错误恢复、超时、环境依赖或进程生命周期问题；
- 问题属于 agent 推理、工具使用、harness 行为还是基础设施；
- 哪个 harness 模块应该怎样改，以及如何验证。

输出：

```text
step-XX/diagnosis/per_trace/<trace_id>.json
```

每个诊断至少包含 `trace_id`、`task`、`reward_status`、`failure_category`、`evidence_spans`、`root_cause`、`harness_surface`、`recommendation` 和 `confidence`。

## 8. 阶段 E：跨轨迹汇总

汇总器读取本轮所有可分析的逐轨迹诊断，不要把 32 条完整原始轨迹一次性拼进一个 prompt。

输出：

```text
step-XX/diagnosis/coverage.json
step-XX/diagnosis/aggregate.json
step-XX/diagnosis/aggregate.md
```

汇总必须包含：

- 共性失败模式；
- 成功模式；
- 每个模式的触发条件；
- 受影响的 OpenCode/harness 模块；
- 支持该结论的 trace id；
- 具体改进建议；
- 风险、潜在副作用和验证方法。

`aggregate.json` 和 `aggregate.md` 不是可有可无的报告，而是 harness 改进器的主要跨轨迹输入。改进器可以回读 `diagnosis/per_trace/*.json` 和原始五类轨迹文件进行核对。

## 9. 阶段 F：生成候选 harness

候选改进器的输入包括：

```text
coverage.json
aggregate.json
aggregate.md
diagnosis/per_trace/*.json
必要时：instruction.md、trajectory.json、openai-messages.jsonl、openai-tools.json、verifier-summary.json
当前 OpenCode 仓库
opencode-architecture-evolution skill
```

改进约束：

- 只修改通用 OpenCode/harness 行为；
- 不修改 benchmark 题目解法；
- 不读取或修改其他仓库；
- 不读取 `verifier.txt`；
- 每个候选使用独立 worktree；
- 候选数量由 `--candidates` 或配置文件控制；
- 候选结束后必须产生 commit，并记录 changed files。

候选生成失败、没有改动或只生成文本建议时，标记为 invalid，不进入 gate。

## 10. 阶段 G：静态 gate 和 Bun 编译

每个候选至少执行：

1. no-op 检查；
2. TypeScript typecheck；
3. Bun build；
4. 最小 smoke test；
5. 记录完整 commit、二进制路径和日志。

静态 gate 不通过的候选不能占用完整 89 题 gate 资源。

## 11. 阶段 H：动态 held-out gate

对 parent 和每个静态通过候选使用同一批完整 89 个 TB2.1 题目：

- 相同的模型 endpoint/replica；
- 相同的 task 顺序；
- 相同的 timeout 和 retry policy；
- 相同的 verifier；
- 相同的 sandbox 创建池；
- 相同的并发配置。

推荐让 baseline、候选 1、候选 2 共用一个 gate 调度池，以减少环境差异。

每个任务都必须保存：

```text
trial.json
result.json
retry_history.json
agent.remote.jsonl
verifier 输出和错误分类
```

动态 gate 的晋级规则：

1. 先比较 89 个 held-out 的有效通过数或通过率；
2. 候选必须达到完整性要求，不能有未完成任务；
3. 基础设施错误必须重试并单独报告，不能静默当成 agent 失败；
4. 选择 held-out 分数最高且满足静态 gate 的候选；
5. 如果没有候选超过 parent，则 parent 晋级到下一轮；
6. 无论是否晋级，下一轮都重新抽取 32 个 TMAX held-in。

历史实验曾把 32 个 TMAX 和 89 个 TB2.1 合并调度成 121 个 trial，但最终应按 held-in/held-out 分开统计，晋级只看 89 个 TB2.1。

## 12. 阶段 I：进入下一轮和 resume

每轮结束写入：

```text
step-XX/gate.json
step-XX/step.json
state.json
```

必须记录：

- parent commit；
- 每个候选 commit；
- 每个候选的 changed files；
- held-in 32 题统计；
- held-out 89 题统计；
- infrastructure error 和 retry 统计；
- 是否接受候选；
- winner commit；
- 下一轮 32 题的 task ids；
- 下一轮轨迹来源目录。

resume 前检查：

```bash
python3 -m json.tool "$RUN_ROOT/state.json"
python3 -m json.tool "$RUN_ROOT/step-XX/gate.json"
find "$RUN_ROOT/step-XX" -name '*.json' -size 0
```

不要只根据 console.log 判断已完成。必须以 `step.json`、`gate.json` 和各 trial 的 `trial.json` 为准。

### 12.1 可插拔的晋升记忆

memory 默认关闭。旧配置未提供 `memory`，或显式设置为下面这样时，候选提示词、候选工作区和状态文件与原流程一致：

```yaml
memory: none
```

需要让后续候选了解当前 harness 的晋升历史时，使用：

```yaml
memory:
  method: accepted_lineage
  options: {}
```

该策略只在候选通过 held-out 动态门控并成为 winner 后追加一条记录。长期文件默认是：

```text
$RUN_ROOT/reports/memory/accepted-lineage.jsonl
```

生成候选前，同一份内容会复制到每个候选工作区：

```text
.harness_evolver/memory/accepted-lineage.jsonl
```

每条记录包含 step、parent commit、winner commit、候选编号、改动摘要、changed files、parent 分数和 winner 分数。摘要只用来定位历史改动；模型需要使用 `git show <winner_commit>` 或 `git diff <parent_commit>..<winner_commit>` 查看真实代码。memory 文件通过 worktree 的本地 Git exclude 排除，不属于候选提交。

如需继承另一轮实验的已接受历史，可配置只读种子：

```yaml
memory:
  method: accepted_lineage
  options:
    seed_path: /absolute/path/to/accepted-lineage.jsonl
```

resume 重复处理同一步时按 `step + winner_commit` 去重。失败候选、静态 gate 失败候选、动态 gate 未晋升候选都不会写入 memory。

## 13. 并发、容量和重试

模型请求并发、sandbox 创建并发和题目评测并发是三个不同参数，不能混为一个 `-j`：

```text
题目评测并发       --jobs / -j
模型请求并发       MODEL_REQUEST_CONCURRENCY
sandbox 创建并发   SANDBOX_CREATE_CONCURRENCY
```

使用本地容量账本时，必须保证 run id、容量目录和实际 worker 环境一致。worker 没有继承环境变量时，本地模式不会生效。

建议对以下错误做有限指数退避重试：

- sandbox route not active / route not found；
- 控制面连接超时；
- verifier tests did not execute；
- command transport error；
- sandbox 创建的临时网络错误。

重试必须写入 `retry_history.json`，每次重试使用新的 sandbox，不能复用已经失效的 route。重试仍失败时保留原始错误类别，不要把它伪装成模型 reward。

## 14. 历史两个好候选实验的说明

之前产生两个好候选的实验目录是：

```text
/mnt/shared-storage-gpfs2/safewt-share02/sunyoubang/yuanjiarui/evolution-harnessRL-tmax32-fulltb21-rerun-step1-20261001
```

候选 commit：

```text
768a0a0282b8ffad1e70693c029b193a4ae49e7e
281d5130f27e9e79d8142034b4bc00098d9a28c6
```

它们的 held-out 分数分别是 30/89 和 28/89，baseline 是 24/89。这个实验使用的是旧版输入：主要有 `trajectory.json`、`instruction.md` 和结果摘要，没有完整的 `openai-messages.jsonl`、`openai-tools.json` 和 `llm-context.jsonl`。

因此，这两个候选的效果是真实的，但不能把那次实验描述成“完整 OpenAI 上下文进化”。以后按本手册运行时，应先通过 artifact preflight 确认五类单轨迹输入齐全，再进入诊断和候选生成。

## 15. 常见错误

- 用 held-out 轨迹作为进化输入，导致评测污染；
- 只分析 reward=1 的轨迹，丢失失败模式；
- 把 agent timeout 直接记为 missing 并跳过 verifier；
- 只保存 `agent.remote.jsonl`，却宣称有完整上下文；
- 把 `verifier.txt` 或隐藏测试输出交给进化模型；
- baseline、candidate 使用不同 task 顺序、不同重试策略或不同模型池；
- 用总 121 题分数代替 89 题 held-out 分数做晋级；
- 使用 `max_failed_trials: 1` 导致一次环境错误就让候选失效；
- 修改 wxh 下的实验结果，造成空间不足或污染原始代码；
- 只看 tmux 或 console.log，不核对 `state.json`、`step.json`、`gate.json` 和 trial 文件。
