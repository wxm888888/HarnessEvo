# harness-evolution-new 使用说明

> 仓库位置：`/mnt/shared-storage-user/yuanjiarui-p/wxh/harnessRL/harness-evolution-new`
> 这是旧仓库 `harness-evolution` 的重构版本，功能和行为不变，代码按流程阶段重新组织成一个 Python 包 `harness_evolution`，命令行入口叫 `hevo`。旧仓库保持原样，正在跑的实验不受影响。

## 1. 这个项目做什么

> 每个阶段的判定规则和完整的启动步骤见 [evolution.zh.md](evolution.zh.md)。

我们不训练模型，而是**让强模型（Codex）根据 agent 的运行轨迹，自动修改 agent harness（目前是 OpenCode）的源码**，再用固定的 benchmark 判断改动是否真的有效。

一轮（一个 step）的流程：

```
当前父版本 harness（OpenCode 的某个 commit）
  │
  ├─ ① 采样     在 OpenSandbox 里用父版本跑 32 道新的 TMAX 题（held-in）
  │             → 每道题整理成标准轨迹文件
  ├─ ② 诊断     Codex 逐条分析轨迹（成功、失败都分析）→ 再跨轨迹汇总
  ├─ ③ 生成候选 Codex 在独立 git worktree 里改 OpenCode 源码，默认 2 个候选
  ├─ ④ 检查     静态：typecheck + build；动态：在固定的 89 道 TB2.1（held-out）上评测
  └─ ⑤ 晋级     held-out 得分超过父版本的最好候选成为下一轮父版本；否则父版本不变
```

held-in 只用于诊断和进化，held-out 只用于评测，两者严格分开，避免评测污染。

## 2. 目录结构

```
harness-evolution-new/
├── pyproject.toml              可 pip install，提供命令 hevo
├── README.md                   英文简介
├── configs/examples/run.yaml   完整实验的配置示例
├── docs/
│   ├── guide.zh.md             本文档
│   ├── evolution.zh.md         进化流程详解和循环启动步骤
│   ├── restructure.md          重构计划和新旧文件对照
│   ├── pipeline.md             原实验流程手册（held-in / held-out、产物规范等）
│   └── design-notes/           历史设计笔记
├── src/harness_evolution/
│   ├── cli.py                  hevo 命令行
│   ├── core/                   公共基础：数据类型、配置、路径、git、prompt 加载、run.yaml
│   ├── traces/                 ① 原始沙箱产物 → 标准轨迹文件；完整性预检；脱敏
│   ├── diagnosis/              ② 证据策略：逐条诊断 + 跨轨迹汇总（含 prompts/）
│   ├── proposer/               ③ 生成候选：给 Codex 的指令、worktree、提交（含 prompts/）
│   ├── agents/                 Codex CLI / Claude Code CLI / OpenCode / OpenAI 的调用封装
│   ├── memory/                 晋级历史（accepted lineage），可选（含 prompts/）
│   ├── harnesses/opencode/     被进化对象的适配：Bun 编译、typecheck、skill
│   ├── evaluation/             ④ 静态检查、held-out 评测（OpenSandbox 后端）、计分、baseline 缓存
│   ├── selection/              ⑤ 候选选择和晋级排序
│   ├── tasks/                  数据集划分、每轮选题、任务目录解析
│   ├── loop/                   主循环编排和续跑
│   └── integrations/slime/     和 slime RL 训练的协同进化
├── tests/                      91 个测试（含 prompt 回归快照）
└── tools/trace-proxy/          轨迹抓取代理的源码
```

### 各阶段的关键文件

| 阶段 | 文件 | 作用 |
|---|---|---|
| 主循环 | `loop/two_sided.py` | 正式实验的主循环（held-in 采样 + held-out 评测 + 晋级） |
| | `loop/resume.py` | 续跑、断点、旧产物清理 |
| | `loop/from_existing_job.py` | 只用已有轨迹做诊断和进化（不采样、不评测） |
| ① 轨迹 | `traces/materialize.py` | 从沙箱产物生成 `openai-messages.jsonl` 等标准文件 |
| | `traces/preflight.py` | 检查轨迹是否完整，排除基础设施错误 |
| | `traces/verifier_feedback.py` | 脱敏后的 verifier 输出 |
| ② 诊断 | `diagnosis/per_trace_then_aggregate.py` | 当前使用的证据策略：先逐条诊断、再汇总 |
| | `diagnosis/aggregate.py` | 诊断结果加载、汇总索引、生成 `aggregate.md` |
| ③ 候选 | `proposer/batch.py` | 一轮候选的生成主流程 |
| | `proposer/requirement.py` | 拼装给 Codex 的主指令 |
| | `harnesses/opencode/build.py`、`validate.py` | 编译 OpenCode、跑 typecheck/build |
| ④ 评测 | `evaluation/static_gate.py` | 静态检查 |
| | `evaluation/backends/opensandbox.py` | 在 OpenSandbox 上跑每道题 |
| | `evaluation/scoring.py`、`baseline_cache.py` | 计分；父版本 89 题结果按 commit 缓存 |
| ⑤ 晋级 | `selection/promotion.py` | 候选排序规则 |

## 3. 环境准备

在集群上直接用实验的 Python 环境即可：

```bash
cd /mnt/shared-storage-user/yuanjiarui-p/wxh/harnessRL/harness-evolution-new
PY=/mnt/shared-storage-user/yuanjiarui-p/.conda/envs/yjr_sglang/bin/python   # Python 3.12，已有 yaml/openai
export PYTHONPATH=$PWD/src          # 或者 pip install -e . 后直接用 hevo 命令
alias hevo="$PY -m harness_evolution.cli"
```

还需要：

| 依赖 | 说明 |
|---|---|
| `polar`、`harness_dist` | 同级目录下的源码，OpenSandbox 评测需要。默认在仓库的上一级目录查找，可用 `HARNESSRL_ROOT` 指定 |
| Codex CLI | 诊断和改代码都用它。`CODEX_HOME` 必须指向**有效的登录目录**，例如 `.codex-chatgpt/<机器名>`。`.codex-rjob/codex56sol-v8-20261008` 已于 2026-10-12 失效（401） |
| Bun | 编译 OpenCode：`/mnt/shared-storage-user/yuanjiarui-p/wxh/harnessRL/.bun/bin/bun`，需要加进 `PATH` |
| 代理 | Codex 需要出网：`HTTP_PROXY=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128` |

## 4. 使用方法

### 4.1 不需要卡：用已有轨迹做诊断和进化

适合调 prompt、调诊断策略，或者验证代码改动。约 20 分钟（4 条轨迹、1 个候选）。

```bash
hevo evolve --config evolve.yaml --job-dir /path/to/held-in-job --rollouts 1
```

- `--job-dir`：已有的采样目录，每道题下面有 `agent/trajectory.json` 和 `result.json`，比如旧实验的 `step-XX/sampling/...-held-in`。
- `evolve.yaml` 里所有输出路径（`workspace_root`、`report_root`、`evidence.options.result_root` 等）要指向自己的目录。

可以直接参考已经跑通的例子：
`/mnt/shared-storage-gpfs2/safewt-share02/sunyoubang/yuanjiarui/wangxianghe/evoloop/smoke-harness-evolution-new-20261012/`。里面的 `run_smoke.sh` 配好了环境变量，`config.yaml` 是对应的配置。

产物：
- `diagnosis-cache/<key>/per_trace/*.json`：逐条诊断
- `diagnosis-cache/<key>/aggregate.md`：汇总结论
- `workspaces/.../worktrees/attempt_001_rollout_001`：候选的源码改动和 commit
- `reports/modular_iteration_001.md`：候选和校验结果

### 4.2 完整实验（需要模型 endpoint 和 OpenSandbox）

1. 复制 `configs/examples/run.yaml`，填好 `run_dir`、evolve/gate 配置、数据路径、gate 模型和环境变量。
2. 先用 dry run 检查：它会打印主循环参数和导出的环境变量，不会改动任何东西；`env_files` 里的值会被隐藏。

   ```bash
   hevo run my-run.yaml --dry-run
   ```

3. 正式运行：

   ```bash
   hevo run my-run.yaml
   ```

4. 中断后续跑：

   ```bash
   hevo run my-run.yaml --resume
   ```

5. 查看进度：

   ```bash
   hevo status /path/to/run_dir
   ```

   输出示例：

   ```
   status: complete  next_step: 11
   current commit: 2abd490...
     step 1: accepted  515b6b3 -> b6543ac  held-out 30/89
     step 2: kept parent  b6543ac ->   held-out -
   ```

在 rjob 上跑时，提交 rjob、Codex 登录检查和网络预检、代理设置这些集群相关的步骤，仍然由启动脚本负责。脚本最后改成调用 `hevo run` 即可。

### 4.3 其他命令

| 命令 | 作用 |
|---|---|
| `hevo loop ...` | 主循环，使用原始命令行参数（`hevo run` 就是用 run.yaml 拼出这些参数） |
| `hevo gate ...` | 对单个候选 worktree 跑静态检查和 held-out 评测 |
| `hevo rl-round --config rl_evolve.yaml` | RL 训练暂停时调用的一轮协同进化（等价于 `python -m harness_evolution.rl`） |
| `hevo <命令> --help` | 查看对应模块的参数 |

## 5. 配置说明

一次完整实验涉及三份配置：

**`run.yaml`（新增，统一入口）**：描述整个实验。包括 `run_dir`、`evolve` 和 `gate` 两份配置（可以写路径，也可以直接内联）、`data`（held-in 目录、held-in 题数、held-out 划分、任务 catalog）、`loop`（轮数、候选数、重复次数、并发）、`gate_model`、`env_files`、`env`。完整字段见 `core/run_config.py` 的文件头注释。

**`evolve.yaml`（进化配置）**，主要段落：

| 段落 | 内容 |
|---|---|
| `initial` | 起始 harness 仓库路径和 commit |
| `loop` | 每轮候选数、超时、输出目录 |
| `evidence` | 证据策略（`per_trace_then_aggregate`）、诊断并发、分析用的模型（Codex） |
| `opencode_evolver` | 改代码用的 Agent（`codex_cli_client`）、模型、skill |
| `memory` | 晋级历史，默认 `none`，可选 `accepted_lineage` |
| `validation` | 候选生成后跑的命令（typecheck、build） |
| `candidate_selection` | 候选选择方式（`first_valid`） |

**`gate.yaml`（评测配置）**，主要段落：

| 段落 | 内容 |
|---|---|
| `paths` | baseline 仓库、commit 和二进制，以及报告目录 |
| `static_gate` | 是否做 typecheck、build、smoke 和 diff 限制 |
| `heldout_gate` | 评测后端（`opensandbox`）、重复次数、并发，以及 OpenSandbox 的地址、模型、超时等 |

**常用环境变量**（建议写在 `run.yaml` 的 `env` 里）：

| 变量 | 作用 |
|---|---|
| `HARNESSRL_ROOT` | 外部源码（polar、harness_dist）所在目录 |
| `HARNESSRL_OUTPUT_ROOT` | 相对路径的 `run_dir` 以它为根；默认是仓库下的 `runs/` |
| `HARNESSRL_OPENCODE_SETUP` | 沙箱里安装 OpenCode 的方式，正式实验用 `afs_bin` |
| `HARNESSRL_GATE_TIMEOUT_SEC` | 一次 89 题评测的总时限 |
| `SANDBOX_TRIAL_RETRIES` | 基础设施错误的重试次数 |
| `HARNESSRL_KEEP_RAW_TRACE_WIRE` | 设为 1 时保留原始模型请求流，仅用于调试 |

## 6. 运行目录里有什么

`hevo run` 的所有产物都写在 `run_dir` 下：

```
run_dir/
├── config.yaml / gate.yaml     启动时冻结的配置（续跑时沿用）
├── state.json                  当前进度：父版本 commit、下一步编号、每步历史（以它为准，不要只看日志）
├── held-in-schedule.json       每一轮抽到的 held-in 题目
├── console.log
├── step-XX/
│   ├── sampling/               本轮 held-in 采样轨迹
│   ├── candidate-checkpoint.json
│   ├── validation-jobs/        89 题 held-out 评测的每道题结果
│   ├── gate.json               本轮评测和晋级结果
│   └── step.json
├── diagnosis-cache/            诊断和汇总结果（按轨迹内容缓存）
├── baseline-cache/             父版本 89 题结果（按 commit 缓存，晋级前不重复跑）
├── parent-binaries/            各父版本编译出的 OpenCode
└── workspaces/                 候选的 git worktree
```

## 7. 修改 prompt

给 Agent 的 prompt 都是 Markdown 模板，放在对应阶段的 `prompts/` 目录：

| 文件 | 用途 |
|---|---|
| `diagnosis/prompts/trace_diagnosis.md` | 逐条诊断 |
| `diagnosis/prompts/aggregate.md` | 跨轨迹汇总 |
| `diagnosis/prompts/evidence_*.md` | 告诉改代码的 Agent 证据文件在哪、怎么用 |
| `memory/prompts/accepted_lineage.md` | 晋级历史说明 |
| `proposer/prompts/candidate.md` | 改代码的主指令，分段写，可选段落由代码决定是否加入 |

模板用 `${变量名}` 作为占位符，`$$` 表示字面的 `$`。直接改 `.md` 就能生效。`tests/test_prompts.py` 锁定了当前 prompt 的渲染结果，所以**有意修改 prompt 之后，要同步更新 `tests/fixtures/prompts_golden.json`**，否则测试会失败。这是有意设计的，防止不小心改动 prompt。

## 8. 扩展：替换某个阶段的策略

每个可替换的阶段都有一个注册表，在配置里用 `method` 选择：

| 阶段 | 注册表位置 | 现有实现 |
|---|---|---|
| 证据 / 诊断 | `diagnosis/__init__.py` | `per_trace_then_aggregate`（当前使用）、`agent_analyzed`、`sanitized_instruction`、`instruction_only` |
| 晋级记忆 | `memory/__init__.py` | `none`、`accepted_lineage` |
| 选题 | `tasks/selection/__init__.py` | `static_list`、`history_scheduler` |
| 候选选择 | `selection/candidates/__init__.py` | `first_valid`、`smallest_diff_valid` |
| 分析 / 改代码的 Agent | `agents/registry.py` | `codex_cli_client`、`claude_code_cli_client`、`opencode_bun_dev_client`、`openai_chat` |

各策略要实现的接口（`EvidenceStrategy`、`MemoryStrategy` 等）定义在 `core/types.py`。新增一种策略的步骤是：实现接口，在注册表里登记，然后在配置里填上它的名字。

## 9. 测试

```bash
python3 -m pytest -q                 # 91 个测试，不需要 GPU，几秒内跑完
$PY tests/test_imports.py            # 用实验环境逐个 import 所有模块
```

重构过程中的验证：
- 每个被搬动的函数，都和原代码做了 AST 对比，确认完全一致。
- 所有 prompt 都和旧代码逐字节对比过。
- `run.yaml` 拼出的主循环参数和环境变量，与 2026-10-11 实验的启动脚本完全相同。
- 已用 4 条真实轨迹端到端跑通“诊断 → 汇总 → 生成候选 → typecheck/build”。

## 10. 注意事项

- **还没有在新仓库上跑过完整的 89 题 held-out 评测**，因为目前申请不到卡。正式实验切换过来之前，建议先用新仓库跑一轮小规模实验。
- **旧仓库 `harness-evolution` 没有改动。** 实验启动脚本、`fresh-templates/`、历史记录都还在那里；`polar/examples` 里的几个脚本也仍然引用旧仓库。
- 和旧代码行为不同的两处：
  - 相对路径的 `run_dir`，默认根目录从写死的 GPFS2 路径改成了 `$HARNESSRL_OUTPUT_ROOT`（没设时是仓库下的 `runs/`）。传绝对路径不受影响。
  - Harbor 评测后端（已经不用了）的 agent 路径改成了 `harness_evolution.evaluation.backends.harbor_filebridge:...`。
- 有几处长函数还没拆，计划等能跑评测时再处理：`loop/two_sided.py` 的 `main()`（约 500 行）、`evaluation/gate.py` 的多候选评测函数。
- 新仓库目前**还没有 git 提交**。
- 新旧文件逐个对照，见 `docs/restructure.md`。
