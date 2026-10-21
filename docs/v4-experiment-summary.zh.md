# v4 实验总结与 v5 准备

实验目录：`evoloop/harnessRL-mimo32-gate100-loop-20261014-v4`（2026-10-14 17:55 启动，2026-10-15 10:44 因磁盘配额写满崩溃）。
配置：MiMo 题 32 道 held-in（每步），固定 100 道 MiMo 门控集 × 1 次，每步 4 个候选，候选 Codex 用 `gpt-5.6-sol`，起点 `1802c9e`（OpenCode v1.17.9 + 请求记录补丁）。

## 一、结果

门控分数是 100 题通过数。基线分数来自缓存：每步的基线就是上一步胜出者当时的门控分数。

| 步 | 父版本 | 基线 | 4 个候选 | 结论 |
|---|---|---|---|---|
| 1 | 1802c9e | 43 | 未取到（目录名与后几步不同） | 接受 `8b236f9`，49 |
| 2 | 8b236f9 | 49 | 49 / **55** / 39 / 49 | 接受 `cecbb0bb`，55 |
| 3 | cecbb0bb | 55 | 44 / 42 / 50 / 49 | 全部被拒 |
| 4 | cecbb0bb | 55 | 48 / 48 / 51 / 43 | 全部被拒 |
| 5 | — | — | — | 采样阶段磁盘配额写满，进程退出 |

- 4 个候选的静态检查（typecheck、build）在 step 1–4 的控制器里全部通过。
- 崩溃时 `state.json` 还停在 07:19 的内容（`status: running`、`next_step: 4`、历史到 step 3），step 4 的拒绝结果只在 `step-04/gate.json` 里。直接 `--resume` 是否会重做 step 4 没有验证过，所以 v5 选择新开一轮，不续跑。
- 最终 TB2.1 评估没有做（循环里没有自动步骤）。

## 二、这轮发现的问题

### 1. 循环实际使用的是未修复的旧 skill

- 修过的 `SKILL.md` 在 `harness-evolution-new`（未提交），但 v4 的 `config.yaml` 里 `skills.source_dir` 指向旧仓库 `harness-evolution/skills`，那份自 9 月 28 日没变过。
- 旧 skill 仍把 `packages/core/src/session/runner/` 列为“没有自我验证或过早宣布成功”的修改位置，而评测实际走的是 `packages/opencode/src/session/prompt.ts` 里的 `SessionPrompt` 循环，所以候选会改到不生效的地方。
- 新 skill 补了：评测实际走的执行路径、循环退出条件（`lastUser.id < lastAssistant.id`，往已有 user 消息追加 part 不会重新打开循环，必须新建 user 消息）、运行时工具名对照表（运行时是 `bash`，没有 `shell`）、改动必须可达、用一个具体场景证明改动会触发。
- 我用 step 1 的诊断工作区重放了一次（候选 2 个）：
  - 候选 1 改对了位置和方式：在 `prompt.ts` 的退出分支里新建合成 user 消息继续循环；typecheck、build 通过。
  - 候选 2 位置也对，但控制器 typecheck 失败（`prompt.ts(1163,11)` TS2322）。
- 结论：新 skill 能把候选带到正确文件和正确做法上；但这不等于分数会涨，还没有在门控上验证。

### 2. 候选自己跑 `bun typecheck` 偶发 127（`tsgo: command not found`）

- 原因：`_codex_client_env` 给 Codex 设的 `BASH_ENV` 脚本里写的是 `export PATH=<固定路径>`，整体覆盖，bun 临时加进去的 `node_modules/.bin` 随之丢失。
- 是否触发取决于 bash：Ubuntu 的 bash 在 stdin 是套接字、且 shell 层级小于 2 时认为自己是远程 shell，不读 `BASH_ENV`。Codex 用套接字 stdin，所以只有 bun 启动的 `bash -c` 看到 `SHLVL≥1` 时才会读到脚本。
- `SHLVL` 又取决于命令写法：`bun typecheck` 单独或排在最后时 bash 直接 exec，正常；`bun typecheck && xxx`、`bun typecheck; xxx` 要 fork，必然 127。
- 控制器的 typecheck、build 不经过这个脚本，不受影响；受影响的只是候选自己的验证。v4 里 `max_attempts: 1`，控制器报错没有重试。
- 已修（见下文“v5 的改动”）。

### 3. 门控的基线有“赢家诅咒”（推断，未验证）

- 代码里（`two_sided.py` 约 1082 行）胜出者的门控分数直接写进基线缓存，成为下一步的基线。
- 但胜出者是 4 个候选里挑最高的那个，这个分数天然偏高。step 2 的 55 是 4 个候选里的最大值（其他 49、49、39）；而 step 3、step 4 共 8 个候选的平均只有约 46.9。
- 一个佐证：step 2 的基线 49 对应的就是 step 1 胜出者自己的 49。
- 这意味着 step 3、4 不是“候选变差”，也可能只是基线被抬高。要验证需要把 `cecbb0bb` 重新独立评测几次看真实水平。
- 这个问题 v5 没有改，需要你决定怎么处理（见“未包含”）。

### 4. 线程限制（v4 没有）

- 沙箱里 `nproc` 显示 ~320 核，实际配额 2 核，数值库开太多线程会卡死（`format-code-task-001226` 的 verifier 卡 1800 秒）。
- 线程限制的代码是 10-14 19:20 才改的，v4 在 17:55 就启动了，所以 v4 没用上。代码已改、测试通过、端到端复测通过，v5 会带上。

### 5. 其他

- **磁盘**：10:44 崩溃的直接原因是 `gpfs2` 写满配额。现在 `gpfs2` 剩 603G，可以写。但 **`gpfs1` 现在是 100G 用满、剩余 0**，写 20MB 的测试文件写出 0 字节，只能写极小的文件。Codex 凭证、代码仓库都在 gpfs1 上。
- **MiMo 的 verifier 摘要字段错误**：`verifier-summary.json` 对 MiMo 题写成 `verifier_receipt_valid: false`、`agent_exit: null`，但 `reward` 是对的。诊断可能因此误以为 verifier 没跑。这个文件不是我们仓库里生成的，没有改。
- **沙箱偶发错误**：`409 sandbox route not active`、连接超时等，目前靠自动重试兜住。

## 三、v5 的改动

v5 目录：`evoloop/harnessRL-mimo32-gate100-loop-20261015-v5`，从 v4 复制 `configs/`、`launch.sh`、`submit_rjob.sh`，只改以下几处。

| 改动 | 位置 | 说明 |
|---|---|---|
| 使用修复后的 skill | `configs/evolve.yaml` 的 `opencode_evolver.skills.source_dir` | 改为指向 `harness-evolution-new/src/harness_evolution/harnesses/opencode/skills`，不再依赖旧仓库 |
| `BASH_ENV` 不再覆盖 PATH | `harness-evolution-new/src/harness_evolution/proposer/batch.py` 的 `_codex_client_env` | 预设目录放前面，已有目录去重追加；新增 `tests/test_codex_env.py`，12 个用例覆盖 stdin 三种、`SHLVL` 四种 |
| 控制器报错后重试一次 | `configs/evolve.yaml` 的 `validation.max_attempts` | 1 → 2，与 `feed_errors_to_evolver: true` 配合 |
| 线程限制 | `evaluation/backends/mimo.py`（已有代码） | 之前已改好，v5 起生效 |
| 起点检出检查 | `loop/two_sided.py::_require_checkout_at`（已有代码） | v5 起生效 |
| 运行名 | `submit_rjob.sh` | `...-v5` |

验证：
- 全部测试 116 个通过。
- 新旧脚本对照：在之前必现 127 的工作区里，旧脚本 `bun typecheck && true` 报 127，新脚本在 `SHLVL=1/2` 下都正常。
- `launch.sh --dry-run` 通过，起点工作区 HEAD = `1802c9e`。

## 四、v5 未包含

- **门控基线的赢家诅咒**：可选做法是接受后重新独立评测一次胜出者做下一步基线（每个接受步多 100 题），或把 `gate_repeats` 提到 2，或提高接受门槛。会改变实验设置，需要你决定。
- **`model.diff` 进诊断**：MiMo code 题的 agent 改动只存在采样目录，诊断看不到，改动较大。
- **候选“改动是否真的生效”的冒烟检查**：skill 里已经要求，但控制器没有强制。
- **MiMo verifier 摘要字段**：不在我们仓库里生成。
- **提交与推送**：所有改动（含 `SKILL.md`、线程限制、`_require_checkout_at`、`_codex_client_env`、测试）都还没提交。

## 五、v5 启动前必须处理

- **清理 gpfs1**：见上文，当前基本写不进去。清出几个 G 之前不建议启动，否则 Codex 候选生成阶段可能失败。
- 清理旧目录（todo 第 13 项）也能给 gpfs2 留余量，需要你确认后再删。
