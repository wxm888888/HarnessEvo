# Harness Evolution 后续改动思路

本文只描述后续实现方案，不修改当前 Harness Evolution 代码。目标是在 Harness candidate 完成一次 OpenCode 进化后，沿用同一个 OpenCode session，让 agent 总结本次改动，并把总结交给 Reward Design。

## 目标流程

```text
OpenCode 进化 candidate
        ↓
同一个 session 发送总结请求
        ↓
生成 candidate/.harness_evolver/harness_change.md
        ↓
Gate 选择 winner
        ↓
复制 winner 的 harness_change.md 到 run_dir
        ↓
写入 harness_gate_pairs.v1 的 harness_change_analysis_path
        ↓
Reward Design 读取 Markdown 和配对轨迹
```

总结请求只允许分析和输出 Markdown，不允许继续修改 Harness。每个 candidate 都可以生成一份总结，Gate 通过后只使用 winner 的总结。

## 需要修改的位置

### 1. OpenCode client

文件：

```text
harness-evolution/harness-evolver/src/harness_evolver/clients/opencode_bun_dev.py
```

当前 `run_instruction()` 只返回归一化 trajectory。需要让详细结果保存：

```text
session_id
raw_output
trajectory
XDG_DATA_HOME / XDG_STATE_HOME
```

然后增加同一 session 的 follow-up 方法，例如：

```text
continue_session(workspace, session_id, instruction)
```

实现前先确认当前 Bun/OpenCode 版本实际支持的 session continuation 参数或 API。不能假设命令行选项名称。

### 2. Evolver

文件：

```text
harness-evolution/harness-evolver/src/harness_evolver/evolvers/opencode_bun_dev.py
```

把一次进化改为：

```text
run_instruction_detailed()
        ↓
读取 session_id
        ↓
continue_session("总结这次 Harness 改动，只输出 Markdown")
        ↓
写入 .harness_evolver/harness_change.md
```

在 `EvolverResult.metadata` 中保存：

```json
{
  "harness_change_analysis_path": ".../.harness_evolver/harness_change.md",
  "harness_change_analysis": "..."
}
```

总结 Prompt 应要求模型覆盖：

1. 修改了哪些文件以及每个文件的作用；
2. 解决了什么问题；
3. 新增了什么 Harness 能力；
4. 什么事件或行为表示能力被触发；
5. 什么行为属于正确使用；
6. 什么行为属于错误使用；
7. baseline/candidate 轨迹中有哪些证据；
8. 如果只是提示词改动，说明提示词希望模型形成的行为。

### 3. Candidate artifact 和 pipeline

文件：

```text
harness-evolution/scripts/opencode_harness_evolver_loop_modular.py
harness-evolution/scripts/harness_evolution_pipeline.py
```

candidate artifact 需要保留分析文件路径。Gate 选出 winner 后，pipeline 根据 `winner_commit` 找到对应 candidate，把它的 Markdown 复制到：

```text
<run_dir>/harness_change.md
```

只有通过 Gate 的 winner 分析进入 Reward Design。失败 candidate 的分析只作为调试产物。

### 4. Manifest 交接

在 `harness_gate_pairs.v1` 中增加：

```json
{
  "harness_change_analysis_path": "/absolute/path/to/harness_change.md"
}
```

程序继续负责填写 parent/candidate commit、changed files、轨迹路径和 verifier 结果。Markdown 只负责表达改动语义和预期行为。

## 设计约束

- 总结请求必须复用原始进化 session，避免重新解释完整上下文；
- 总结阶段必须是只读分析，不能产生新的 Harness patch；
- Markdown 中的文件路径和能力描述仍应由 pipeline 用 git diff 和 gate 结果校验；
- Reward Design 不应要求一定存在 `new_events`，prompt-only 改动使用行为触发条件即可；
- 没有 session id 或 follow-up 失败时，应记录失败原因，不要生成未经验证的分析文件；
- `harness_change.md` 是语义交接文件，manifest 仍然保留机器可验证的路径和结果字段。

## 验证方式

实现后至少验证：

1. OpenCode follow-up 使用与原进化相同的 session；
2. 总结请求没有修改工作区代码；
3. candidate artifact 能保存分析路径；
4. Gate winner 能对应到正确的 Markdown；
5. Reward Design 能读取 Markdown 并为 prompt-only 改动生成 rubric；
6. 没有分析文件时，Reward Design 给出明确错误，而不是静默生成空 rubric。
