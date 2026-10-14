# 逐条轨迹诊断与汇合

证据策略 `per_trace_then_aggregate` 把一次采样拆成两个阶段：先让诊断器一次只读取一条完整轨迹，再让汇合器读取所有结构化诊断和覆盖率报告。它是 Harness Evolution 的独立策略，不依赖 Self-Harness 源码。

在进化配置的 `evidence` 节中启用：

```yaml
evidence:
  method: per_trace_then_aggregate
  options:
    success_threshold: 1.0
    analyze_passed: true
    analyze_failed: true
    diagnosis_concurrency: 8
    diagnosis_timeout_sec: 240
    aggregate_enabled: true
    aggregate_timeout_sec: 600
    require_complete_failed_traces: true
    # 可选：把诊断缓存放到 GPFS2 或其他持久目录
    # result_root: /path/to/diagnosis-cache
```

这里的轨迹是使用 OpenCode 作为 harness，在 terminal 任务上采样得到的一次执行轨迹，并不限定来自 Terminal-Bench。每条轨迹的输入目录只包含同一次采样的多种表示和元数据：

- `instruction.md`：原始 terminal 任务目标、约束和交付要求；
- `trajectory.json`：按时间排序的 OpenCode 事件，包括 assistant 输出、工具调用、工具结果和步骤边界；
- `openai-messages.jsonl`：首选的 OpenAI 兼容增量对话，每行是初始消息或新增加的消息，不重复写入每次请求的完整累计上下文；
- `llm-context.jsonl`：provider 原生的累计请求快照，每行通常对应一次请求，会重复早期上下文，只在增量消息缺失或需要核对请求边界时使用；
- `openai-tools.json`：采样时模型实际可见的工具名称、说明和输入 schema；
- `openai-request.meta.json`、`openai-messages.meta.json`：请求边界、数量、来源和转换状态等辅助元数据；
- `verifier-summary.json`：reward、通过状态、执行阶段、退出码、超时或基础设施类别等脱敏结果摘要；
- `verifier-feedback.json`：verifier 执行产生的脱敏、限长 stdout/stderr，以及可解析时的测试数量、失败检查名称和 assertion 信息；
- `sanitized_result.json`：trial 标识、reward、状态和异常等脱敏元数据。

这些文件不能被当作多条独立证据。诊断器先用 `verifier-feedback.json` 确认具体失败检查，再从轨迹反向定位导致该失败的 agent 行为。输入中不复制 verifier 源码。

`harness_relevance` 衡量的是这条轨迹对“可复用 harness 干预”的支持强度，并不要求 harness 本身发生故障。可干预范围包括提示词、推理与验证策略、上下文管理、工具使用规则、恢复策略、停止条件以及工具运行行为：

- `none`：没有证据支持可复用干预，例如只有环境故障或纯任务事实；
- `low`：存在可能的通用干预方向，但因果关系或泛化性较弱；
- `medium`：观察到具体 agent 机制，并有合理的通用 harness 干预方式；单条轨迹可以达到此级别；
- `high`：证据直接而有力地表明某个与任务无关的 harness 策略、提示或工具行为影响了结果。

工具正常工作不构成 `none` 的理由。诸如只验证进程存活而没有验证任务语义、恢复策略反复失败、上下文遗漏和过早停止，均可能支持可复用的 harness 改进。跨任务是否反复出现仍由聚合阶段判断。

单轨迹的模式字段按最终结果收敛：失败轨迹最多保留 3 条对最终失败有因果贡献的 `failure_patterns`，其 `success_patterns` 默认为空，仅允许最多 1 条对重要子目标有明确贡献且适合跨任务比较的行为；成功轨迹最多保留 3 条对最终通过有因果贡献的 `success_patterns`，其 `failure_patterns` 默认为空，仅允许最多 1 条有明确证据、险些导致失败的行为。提示词要求按因果相关性从强到弱排列，控制器读取 `diagnosis.json` 后再次按结果强制执行这些数量上限。

单条诊断器必须把结果写到其输入目录中的 `diagnosis.json`。控制器只读取这个文件，不解析模型最终回复中的 JSON。验证后，控制器把诊断结果及运行元数据写入：

```text
<report_root>/diagnosis/<cache-key>/
├── coverage.json
├── index.json
├── per_trace/<trace-id>.json
├── raw/<trace-id>.log
├── aggregate.json
└── aggregate.md
```

`coverage.json` 记录通过、失败、基础设施不完整、轨迹缺失和诊断回退数量。奖励缺失但轨迹完整的样本会作为基础设施信号保留；失败且没有完整轨迹的样本会保留在 coverage 中并跳过诊断证据。只要本轮仍有至少一条可分析的完整轨迹，提案阶段就可以继续；只有完整轨迹数量为 0 时，`require_complete_failed_traces` 才会阻止提案，避免模型凭空补全证据。

汇合器只读取 `coverage.json`、`index.json` 和 `per_trace/` 下的结构化诊断，并将跨轨迹的共同模式、成功模式、任务特有模式和 Harness 改动假设写入 `aggregate.json` 与 `aggregate.md`。提案器会收到路径提示，应该先读覆盖率，再读汇总，最后按需核查单条轨迹文件。
