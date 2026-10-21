# 待办与已知问题

记录 MiMo 进化循环（`evoloop/harnessRL-mimo32-gate100-loop-20261014-v4`，2026-10-14 启动）过程中发现、还没处理完的事项。处理完一项就把它移到最后的“已完成”。

## 一、代码（下一次新运行前要完成）

| # | 事项 | 现状 | 怎么做 |
|---|---|---|---|
| 1 | **MiMo 题加线程限制** | 代码已改、单元测试通过，真实沙箱端到端复测通过（001226 verifier 31 秒跑完；agent 正常，`agent_rc=0`、轨迹完整）；未提交 git，下一次新运行启用 | 沙箱里 `nproc` 显示宿主机 ~320 核，实际配额 2 核，数值库开太多线程会卡死（`format-code-task-001226` 的 verifier 卡 1800 秒超时）。已按筛选脚本给 agent、准备步骤和 verifier 都加 `OMP_NUM_THREADS=2` 等限制，并上传 Python/Node 的 CPU 数钩子（`evaluation/backends/mimo.py`）。对比测试：不加限制卡死，加了 33 秒跑完 |
| 2 | **MiMo 的 verifier 摘要标错** | 未修 | `verifier-summary.json` 只认 TB 的回执格式，MiMo 题被写成 `verifier_receipt_valid: false`、`verifier_rc: 0`，诊断会误以为 verifier 没跑（如 001980）。要让摘要解析 `__MIMO_VERIFIER_RC__=N`，并补测试 |
| 3 | **起点检出检查** | 代码已改、测试通过；未提交 git | 全新启动时 `initial.harness_path` 必须检出在 `base_commit`，否则报错退出（v3 因此把 `515b6b3` 当成 `1802c9e` 编译）。`loop/two_sided.py::_require_checkout_at` |
| 4 | 提交并推送 | 未做 | 1、3（以及 2 修好后）提交到本地 git，推送到 `github.com/wxm888888/HarnessEvo` |
| 5 | `model.diff` 进诊断 | 未做 | MiMo code 题的 agent 改动只存在采样目录，诊断看不到；需要改诊断 prompt 和对应的快照测试 |

## 二、环境 / 数据

| # | 事项 | 现状 | 怎么做 |
|---|---|---|---|
| 6 | **verifier 环境偶发失败** | 观察中 | 例：001980 的测试数据库还没起来测试就开始跑（`database system is starting up`），被算成 agent 失败。先统计每步有多少；多的话考虑 verifier 在环境未就绪时自动重试一次 |
| 7 | 沙箱服务不稳定 | 观察中 | 傍晚频繁出现连接超时、`409 sandbox route not active`、删除未确认。目前都靠自动重试兜住；重试用完的门控题记 0 分（不算不完整） |
| 8 | **gpfs1 空间快满** | 需要你清理 | `/mnt/shared-storage-user/yuanjiarui-p` 100G 只剩约 0.4G。Codex 凭证、代码仓库都在这块盘上 |
| 9 | 依赖缓存的位置 | 已规避 | `link_shared_node_modules` 会把整个 node_modules 复制到 `<工作区>/../../.node_modules_cache`（约 11G）。起点工作区必须放在 gpfs2，放 gpfs1 会写满配额 |
| 10 | 版本差异 | 记录 | 筛选用的是官方 OpenCode **1.18.21** 二进制，我们的进化起点是 **v1.17.9 + 30 行请求记录补丁**（`1802c9e`），两者通过率不能直接对比 |

## 三、实验

| # | 事项 | 现状 |
|---|---|---|
| 11 | v4 跑完后，用最终版本在 TB2.1（89 题）上做一次最终评估 | 循环里没有自动步骤，需要手动跑 |
| 12 | 文档更新：`docs/evolution.zh.md` 补线程限制、起点检查、门控 100 题×1 次、v4 配置 | 未做 |
| 13 | 清理不再需要的运行目录：`…-20261014`（v1，pyenv 问题）、`-v2`（起点 515b6b3）、`-v3`（二进制版本错误）、各个探针目录 | 等你确认后再删 |

## 已完成

- MiMo 准备和 verifier 改用 `HOME=/root`（pyenv 需要 `/root/.bashrc`）。
- `hevo run` 在 local 放行模式下自动设置记账目录和 run id；示例配置补齐放行、代理、二进制上传设置。
- 固定 100 题 MiMo 门控集 + 1609 题进化题库（`tasks/mimo_catalog.py --gate-size`）。
- 代码推送到私有仓库 `wxm888888/HarnessEvo`（部署密钥，只对该仓库生效）。
