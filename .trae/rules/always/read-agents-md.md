---
alwaysApply: true
---

# 全局核心红线入口

本项目的**全局架构红线与工程行为准则**位于仓库根目录的 `AGENTS.md`（跨工具通用规范，同时被 Antigravity / Cursor / Claude Code 读取）。

在任何任务规划、代码编写、重构或评审开始前，**必须先读取并无条件遵守 `AGENTS.md` 的全部内容**。它定义了：

- Agent 工程行为准则（契约与不变量先行、验证门禁）
- 分层边界与硬性不变量（分层由依赖决定而非目录名、单向依赖不可逆、`model/` 只接受纯张量且严禁认识 Cache 容器类型、物理池不承担输入组装）
- 零黑盒依赖、确定性与零精度妥协的底层不变量
- 提交前必跑的验证命令

子域深水区细则由 `.trae/rules/glob/` 下的 glob 规则按需自动挂载。**权威源是 `.agents/rules/` 下的同名文件**（供 Antigravity / Cursor 读取），`.trae/rules/glob/` 内的副本由 `.agents/scripts/sync_rules.sh` 在 pre-commit 时自动同步，必须与权威源逐字节一致。修改子域规则时只编辑 `.agents/rules/` 下的文件，不要直接改副本；两处内容不一致视为规则失效。

本文件所在的 `.trae/rules/always/` 存放 Trae 专属的常驻规则（`alwaysApply: true`），不参与上述同步。
