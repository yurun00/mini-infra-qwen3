---
alwaysApply: true
---

# 全局核心红线入口

全局架构红线与工程行为准则位于仓库根目录 `AGENTS.md`（跨工具通用，同时被 Antigravity / Cursor / Claude Code 读取），涵盖工程行为准则、分层边界与硬性不变量、零黑盒依赖与零精度妥协、提交前必跑的验证命令。任何任务规划、代码编写、重构或评审开始前，**必须先读取并遵守 `AGENTS.md` 的全部内容**。

子域细则由 `.trae/rules/glob/` 下的 glob 规则按需自动挂载。**权威源是 `.agents/rules/` 下的同名文件**（供 Antigravity / Cursor 读取），`.trae/rules/glob/` 内的副本由 `.agents/scripts/sync_rules.sh` 在 pre-commit 时自动同步，必须与权威源逐字节一致。修改子域规则时只编辑 `.agents/rules/` 下的文件，不要直接改副本；两处内容不一致视为规则失效。

本文件所在的 `.trae/rules/always/` 存放 Trae 专属的常驻规则（`alwaysApply: true`），不参与上述同步。
