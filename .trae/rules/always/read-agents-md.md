---
alwaysApply: true
---

# 全局核心红线入口

全局架构红线与工程行为准则位于仓库根目录 `AGENTS.md`（被 Antigravity / Cursor / Claude Code / Trae 共同读取），任何任务规划、代码编写、重构或评审开始前，**必须先读取并遵守 `AGENTS.md` 的全部内容**。

子域细则由 `.trae/rules/glob/` 按需挂载，其**唯一权威源为 `.agents/rules/` 下的同名文件**；`.trae/rules/glob/` 副本由 `.agents/scripts/sync_rules.sh` 在 pre-commit 时自动同步，修改子域规则时只编辑 `.agents/rules/` 并运行同步脚本。`.trae/rules/always/` 存放 Trae 专属常驻规则，不参与同步。
