#!/usr/bin/env bash
# 把 .agents/rules/（权威源，供 Antigravity / Cursor 读取）同步到 .trae/rules/glob/（Trae 读取的副本）。
# .trae/rules/always/ 存放 Trae 专属的常驻规则，目录结构本身即排除语义，不参与同步。
# 退出码非 0 表示发生了同步或发现了孤儿副本，pre-commit 会因此中断提交，需 git add 后重试。
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SRC=.agents/rules
DST=.trae/rules/glob

mkdir -p "$DST"

changed=0
for src in "$SRC"/*.md; do
    [ -e "$src" ] || continue
    dst="$DST/$(basename "$src")"
    if ! cmp -s "$src" "$dst"; then
        cp -f "$src" "$dst"
        echo "synced: $src -> $dst"
        changed=1
    fi
done

# 权威源已删除但副本还在，说明规则改名或废弃后忘了清理。
for dst in "$DST"/*.md; do
    [ -e "$dst" ] || continue
    name=$(basename "$dst")
    if [ ! -e "$SRC/$name" ]; then
        echo "orphan: $dst 的权威源 $SRC/$name 不存在，请删除该副本" >&2
        exit 1
    fi
done

if [ "$changed" -ne 0 ]; then
    echo "$DST 已重新同步，请 git add 后重新提交" >&2
    exit 1
fi
