"""跨版本历史演进 Markdown 汇总表与自动化瓶颈诊断报告生成（纯文本报表，零 Matplotlib 依赖）。"""

from collections.abc import Sequence
from typing import Any

from mini_infra.bench.plotting.tidy import (
    extract_primary_case_dict,
    records_to_wide_df,
    resolve_record_label,
)
from mini_infra.bench.result import (
    LockstepBenchmarkRecord,
    SchedulerBenchmarkRecord,
    validate_benchmark_record,
)

RecordInput = LockstepBenchmarkRecord | SchedulerBenchmarkRecord | dict[str, Any]


def markdown_table(records: Sequence[RecordInput], base: RecordInput) -> str:
    """基于 Pandas 宽表与边界感知规则生成跨版本多规模演进 Markdown 汇总表。"""
    wide_df = records_to_wide_df(records)
    base_df = records_to_wide_df([base])
    default_b_row = base_df.iloc[0]
    base_by_scale = {str(r["scale"]): r for _, r in base_df.iterrows()}

    engines = set(wide_df["engine"].unique())
    is_pure_lockstep = engines == {"lockstep"}
    is_pure_scheduler = engines == {"scheduler"}

    lines = [
        f"基准版本：`{default_b_row['label']}` (`{default_b_row['commit']}`)，GPU：{default_b_row['gpu']}，"
        "数值为 median，括号内为相对同 Scale 基准的变化比（>1 表示更快/更高）。",
        "",
    ]

    if is_pure_lockstep:
        lines.extend(
            [
                "| 版本 | Scale | commit | total_requests | prompt | max_new_tokens | TTFT (ms) "
                "| TPOT (ms) | 吞吐 (tok/s) | 峰值显存 (MiB) | 活跃 KV (MiB) | batch slot 利用率 | 正确性 |",
                "| :--- | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
            ]
        )
    elif is_pure_scheduler:
        lines.extend(
            [
                "| 版本 | Scale | commit | total_requests | prompt | max_new_tokens | Makespan (ms) "
                "| TTFT P50 (ms) | ITL P95 (ms) | 吞吐 (tok/s) | 峰值显存 (MiB) | 活跃 KV (MiB) "
                "| KV physical block pool (MiB) | batch slot 利用率 | 正确性 |",
                "| :--- | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
            ]
        )
    else:
        lines.extend(
            [
                "| 版本 | Engine | Scale | commit | total_requests | prompt | max_new_tokens | TTFT (ms) "
                "| TPOT / ITL P50 (ms) | E2E (ms) | 吞吐 (tok/s) | 峰值显存 (MiB) | 活跃 KV (MiB) "
                "| batch slot 利用率 | 正确性 |",
                "| :--- | :-: | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
            ]
        )

    for _, row in wide_df.iterrows():
        sc = str(row.get("scale") or "medium")
        b_row = base_by_scale.get(sc, default_b_row)
        b_ttft = float(b_row.get("ttft_ms.median", 0.0))
        b_thr = float(b_row.get("throughput_tok_s.median", 0.0))
        b_e2e = float(b_row.get("e2e_wall_ms.median", 0.0))
        b_tpot = float(b_row.get("tpot_ms.median", 0.0)) if "tpot_ms.median" in b_row else 0.0
        b_mksp = float(b_row.get("makespan_ms.median", 0.0)) if "makespan_ms.median" in b_row else 0.0

        flag = {True: "✅", False: "❌", None: "—"}.get(row["correctness_passed"], "—")
        ttft = float(row.get("ttft_ms.median", 0.0))
        thr = float(row.get("throughput_tok_s.median", 0.0))
        mem = float(row.get("peak_allocated_mb.median", 0.0))
        live_kv = float(row.get("peak_live_kv_cache_mb.median", 0.0))
        pool = float(row.get("memory.kv_cache.peak_kv_cache_pool_mb", 0.0) or 0.0)
        pool_str = f"{pool:.1f}" if pool > 0 else "—"
        slot = float(row.get("slot_utilization.median", 1.0)) * 100.0

        s_ttft = f" ({b_ttft / ttft:.2f}x)" if b_ttft > 0 and ttft > 0 else ""
        s_thr = f" ({thr / b_thr:.2f}x)" if b_thr > 0 else ""

        if is_pure_lockstep:
            tpot = float(row.get("tpot_ms.median", 0.0))
            s_tpot = f" ({b_tpot / tpot:.2f}x)" if b_tpot > 0 and tpot > 0 else ""
            lines.append(
                f"| {row['label']} | `{sc}` | {row['commit']} | {row['num_total_requests']} | "
                f"{row['prompt_len_str']} | {row['max_new_tokens_str']} | {ttft:.2f}{s_ttft} | {tpot:.2f}{s_tpot} | "
                f"{thr:.1f}{s_thr} | {mem:.0f} | {live_kv:.1f} | {slot:.1f}% | {flag} |"
            )
        elif is_pure_scheduler:
            mksp = float(row.get("makespan_ms.median", 0.0))
            itl_p95 = float(row.get("itl_ms_p95.median", 0.0))
            s_mksp = f" ({b_mksp / mksp:.2f}x)" if b_mksp > 0 and mksp > 0 else ""
            lines.append(
                f"| {row['label']} | `{sc}` | {row['commit']} | {row['num_total_requests']} | "
                f"{row['prompt_len_str']} | {row['max_new_tokens_str']} | {mksp:.2f}{s_mksp} | {ttft:.2f}{s_ttft} | "
                f"{itl_p95:.2f} | {thr:.1f}{s_thr} | {mem:.0f} | {live_kv:.1f} | {pool_str} | {slot:.1f}% | {flag} |"
            )
        else:
            tpot = float(row.get("tpot_ms.median", 0.0))
            s_tpot = f" ({b_tpot / tpot:.2f}x)" if b_tpot > 0 and tpot > 0 else ""
            e2e = float(row.get("e2e_wall_ms.median", 0.0))
            s_e2e = f" ({b_e2e / e2e:.2f}x)" if b_e2e > 0 and e2e > 0 else ""
            lines.append(
                f"| {row['label']} | `{row['engine']}` | `{sc}` | {row['commit']} | {row['num_total_requests']} | "
                f"{row['prompt_len_str']} | {row['max_new_tokens_str']} | {ttft:.2f}{s_ttft} | "
                f"{tpot:.2f}{s_tpot} | {e2e:.2f}{s_e2e} | {thr:.1f}{s_thr} | {mem:.0f} | {live_kv:.1f} | "
                f"{slot:.1f}% | {flag} |"
            )

    return "\n".join(lines)


def generate_optimization_guidance(target_rec: RecordInput) -> str:
    """根据当前版本的代表性 Case profiling 数据，自动提炼诊断结论与下一步优化建议（支持 Lockstep 与 Scheduler）。"""
    validated_target = validate_benchmark_record(target_rec, require_single_case=True)
    w = extract_primary_case_dict(validated_target)

    lines = [
        "## 二、当前版本深度剖析与优化指引 (Profiling & Next Steps)",
        "",
        "![Profiling](./profiling_and_optimization.png)",
        "",
    ]

    if isinstance(validated_target, SchedulerBenchmarkRecord):
        sched = w.get("schedule") or {}
        pack = w.get("packing") or {}
        mem = w.get("memory") or {}
        slot_u = float((pack.get("slot_utilization") or {}).get("median", 0.0))
        cap_u = float((pack.get("running_capacity_utilization") or {}).get("median", 0.0))
        pad_e = float((pack.get("padding_efficiency") or {}).get("median", 0.0))
        q_p50 = float((sched.get("queue_ms_p50") or {}).get("median", 0.0))
        q_p95 = float((sched.get("queue_ms_p95") or {}).get("median", 0.0))
        ttft_p50 = float((sched.get("ttft_ms_p50") or {}).get("median", 0.0))
        ttft_p95 = float((sched.get("ttft_ms_p95") or {}).get("median", 0.0))
        itl_p50 = float((sched.get("itl_ms_p50") or {}).get("median", 0.0))
        itl_p95 = float((sched.get("itl_ms_p95") or {}).get("median", 0.0))
        itl_max = float((sched.get("itl_ms_max") or {}).get("median", 0.0))
        cache_frac = float((sched.get("cache_management_fraction") or {}).get("median", 0.0))
        kv_mb = float((mem.get("kv_cache") or {}).get("peak_live_kv_cache_mb") or 0.0)
        dyn_mb = float(mem.get("dynamic_peak_mb") or 0.0)
        frag_mb = float(mem.get("fragmentation_mb") or 0.0)

        lines.append(
            f"基于当前调度器版本 `{resolve_record_label(validated_target)}`（`requests={w['num_total_requests']}, "
            f"max_running={sched.get('max_running_requests', 4)}`）下的基准测量与轨迹剖析，诊断如下：\n"
        )
        lines.append("### 1. batch slot 与并发容量利用率诊断（调度效率）")
        lines.append(
            f"- 🔹 **执行 batch slot 有效率（Slot Utilization）为 {slot_u * 100:.1f}%**，"
            f"**并发容量利用率（Running Capacity Utilization）为 {cap_u * 100:.1f}%**，"
            f"**Prompt Padding 有效率为 {pad_e * 100:.1f}%**。"
        )
        if pad_e < 0.75:
            lines.append(
                "- 🟡 **变长 Prompt 左填充浪费明显**：建议后续演进引入 "
                "**FlashAttention-Varlen / Ragged Prefill** 消除无效 Padding 计算。"
            )
        lines.append("")

        lines.append("### 2. 请求排队与尾延迟诊断（P50 vs P95）")
        lines.append(
            f"- 🔹 **排队延迟**：P50 = `{q_p50:.1f} ms`，P95 = `{q_p95:.1f} ms`；"
            f"**TTFT**：P50 = `{ttft_p50:.1f} ms`，P95 = `{ttft_p95:.1f} ms`。"
        )
        lines.append(
            f"- 🔹 **Decode 步间延迟（ITL）**：P50 = `{itl_p50:.2f} ms`，"
            f"P95 = `{itl_p95:.2f} ms`，Max = `{itl_max:.2f} ms`。"
        )
        # 迭代调度步数类指标：早于 token-budget 的历史记录无此字段，缺失时跳过
        pf_streak = sched.get("max_consecutive_prefill_steps")
        if pf_streak is not None:
            num_prefill = float((sched.get("num_prefill_steps") or {}).get("median", 0.0))
            num_decode = float((sched.get("num_decode_steps") or {}).get("median", 0.0))
            num_oversize = float((sched.get("num_oversize_prefill_steps") or {}).get("median", 0.0))
            lines.append(
                f"- 🔹 **迭代调度步数**：Prefill `{num_prefill:.0f}` 步 / Decode `{num_decode:.0f}` 步，"
                f"最长连续 Prefill `{float(pf_streak.get('median', 0.0)):.0f}` 步，"
                f"oversize prefill `{num_oversize:.0f}` 次。"
            )
        if itl_max > itl_p50 * 1.4:
            is_token_budget = "token-budget" in str(w.get("engine_config", ""))
            next_step = (
                "**Chunked Prefill**，将长 Prompt 拆分为小块与 Decode 交织执行"
                "（软预算下 oversize prefill 仍会整段阻塞）。"
                if is_token_budget
                else "**token-budget 调度策略**（`--scheduling-policy token-budget`）限制单次 Prefill 规模并启用 "
                "decode protection，再演进到 **Chunked Prefill**。"
            )
            lines.append(
                "- 🟡 **观测到新请求 Prefill 插队导致的 Decode 尾延迟毛刺（ITL Max > 1.4x P50）**："
                f"建议引入 {next_step}"
            )
        lines.append("")

        lines.append("### 3. 显存足迹与 KV Cache 管理诊断")
        lines.append(
            f"- 🔹 **峰值活跃 KV Cache 为 {kv_mb:.1f} MiB**，动态显存峰值增量为 **{dyn_mb:.1f} MiB**，"
            f"Allocator 预留碎片为 **{frag_mb:.1f} MiB**，Cache 搬移开销占比为 **{cache_frac * 100:.2f}%**。"
        )
        is_paged = (
            "paged" in str(w.get("engine_config", "")).lower()
            or "paged" in str(getattr(validated_target, "release_feature", "")).lower()
        )
        if is_paged:
            lines.append(
                "- 👉 **下一步核心演进方向**：引入 **Prefix Cache（前缀缓存与物理块复用）** 与 "
                "**Chunked Prefill / 融合 Paged Attention Kernel**，"
                "按物理 Block 按需复用长 Prompt 并消除 Prefill 阶段的高峰值激活与跨层 gather 拷贝开销。"
            )
        else:
            lines.append(
                "- 👉 **下一步核心演进方向**：引入 **Paged KV Cache（分页键值缓存）** 与 **Prefix Cache**，"
                "按固定大小物理 Block 按需分配与复用，彻底消除按最长序列预分配的显存碎片与稠密 batch slot 搬移。"
            )
        lines.append("")
        return "\n".join(lines)

    ob = w.get("op_latency_breakdown")
    if not (isinstance(ob, dict) and ob.get("time_share_by_module")):
        lines.append(
            f"> 提示：当前版本 `{resolve_record_label(validated_target)}` 中未包含详细的算子剖析采样数据。"
            "使用 `mini-infra bench lockstep case --profiling-mode dedicated-pass` 重跑可获取完整算子诊断。"
        )
        return "\n".join(lines)

    by_mod = ob.get("time_share_by_module", {})
    by_cat = ob.get("time_share_by_kernel_category", {})
    eff = w.get("efficiency") or {}
    mem = w.get("memory") or {}
    busy = (w.get("gpu_time_ratio") or {}).get("median", 1.0)

    prompt_desc = f"prompt={w['prompt_len']}"
    lines.append(
        f"基于当前版本 `{resolve_record_label(validated_target)}`（`{prompt_desc}`）下的"
        "基准测量与 profiling 结果，诊断如下：\n"
    )

    attn_share = by_mod.get("attention_block", 0.0)
    norm_share = by_mod.get("norm", 0.0)
    mlp_share = by_mod.get("mlp_block", 0.0)

    lines.append("### 1. 架构模块瓶颈诊断（改哪里？）")
    if attn_share >= 0.25:
        lines.append(
            f"- 🔴 **Attention 模块耗时占比最高（{attn_share * 100:.1f}%）**：\n"
            "  - 👉 **优先优化行动**：引入 KV Cache / FlashAttention 算子融合。"
        )
    if norm_share > 0.08:
        lines.append(f"- 🟡 **RMSNorm 调用开销显著（耗时占比 {norm_share * 100:.1f}%）**：建议实施 Fused RMSNorm。")
    if mlp_share > 0.15:
        lines.append(
            f"- 🟡 **MLP Block 占比偏高（耗时占比 {mlp_share * 100:.1f}%）**：建议针对 Gate/Up 实施 SwiGLU 融合。"
        )
    if not (attn_share >= 0.25 or norm_share > 0.08 or mlp_share > 0.15):
        top_mod = max(by_mod.items(), key=lambda kv: kv[1])[0] if by_mod else "unknown"
        lines.append(f"- 🔹 当前耗时最高的架构模块为 `{top_mod}`（占比 {by_mod.get(top_mod, 0) * 100:.1f}%）。")
    lines.append("")

    gemm_share = by_cat.get("gemm", 0.0)
    elem_mem_share = by_cat.get("elementwise", 0.0) + by_cat.get("memory", 0.0)
    lines.append("### 2. 算子类型与访存诊断（怎么改？）")
    lines.append(
        f"- 🔹 **GEMM 矩阵乘占 {gemm_share * 100:.1f}%**，**Elementwise + Memory 搬运占 {elem_mem_share * 100:.1f}%**。"
    )
    lines.append("")

    mbu = eff.get("decode_mbu")
    lines.append("### 3. 硬件物理天花板差距")
    if mbu is not None:
        lines.append(f"- 🔹 **Decode MBU（显存带宽利用率）为 {mbu * 100:.1f}%**。")
    if busy < 0.90:
        lines.append(f"- 🔹 **GPU 忙碌比例为 {busy * 100:.1f}%**：建议引入 CUDA Graph 消除 CPU 发射开销。")
    else:
        lines.append(f"- 🔹 **GPU 忙碌比例达到 {busy * 100:.1f}%**：瓶颈主要在 GPU 端计算与访存。")
    lines.append("")

    frag = mem.get("fragmentation_mb", 0.0)
    dyn = mem.get("dynamic_peak_mb", 0.0)
    lines.append("### 4. 显存足迹与碎片诊断")
    lines.append(f"- 🔹 纯动态增量为 **{dyn:.1f} MiB**，Allocator 碎片与预留为 **{frag:.1f} MiB**。")
    lines.append("")

    top_kernels = ob.get("top_kernels", [])
    if top_kernels:
        lines.append("### 5. 当前 Top 8 热点 Kernel 明细")
        lines.append("| 序号 | Kernel 名称 | 类别 | CUDA 耗时 (ms) | 耗时占比 | 调用次数 |")
        lines.append("| :-: | :--- | :-: | :-: | :-: | :-: |")
        for rank, k in enumerate(top_kernels[:8], 1):
            k_name = k["name"][:40]
            lines.append(
                f"| {rank} | `{k_name}` | {k['category']} | {k['cuda_ms']:.2f} | "
                f"{k['share'] * 100:.1f}% | {k['calls']} |"
            )
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "generate_optimization_guidance",
    "markdown_table",
]
