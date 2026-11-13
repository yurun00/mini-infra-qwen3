"""bench.result 与 bench.plotting (Pydantic + Pandas + Seaborn) 单元测试。"""

import json
from pathlib import Path

import pytest

from mini_infra.bench.plotting.compare import (
    generate_comparison_table,
    plot_mode_comparison,
)
from mini_infra.bench.plotting.history import (
    load_results,
    plot_evolution_and_scaling,
    plot_profiling_and_optimization,
)
from mini_infra.bench.plotting.report import (
    generate_optimization_guidance,
    markdown_table,
)
from mini_infra.bench.plotting.tidy import (
    LOCKSTEP_ONLY_PLOT_SPECS,
    SCHEDULER_ONLY_PLOT_SPECS,
    SHARED_PLOT_SPECS,
    records_to_wide_df,
    resolve_plot_specs,
    resolve_record_label,
    wide_to_tidy_df,
)
from mini_infra.bench.result import (
    LockstepBenchmarkRecord,
    SchedulerBenchmarkRecord,
    load_benchmark_record,
    validate_benchmark_record,
)


def _make_single_case_scheduler_dict(
    tag: str, engine_config: str, makespan: float, ttft_p50: float, workload_id: str = "custom"
) -> dict:
    """构造单 Case 的 SchedulerBenchmarkRecord 字典供测试使用。"""

    def stat(v: float) -> dict[str, float]:
        return {
            "median": v,
            "mean": v,
            "std": 1.0,
            "min": v - 1.0,
            "p90": v + 0.5,
            "p95": v + 1.0,
            "p99": v + 1.5,
            "max": v + 2.0,
        }

    return {
        "schema_version": 1,
        "engine": "scheduler",
        "tag": tag,
        "release_feature": f"Scheduler {engine_config}",
        "engine_config": engine_config,
        "timestamp": "2099-12-31T23:59:59+0000",
        "git": {"commit": f"commit_{tag}", "branch": "main", "dirty": False, "message": "test"},
        "env": {"gpu": "NVIDIA L4", "torch": "2.14.1"},
        "peak_specs": {"bandwidth_gbs": 300.0, "tflops": 121.0},
        "model_cost": None,
        "config": {"warmup": 1, "repeats": 3, "model": "Qwen/Qwen3-0.6B", "dtype": "torch.bfloat16"},
        "correctness": {"passed": True},
        "suite": None,
        "cases": [
            {
                "workload_id": workload_id,
                "engine_config": engine_config,
                "num_total_requests": 8,
                "prompt_len": 128,
                "max_new_tokens": [8, 32, 8, 32, 8, 32, 8, 32],
                "repeats": 3,
                "warmup": 1,
                "max_running_requests": 4,
                "arrival_times_ms": 0.0,
                "e2e_wall_ms": stat(makespan),
                "e2e_model_ms": stat(makespan * 0.98),
                "throughput_tok_s": stat(160000.0 / makespan),
                "throughput_req_s": stat(8000.0 / makespan),
                "gpu_time_ratio": stat(0.98),
                "memory": {
                    "weights_mb": 1140.0,
                    "prefill_allocated_mb": 1200.0,
                    "peak_allocated_mb": 1260.0,
                    "dynamic_peak_mb": 120.0,
                    "peak_reserved_mb": 1300.0,
                    "fragmentation_mb": 40.0,
                    "kv_cache": {"peak_live_kv_cache_mb": 45.0},
                },
                "packing": {
                    "padding_efficiency": stat(1.0),
                    "slot_utilization": stat(0.85),
                    "running_capacity_utilization": stat(0.92),
                },
                "efficiency": {"prefill_mfu": 0.35, "decode_mfu": 0.05, "decode_mbu": 0.65},
                "schedule": {
                    "request_count": 8,
                    "max_running_requests": 4,
                    "makespan_ms": stat(makespan),
                    "busy_ms": stat(makespan * 0.99),
                    "cache_management_ms": stat(1.5),
                    "cache_management_fraction": stat(0.001),
                    "queue_ms_p50": stat(0.0),
                    "queue_ms_p95": stat(makespan * 0.3),
                    "ttft_ms_p50": stat(ttft_p50),
                    "ttft_ms_p95": stat(ttft_p50 * 4.0),
                    "end_to_end_ms_p50": stat(makespan * 0.5),
                    "end_to_end_ms_p95": stat(makespan),
                    "itl_ms_p50": stat(42.0),
                    "itl_ms_p95": stat(45.0),
                    "itl_ms_max": stat(50.0),
                },
                "repeat_samples": [],
                "sample_event_trace": [],
            }
        ],
    }


def test_existing_benchmark_json_files_validate():
    """验证仓库内所有基准 JSON 文件均通过 Pydantic Schema 校验，且 history 满足单 Case 约束。"""
    lockstep_history_dir = Path("benchmarks/results/history/lockstep")
    lockstep_files = sorted(lockstep_history_dir.glob("*.json"))
    assert len(lockstep_files) >= 4
    for p in lockstep_files:
        rec = load_benchmark_record(p, require_single_case=True)
        assert isinstance(rec, LockstepBenchmarkRecord)
        assert 1 <= len(rec.cases) <= 3

    scheduler_history_dir = Path("benchmarks/results/history/scheduler")
    for p in sorted(scheduler_history_dir.glob("*.json")):
        rec = load_benchmark_record(p, require_single_case=True)
        assert isinstance(rec, SchedulerBenchmarkRecord)
        assert 1 <= len(rec.cases) <= 3

    ref = load_benchmark_record(Path("benchmarks/golden/reference.json"), require_single_case=True)
    assert isinstance(ref, LockstepBenchmarkRecord | SchedulerBenchmarkRecord)

    eval_rec = load_benchmark_record(
        Path("benchmarks/results/suites/lockstep/serial-vs-fixed-batch-throughput/eval.json")
    )
    assert isinstance(eval_rec, LockstepBenchmarkRecord)
    assert len(eval_rec.cases) > 1
    with pytest.raises(ValueError, match="single-case"):
        load_benchmark_record(
            Path("benchmarks/results/suites/lockstep/serial-vs-fixed-batch-throughput/eval.json"),
            require_single_case=True,
        )


def test_resolve_plot_specs_and_tidy_df_boundary_awareness():
    """验证 resolve_plot_specs 在纯 Lockstep、纯 Scheduler 与跨越边界时的动态图表规格解析与字段对齐。"""
    lockstep_rec = load_benchmark_record(
        Path("benchmarks/results/history/lockstep/1c21386_v0.3.0.json"), require_single_case=True
    )
    sched_rec_1 = validate_benchmark_record(
        _make_single_case_scheduler_dict("v0.4.0", "kv-cache+fixed-batching", 2800.0, 46.0, workload_id="fixed"),
        require_single_case=True,
    )
    sched_rec_2 = validate_benchmark_record(
        _make_single_case_scheduler_dict(
            "v0.5.0", "kv-cache+continuous-batching", 2100.0, 44.0, workload_id="continuous"
        ),
        require_single_case=True,
    )
    assert isinstance(sched_rec_1, SchedulerBenchmarkRecord)

    # 1. 纯 Lockstep：包含 SHARED + LOCKSTEP_ONLY (含 tpot_ms)，默认均为 line
    lockstep_specs = resolve_plot_specs([lockstep_rec])
    assert lockstep_specs == [*SHARED_PLOT_SPECS, *LOCKSTEP_ONLY_PLOT_SPECS]
    assert any(s.metric_key == "tpot_ms" for s in lockstep_specs)
    assert all(s.chart_kind == "line" for s in lockstep_specs)

    # 2. 纯 Scheduler：包含 SHARED + SCHEDULER_ONLY (含对齐后的 tpot_ms，以及 itl_ms_p95 / makespan_ms 等)
    sched_specs = resolve_plot_specs([sched_rec_1, sched_rec_2])
    assert sched_specs == [*SHARED_PLOT_SPECS, *SCHEDULER_ONLY_PLOT_SPECS]
    assert any(s.metric_key == "tpot_ms" for s in sched_specs)
    assert any(s.metric_key == "itl_ms_p95" for s in sched_specs)
    assert any(s.metric_key == "makespan_ms" for s in sched_specs)

    # 3. 跨越 Lockstep <-> Scheduler 边界：保留 SHARED_PLOT_SPECS（ttft_ms <-> ttft_ms_p50，tpot_ms <-> itl_ms_p50）
    cross_specs = resolve_plot_specs([lockstep_rec, sched_rec_2])
    assert cross_specs == list(SHARED_PLOT_SPECS)
    assert any(s.metric_key == "tpot_ms" for s in cross_specs)
    assert not any(s.metric_key == "itl_ms_p95" for s in cross_specs)

    wide_df = records_to_wide_df([lockstep_rec, sched_rec_2])
    assert len(wide_df) == len(lockstep_rec.cases) + len(sched_rec_2.cases)
    # 验证 Lockstep 的 ttft_ms 与 Scheduler 的 schedule.ttft_ms_p50 对齐在同一列 ttft_ms.median
    assert wide_df["ttft_ms.median"].notna().all()
    assert wide_df.loc[wide_df["engine"] == "scheduler", "ttft_ms.median"].iloc[0] == pytest.approx(44.0)
    # 验证 Lockstep 的 tpot_ms 与 Scheduler 的 schedule.itl_ms_p50 对齐在同一列 tpot_ms.median
    assert wide_df["tpot_ms.median"].notna().all()
    assert wide_df.loc[wide_df["engine"] == "scheduler", "tpot_ms.median"].iloc[0] == pytest.approx(42.0)
    # 验证 slot_utilization 对齐
    assert wide_df["slot_utilization.median"].notna().all()

    tidy_df = wide_to_tidy_df(wide_df, cross_specs)
    assert set(tidy_df["metric_key"].unique()) == {s.metric_key for s in SHARED_PLOT_SPECS}


def test_plot_compare_and_history_end_to_end(tmp_path: Path):
    """验证 compare 与 history 在单 Case 约束下可生成 Markdown 表格与 Seaborn PNG 图表（含跨边界场景）。"""
    sched_fixed_path = tmp_path / "sched_fixed.json"
    sched_cont_path = tmp_path / "sched_cont.json"
    sched_fixed_path.write_text(
        json.dumps(
            _make_single_case_scheduler_dict(
                "v0.4.0-fixed", "kv-cache+fixed-batching", 2800.0, 46.0, workload_id="fixed"
            )
        ),
        encoding="utf-8",
    )
    sched_cont_path.write_text(
        json.dumps(
            _make_single_case_scheduler_dict(
                "v0.4.0-cont", "kv-cache+continuous-batching", 2100.0, 44.0, workload_id="continuous"
            )
        ),
        encoding="utf-8",
    )

    # 1. 纯 Scheduler 双文件单 Case 对比
    rec_fixed = load_benchmark_record(sched_fixed_path, require_single_case=True)
    rec_cont = load_benchmark_record(sched_cont_path, require_single_case=True)
    lbl_fixed = resolve_record_label(rec_fixed)
    lbl_cont = resolve_record_label(rec_cont)
    md_text = generate_comparison_table(rec_fixed, rec_cont, lbl_fixed, lbl_cont)
    assert "ITL P95 Latency (ms/token)" in md_text
    assert "Schedule Makespan (ms)" in md_text
    out_compare_png = tmp_path / "compare_sched.png"
    plot_mode_comparison(rec_fixed, rec_cont, lbl_fixed, lbl_cont, out_compare_png)
    assert out_compare_png.exists()

    # 2. 跨边界 (Lockstep vs Scheduler) 双文件单 Case 对比
    lockstep_path = Path("benchmarks/results/history/lockstep/1c21386_v0.3.0.json")
    rec_lock = load_benchmark_record(lockstep_path, require_single_case=True)
    lbl_lock = resolve_record_label(rec_lock)
    cross_md = generate_comparison_table(rec_lock, rec_cont, lbl_lock, lbl_cont)
    assert "TTFT / Prefill Latency (ms)" in cross_md
    assert "TPOT / ITL P50 Latency (ms/token)" in cross_md
    assert "ITL P95 Latency (ms/token)" not in cross_md
    out_cross_png = tmp_path / "cross_compare.png"
    plot_mode_comparison(rec_lock, rec_cont, lbl_lock, lbl_cont, out_cross_png)
    assert out_cross_png.exists()

    # 3. History 跨边界演进图与表格
    hist_dir = tmp_path / "history_input"
    hist_dir.mkdir()
    for p in sorted(Path("benchmarks/results/history/lockstep").glob("*.json")):
        (hist_dir / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    (hist_dir / "zzzz_v0.4.0-cont.json").write_text(sched_cont_path.read_text(encoding="utf-8"), encoding="utf-8")

    records = load_results(hist_dir)
    assert len(records) >= 5
    hist_md = markdown_table(records, records[0])
    assert "v0.4.0-cont" in hist_md
    assert "TPOT / ITL P50 (ms)" in hist_md
    guidance = generate_optimization_guidance(records[-1])
    assert "v0.4.0-cont" in guidance

    evo_png = tmp_path / "evolution.png"
    prof_png = tmp_path / "profiling.png"
    plot_evolution_and_scaling(records, records[0], evo_png)
    plot_profiling_and_optimization(records, records[-1], prof_png)
    assert evo_png.exists()
    assert prof_png.exists()

    # 4. 纯 Scheduler 历史剖析图（验证 memory.weights_mb 为空时自动回退 baseline_allocated_mb）
    sched_records = load_results(Path("benchmarks/results/history/scheduler"))
    sched_prof_png = tmp_path / "scheduler_profiling.png"
    plot_profiling_and_optimization(sched_records, sched_records[-1], sched_prof_png)
    assert sched_prof_png.exists()


def test_scheduler_benchmark_record_with_comparisons_and_workload_id(tmp_path: Path):
    """验证含 comparisons 与 workload_id 的 Scheduler 记录在 Pydantic 与 Pandas 宽表中的往返完整性。"""
    data = _make_single_case_scheduler_dict(
        "v0.5.0-paged",
        "paged-kv-cache+continuous-batching",
        1800.0,
        40.0,
        workload_id="length_ratio_2048",
    )
    data["cases"][0]["repeat_samples"] = [{"makespan_ms": 1800.0, "peak_allocated_mb": 1234.5}]
    data["comparisons"] = [
        {
            "baseline_key": {
                "workload_id": "length_ratio_2048",
                "caching_method": "dense-kv-cache",
                "batching_method": "continuous-batching",
                "flags": [],
                "max_running_requests": 8,
            },
            "candidate_key": {
                "workload_id": "length_ratio_2048",
                "caching_method": "paged-kv-cache",
                "batching_method": "continuous-batching",
                "flags": [],
                "max_running_requests": 8,
                "block_size": 16,
                "num_blocks": 100,
            },
            "metric_path": "schedule_metrics.makespan_ms",
            "metric_name": "makespan",
            "kind": "speedup",
            "baseline_value": 2100.0,
            "candidate_value": 1800.0,
            "ratio": 0.8571,
            "relative_delta": -0.1429,
            "speedup": 1.1667,
            "formatted": "speedup 1.17x",
        }
    ]

    json_path = tmp_path / "paged_record.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    rec = load_benchmark_record(json_path, require_single_case=True)
    assert isinstance(rec, SchedulerBenchmarkRecord)
    assert len(rec.cases) == 1
    assert rec.cases[0].workload_id == "length_ratio_2048"
    assert len(rec.cases[0].repeat_samples) == 1
    assert rec.cases[0].repeat_samples[0]["peak_allocated_mb"] == 1234.5
    assert len(rec.comparisons) == 1
    comp = rec.comparisons[0]
    assert comp.baseline_key.workload_id == "length_ratio_2048"
    assert comp.speedup == 1.1667
    assert comp.metric_name == "makespan"
    assert comp.candidate_key.block_size == 16

    wide_df = records_to_wide_df([rec])
    assert "workload_id" in wide_df.columns
    assert wide_df["workload_id"].iloc[0] == "length_ratio_2048"


def test_gpu_info_includes_host_environment_fields(monkeypatch: pytest.MonkeyPatch):
    """验证 gpu_info 返回宿主机 CPU、系统内存、OS 平台与 CUDA 分配器配置。"""
    from types import SimpleNamespace

    import torch

    from mini_infra.bench.env import gpu_info

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device=0: SimpleNamespace(name="Fake GPU", total_memory=24 * 1024**3),
    )
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    info = gpu_info(0)
    assert info["gpu"] == "Fake GPU"
    assert info["total_memory_mb"] == 24 * 1024.0
    assert isinstance(info["cpu"], str) and len(info["cpu"]) > 0
    assert info["system_memory_mb"] is None or info["system_memory_mb"] > 0
    assert isinstance(info["platform"], str) and len(info["platform"]) > 0
    assert info["cuda_alloc_conf"] == "expandable_segments:True"
