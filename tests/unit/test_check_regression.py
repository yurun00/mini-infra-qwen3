"""回归判定逻辑的单元测试（纯 CPU，不依赖 GPU）。"""

from mini_infra.bench.check_regression import compare


def make_record(ttft, tpot, std=0.1):
    def stat(median):
        return {"median": median, "std": std}

    return {
        "cases": [
            {
                "batch_size": 1,
                "prompt_len": 32,
                "max_new_tokens": 32,
                "ttft_ms": stat(ttft),
                "tpot_ms": stat(tpot),
            }
        ]
    }


def test_no_change_passes():
    regressed, _ = compare(make_record(40, 37), make_record(40, 37), 0.05)
    assert regressed is False


def test_regression_over_threshold_detected():
    regressed, lines = compare(make_record(40, 37 * 1.10), make_record(40, 37), 0.05)
    assert regressed is True
    assert any("回归" in line for line in lines)


def test_improvement_not_flagged():
    regressed, _ = compare(make_record(20, 10), make_record(40, 37), 0.05)
    assert regressed is False


def test_change_within_noise_not_flagged():
    # 劣化 8% 超过阈值，但 std 很大，变化量小于 3 倍 std，视为噪声
    regressed, _ = compare(make_record(40, 37 * 1.08, std=2.0), make_record(40, 37, std=2.0), 0.05)
    assert regressed is False


def test_no_common_case_returns_none():
    candidate = make_record(40, 37)
    candidate["cases"][0]["prompt_len"] = 64
    regressed, _ = compare(candidate, make_record(40, 37), 0.05)
    assert regressed is None
