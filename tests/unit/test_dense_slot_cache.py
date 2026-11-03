"""DenseSlotCache 的单元测试（CPU，tiny 模型）：入场写槽位、左侧扩容、Decode 输入与释放。"""

import pytest
import torch

from mini_infra.core.request import RequestSpec, RequestState, RequestStatus
from mini_infra.engine.preprocessing import left_pad_sequences
from mini_infra.memory import DenseSlotCache, KVCache
from mini_infra.verification.smoke import build_tiny_model

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def model():
    return build_tiny_model()


def prefill(model, prompts: list[list[int]]) -> KVCache:
    """对一批 Prompt 做左 Padding Prefill，返回 Cache。"""
    input_ids, mask, positions, _ = left_pad_sequences(prompts, 0, CPU)
    with torch.inference_mode():
        _, present = model(input_ids, attention_mask=mask, position_ids=positions)
    return KVCache(present)


def admitted_state(index: int, prompt: list[int], slot: int) -> RequestState:
    state = RequestState(RequestSpec(index, tuple(prompt), 4))
    state.status = RequestStatus.RUNNING
    state.slot_idx = slot
    state.cache_length = len(prompt)
    return state


def assert_slot_matches_single_request(
    dense_slot_cache: DenseSlotCache, model, state: RequestState, prompt: list[int]
) -> None:
    """槽位中右对齐的有效历史必须与该请求单独 Prefill 的 Cache 完全一致，且左侧空位被 mask。"""
    reference = prefill(model, [prompt])
    length, total = len(prompt), dense_slot_cache.length
    assert dense_slot_cache.history_mask[state.slot_idx].tolist() == [False] * (total - length) + [True] * length
    for layer, (ref_key, ref_value) in enumerate(reference):
        key, value = dense_slot_cache.kv_cache[layer]
        torch.testing.assert_close(key[state.slot_idx, :, total - length :, :], ref_key[0])
        torch.testing.assert_close(value[state.slot_idx, :, total - length :, :], ref_value[0])


def test_absorb_prefill_allocates_and_writes_right_aligned_slots(model):
    prompts = [[3, 5, 7, 9], [11, 13]]
    states = [admitted_state(0, prompts[0], slot=1), admitted_state(1, prompts[1], slot=3)]
    dense_slot_cache = DenseSlotCache(capacity=4, device=CPU)
    assert dense_slot_cache.kv_cache is None and dense_slot_cache.length == 0 and dense_slot_cache.total_bytes == 0

    elapsed_ms = dense_slot_cache.absorb_prefill(states, prefill(model, prompts))

    assert elapsed_ms >= 0
    assert dense_slot_cache.length == 4
    assert dense_slot_cache.kv_cache.length == 4
    config = model.config
    assert tuple(dense_slot_cache.kv_cache[0][0].shape) == (4, config.num_key_value_heads, 4, config.head_dim)
    for state, prompt in zip(states, prompts):
        assert_slot_matches_single_request(dense_slot_cache, model, state, prompt)
    # 未使用的槽位整行被 mask
    assert dense_slot_cache.history_mask[0].tolist() == [False] * 4
    assert dense_slot_cache.history_mask[2].tolist() == [False] * 4


def test_absorb_prefill_longer_prompt_grows_cache_to_the_left_and_preserves_existing_slots(model):
    first_prompt, longer_prompt = [3, 5], [11, 13, 17, 19, 23]
    first = admitted_state(0, first_prompt, slot=0)
    dense_slot_cache = DenseSlotCache(capacity=2, device=CPU)
    dense_slot_cache.absorb_prefill([first], prefill(model, [first_prompt]))
    assert dense_slot_cache.length == 2

    second = admitted_state(1, longer_prompt, slot=1)
    dense_slot_cache.absorb_prefill([second], prefill(model, [longer_prompt]))

    assert dense_slot_cache.length == 5
    assert_slot_matches_single_request(dense_slot_cache, model, first, first_prompt)
    assert_slot_matches_single_request(dense_slot_cache, model, second, longer_prompt)


def test_absorb_prefill_shorter_prompt_keeps_cache_length(model):
    long_prompt, short_prompt = [3, 5, 7, 9, 11], [13, 17]
    dense_slot_cache = DenseSlotCache(capacity=2, device=CPU)
    dense_slot_cache.absorb_prefill([admitted_state(0, long_prompt, 0)], prefill(model, [long_prompt]))
    short = admitted_state(1, short_prompt, 1)
    dense_slot_cache.absorb_prefill([short], prefill(model, [short_prompt]))

    assert dense_slot_cache.length == 5
    assert_slot_matches_single_request(dense_slot_cache, model, short, short_prompt)


def test_prepare_decode_inputs_masks_inactive_slots_and_appends_active_column(model):
    prompt = [3, 5, 7]
    state = admitted_state(0, prompt, slot=2)
    state.generated.append(42)
    dense_slot_cache = DenseSlotCache(capacity=3, device=CPU)
    dense_slot_cache.absorb_prefill([state], prefill(model, [prompt]))

    input_ids, position_ids, attention_mask, prepare_ms = dense_slot_cache.prepare_decode_inputs(
        [state], pad_token_id=0
    )

    assert prepare_ms >= 0
    assert input_ids.tolist() == [[0], [0], [42]]
    assert position_ids.tolist() == [[0], [0], [3]]  # 下一个 Token 的位置 == 已缓存长度
    assert attention_mask.shape == (3, dense_slot_cache.length + 1)
    assert attention_mask[:, -1].tolist() == [False, False, True]
    assert attention_mask[2].tolist() == [True, True, True, True]
    assert not attention_mask[0].any() and not attention_mask[1].any()


def test_commit_decode_step_updates_history_mask(model):
    prompt = [3, 5, 7]
    state = admitted_state(0, prompt, slot=2)
    state.generated.append(42)
    dense_slot_cache = DenseSlotCache(capacity=3, device=CPU)
    dense_slot_cache.absorb_prefill([state], prefill(model, [prompt]))
    assert dense_slot_cache.history_mask.shape == (3, 3)

    input_ids, position_ids, attention_mask, _ = dense_slot_cache.prepare_decode_inputs([state], pad_token_id=0)
    with torch.inference_mode():
        _, present = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=dense_slot_cache.past_key_values,
        )
    dense_slot_cache.commit_decode_step(attention_mask, present)
    assert dense_slot_cache.history_mask.shape == (3, 4)
    assert dense_slot_cache.length == 4
    torch.testing.assert_close(dense_slot_cache.history_mask, attention_mask)


def test_prepare_decode_inputs_requires_allocated_cache():
    with pytest.raises(RuntimeError):
        DenseSlotCache(capacity=2, device=CPU).prepare_decode_inputs([], pad_token_id=0)


def test_release_and_reset(model):
    prompt = [3, 5, 7]
    state = admitted_state(0, prompt, slot=0)
    dense_slot_cache = DenseSlotCache(capacity=2, device=CPU)
    dense_slot_cache.absorb_prefill([state], prefill(model, [prompt]))
    assert dense_slot_cache.total_bytes > 0

    dense_slot_cache.release(state)
    assert not dense_slot_cache.history_mask[0].any()

    dense_slot_cache.reset()
    assert dense_slot_cache.kv_cache is None and dense_slot_cache.history_mask is None
    assert dense_slot_cache.length == 0 and dense_slot_cache.total_bytes == 0


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        DenseSlotCache(capacity=0, device=CPU)


def test_rebuild_pool_zero_copy_and_reset_restores_capacity(model):
    prompts = [[3, 5, 7, 9], [11, 13]]
    states = [admitted_state(0, prompts[0], slot=0), admitted_state(1, prompts[1], slot=1)]
    dense_slot_cache = DenseSlotCache(capacity=4, device=CPU)
    prefill_cache = prefill(model, prompts)

    dense_slot_cache.rebuild_pool(states, prefill_cache)

    assert dense_slot_cache.capacity == 2
    assert dense_slot_cache.kv_cache is prefill_cache
    assert dense_slot_cache.history_mask.tolist() == [[True, True, True, True], [False, False, True, True]]

    dense_slot_cache.reset()
    assert dense_slot_cache.capacity == 4
    assert dense_slot_cache.kv_cache is None and dense_slot_cache.history_mask is None
