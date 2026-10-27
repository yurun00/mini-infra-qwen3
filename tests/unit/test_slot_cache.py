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
    cache = KVCache()
    with torch.inference_mode():
        model(input_ids, kv_cache=cache, attention_mask=mask, position_ids=positions)
    return cache


def admitted_state(index: int, prompt: list[int], slot: int) -> RequestState:
    state = RequestState(RequestSpec(index, tuple(prompt), 4))
    state.status = RequestStatus.RUNNING
    state.slot_index = slot
    state.cache_length = len(prompt)
    return state


def assert_slot_matches_single_request(slots: DenseSlotCache, model, state: RequestState, prompt: list[int]) -> None:
    """槽位中右对齐的有效历史必须与该请求单独 Prefill 的 Cache 完全一致，且左侧空位被 mask。"""
    reference = prefill(model, [prompt])
    length, total = len(prompt), slots.length
    assert slots.history_mask[state.slot_index].tolist() == [False] * (total - length) + [True] * length
    for layer, (ref_key, ref_value) in enumerate(reference):
        key, value = slots.cache[layer]
        torch.testing.assert_close(key[state.slot_index, :, total - length :, :], ref_key[0])
        torch.testing.assert_close(value[state.slot_index, :, total - length :, :], ref_value[0])


def test_admit_allocates_and_writes_right_aligned_slots(model):
    prompts = [[3, 5, 7, 9], [11, 13]]
    states = [admitted_state(0, prompts[0], slot=1), admitted_state(1, prompts[1], slot=3)]
    slots = DenseSlotCache(capacity=4, device=CPU)
    assert slots.cache is None and slots.length == 0 and slots.total_bytes == 0

    elapsed_ms = slots.admit(states, prefill(model, prompts))

    assert elapsed_ms >= 0
    assert slots.length == 4
    assert slots.cache.get_seq_length() == 4
    config = model.config
    assert tuple(slots.cache[0][0].shape) == (4, config.num_key_value_heads, 4, config.head_dim)
    for state, prompt in zip(states, prompts):
        assert_slot_matches_single_request(slots, model, state, prompt)
    # 未使用的槽位整行被 mask
    assert slots.history_mask[0].tolist() == [False] * 4
    assert slots.history_mask[2].tolist() == [False] * 4


def test_admit_longer_prompt_grows_cache_to_the_left_and_preserves_existing_slots(model):
    first_prompt, longer_prompt = [3, 5], [11, 13, 17, 19, 23]
    first = admitted_state(0, first_prompt, slot=0)
    slots = DenseSlotCache(capacity=2, device=CPU)
    slots.admit([first], prefill(model, [first_prompt]))
    assert slots.length == 2

    second = admitted_state(1, longer_prompt, slot=1)
    slots.admit([second], prefill(model, [longer_prompt]))

    assert slots.length == 5
    assert_slot_matches_single_request(slots, model, first, first_prompt)
    assert_slot_matches_single_request(slots, model, second, longer_prompt)


def test_admit_shorter_prompt_keeps_cache_length(model):
    long_prompt, short_prompt = [3, 5, 7, 9, 11], [13, 17]
    slots = DenseSlotCache(capacity=2, device=CPU)
    slots.admit([admitted_state(0, long_prompt, 0)], prefill(model, [long_prompt]))
    short = admitted_state(1, short_prompt, 1)
    slots.admit([short], prefill(model, [short_prompt]))

    assert slots.length == 5
    assert_slot_matches_single_request(slots, model, short, short_prompt)


def test_prepare_decode_inputs_masks_inactive_slots_and_appends_active_column(model):
    prompt = [3, 5, 7]
    state = admitted_state(0, prompt, slot=2)
    state.generated.append(42)
    slots = DenseSlotCache(capacity=3, device=CPU)
    slots.admit([state], prefill(model, [prompt]))

    input_ids, position_ids, attention_mask, prepare_ms = slots.prepare_decode_inputs([state], pad_token_id=0)

    assert prepare_ms >= 0
    assert input_ids.tolist() == [[0], [0], [42]]
    assert position_ids.tolist() == [[0], [0], [3]]  # 下一个 Token 的位置 == 已缓存长度
    assert attention_mask.shape == (3, slots.length + 1)
    assert attention_mask[:, -1].tolist() == [False, False, True]
    assert attention_mask[2].tolist() == [True, True, True, True]
    assert not attention_mask[0].any() and not attention_mask[1].any()


def test_commit_decode_step_updates_history_mask(model):
    prompt = [3, 5, 7]
    state = admitted_state(0, prompt, slot=2)
    state.generated.append(42)
    slots = DenseSlotCache(capacity=3, device=CPU)
    slots.admit([state], prefill(model, [prompt]))
    assert slots.history_mask.shape == (3, 3)

    _, _, attention_mask, _ = slots.prepare_decode_inputs([state], pad_token_id=0)
    slots.commit_decode_step(attention_mask)
    assert slots.history_mask.shape == (3, 4)
    torch.testing.assert_close(slots.history_mask, attention_mask)


def test_prepare_decode_inputs_requires_allocated_cache():
    with pytest.raises(RuntimeError):
        DenseSlotCache(capacity=2, device=CPU).prepare_decode_inputs([], pad_token_id=0)


def test_release_and_reset(model):
    prompt = [3, 5, 7]
    state = admitted_state(0, prompt, slot=0)
    slots = DenseSlotCache(capacity=2, device=CPU)
    slots.admit([state], prefill(model, [prompt]))
    assert slots.total_bytes > 0

    slots.release(state)
    assert not slots.history_mask[0].any()

    slots.reset()
    assert slots.cache is None and slots.history_mask is None
    assert slots.length == 0 and slots.total_bytes == 0


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        DenseSlotCache(capacity=0, device=CPU)
